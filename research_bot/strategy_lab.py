from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .backtest import backtest_positions, performance_metrics
from .ichimoku_advanced import add_ichimoku_state


@dataclass(frozen=True)
class StrategyLabConfig:
    """Frozen v0.17 research contract; all rates are fractions unless named bps."""

    timeframe: str = "4h"
    fee_bps: float = 10.0
    slippage_bps: float = 2.0
    stress_cost_bps: tuple[float, ...] = (12.0, 18.0, 30.0)
    risk_per_trade: float = 0.0025
    max_asset_weight: float = 0.35
    max_portfolio_gross: float = 0.70
    max_drawdown: float = 0.05
    cooling_bars: int = 42
    atr_window: int = 14
    breakout_window: int = 20
    exit_window: int = 10
    ema_window: int = 200
    squeeze_window: int = 252
    squeeze_quantile: float = 0.20
    cusum_threshold: float = 0.75
    meta_horizon: int = 12
    profit_atr: float = 1.5
    stop_atr: float = 1.0
    meta_probability: float = 0.55
    min_meta_train_events: int = 80


META_FEATURES = (
    "price_kijun_atr",
    "price_cloud_atr",
    "cloud_width_atr",
    "tenkan_kijun_atr",
    "kijun_slope_3_atr",
    "atr_pct",
    "volume_z_20",
    "ret_1",
    "ret_3",
    "ret_12",
    "bars_since_tk_cross",
)


def _validate_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing OHLCV columns: {sorted(missing)}")
    x = frame.copy()
    x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True, errors="coerce")
    for col in ("open", "high", "low", "close", "volume"):
        x[col] = pd.to_numeric(x[col], errors="coerce")
    x = x.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"])
    x = x.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    invalid = (x[["open", "high", "low", "close"]] <= 0).any(axis=1)
    invalid |= x["high"] < x[["open", "close", "low"]].max(axis=1)
    invalid |= x["low"] > x[["open", "close", "high"]].min(axis=1)
    if invalid.any():
        raise ValueError(f"invalid OHLC rows: {int(invalid.sum())}")
    if not x["timestamp"].is_monotonic_increasing:
        raise ValueError("timestamps must be monotonic")
    return x


def _atr(x: pd.DataFrame, window: int) -> pd.Series:
    previous = x["close"].shift(1)
    true_range = pd.concat(
        [x["high"] - x["low"], (x["high"] - previous).abs(), (x["low"] - previous).abs()],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1.0 / window, adjust=False).mean()


def _bars_since(event: pd.Series) -> pd.Series:
    last = -1
    values: list[float] = []
    for i, flag in enumerate(event.fillna(False).astype(bool)):
        if flag:
            last = i
        values.append(np.nan if last < 0 else float(i - last))
    return pd.Series(values, index=event.index)


def build_strategy_features(frame: pd.DataFrame, config: StrategyLabConfig | None = None) -> pd.DataFrame:
    """Build point-in-time features. No forward displacement is used as a feature."""

    cfg = config or StrategyLabConfig()
    x = add_ichimoku_state(_validate_ohlcv(frame))
    atr = _atr(x, cfg.atr_window)
    x["atr"] = atr
    x["atr_pct"] = atr / x["close"]
    x["ema_5"] = x["close"].ewm(span=5, adjust=False).mean()
    x["ema_200"] = x["close"].ewm(span=cfg.ema_window, adjust=False).mean()
    x["ema_200_slope"] = x["ema_200"].diff(6)
    x["kijun_slope_3"] = x["ichi_kijun"].diff(3)
    x["price_kijun_atr"] = (x["close"] - x["ichi_kijun"]) / atr
    x["price_cloud_atr"] = (x["close"] - x["ichi_cloud_top_now"]) / atr
    x["cloud_width_atr"] = (x["ichi_cloud_top_now"] - x["ichi_cloud_bottom_now"]) / atr
    x["tenkan_kijun_atr"] = (x["ichi_tenkan"] - x["ichi_kijun"]) / atr
    x["kijun_slope_3_atr"] = x["kijun_slope_3"] / atr
    x["ret_1"] = x["close"].pct_change()
    x["ret_3"] = x["close"].pct_change(3)
    x["ret_12"] = x["close"].pct_change(12)
    median_volume = x["volume"].rolling(20).median()
    mad_volume = (x["volume"] - median_volume).abs().rolling(20).median()
    x["volume_z_20"] = (x["volume"] - median_volume) / (1.4826 * mad_volume).replace(0, np.nan)
    tk_cross = (x["ichi_tenkan"] > x["ichi_kijun"]) != (
        x["ichi_tenkan"].shift(1) > x["ichi_kijun"].shift(1)
    )
    x["bars_since_tk_cross"] = _bars_since(tk_cross)
    # Causal Chikou analogue: current price compared with the known price 26 bars ago.
    x["chikou_causal_positive"] = (x["close"] > x["close"].shift(26)).astype(float)
    log_return = np.log(x["close"]).diff().fillna(0.0)
    normalized = log_return / x["atr_pct"].replace(0, np.nan)
    pos_sum = neg_sum = 0.0
    events = np.zeros(len(x), dtype=bool)
    for i, value in enumerate(normalized.fillna(0.0).to_numpy(float)):
        pos_sum = max(0.0, pos_sum + value)
        neg_sum = min(0.0, neg_sum + value)
        if pos_sum >= cfg.cusum_threshold or neg_sum <= -cfg.cusum_threshold:
            events[i] = True
            pos_sum = neg_sum = 0.0
    x["cusum_event"] = events.astype(float)
    return x.replace([np.inf, -np.inf], np.nan)


def _stateful_position(entry: pd.Series, exit_: pd.Series) -> pd.Series:
    state = 0.0
    out = np.zeros(len(entry), dtype=float)
    for i, (enter, leave) in enumerate(zip(entry.fillna(False), exit_.fillna(False))):
        if state and bool(leave):
            state = 0.0
        elif not state and bool(enter):
            state = 1.0
        out[i] = state
    return pd.Series(out, index=entry.index)


def generate_strategy_positions(features: pd.DataFrame, config: StrategyLabConfig | None = None) -> pd.DataFrame:
    """Generate the frozen B0/S1-S6/IRCP/C1/C2 primary positions."""

    cfg = config or StrategyLabConfig()
    x = features
    cloud_top = x["ichi_cloud_top_now"]
    cloud_bottom = x["ichi_cloud_bottom_now"]
    tenkan = x["ichi_tenkan"]
    kijun = x["ichi_kijun"]
    close = x["close"]
    bull_regime = (close > cloud_top) & (tenkan > kijun) & (x["kijun_slope_3"] >= 0)
    cross_up = (tenkan > kijun) & (tenkan.shift(1) <= kijun.shift(1))
    cross_down = (tenkan < kijun) & (tenkan.shift(1) >= kijun.shift(1))

    s1 = _stateful_position(cross_up, cross_down)
    s2_entry = (close > cloud_top) & (close.shift(1) <= cloud_top.shift(1))
    s2 = _stateful_position(s2_entry, close < kijun)
    s3 = bull_regime.astype(float)
    pullback_touch = (x["low"] <= tenkan) & (x["low"] >= kijun - 0.5 * x["atr"])
    pullback_confirm = pullback_touch & bull_regime & (close > x["open"])
    s4 = _stateful_position(pullback_confirm, (close < kijun) | ~bull_regime)
    near_support = (
        ((x["low"] - kijun).abs() <= 0.5 * x["atr"])
        | ((x["low"] - cloud_top).abs() <= 0.5 * x["atr"])
    )
    rejection = close > x["open"]
    s5 = _stateful_position(bull_regime & near_support & rejection, close < kijun - x["atr"])
    prior_high = x["high"].shift(1).rolling(cfg.breakout_window).max()
    prior_low_exit = x["low"].shift(1).rolling(cfg.exit_window).min()
    s6_entry = (close > prior_high) & (x["ema_200_slope"] > 0)
    s6 = _stateful_position(s6_entry, close < prior_low_exit)
    ircp = _stateful_position(pullback_confirm & (x["chikou_causal_positive"] > 0), (close < kijun) | ~bull_regime)

    down_three = (close < close.shift(1)) & (close.shift(1) < close.shift(2)) & (close.shift(2) < close.shift(3))
    c1_entry = down_three & (close > x["ema_200"]) & (x["ema_200_slope"] > 0)
    c1 = _stateful_position(c1_entry, (close > x["ema_5"]) | (close < x["ema_200"]))

    rolling_mean = close.rolling(20).mean()
    rolling_std = close.rolling(20).std()
    band_width = 4.0 * rolling_std / rolling_mean
    squeeze_level = band_width.shift(1).rolling(cfg.squeeze_window).quantile(cfg.squeeze_quantile)
    volume_median = x["volume"].shift(1).rolling(20).median()
    c2_entry = (band_width.shift(1) <= squeeze_level) & (close > prior_high) & (x["volume"] > volume_median)
    c2 = _stateful_position(c2_entry, close < kijun)

    return pd.DataFrame(
        {
            "B0": 1.0,
            "S1": s1,
            "S2": s2,
            "S3": s3,
            "S4": s4,
            "S5": s5,
            "S6": s6,
            "IRCP": ircp,
            "C1": c1,
            "C2": c2,
        },
        index=x.index,
    ).fillna(0.0)


def _apply_drawdown_guard(
    desired: pd.Series,
    future_open_return: pd.Series,
    cfg: StrategyLabConfig,
) -> tuple[pd.Series, int]:
    """Causal kill switch with a fixed cooling period; never reads future equity."""

    desired = desired.fillna(0.0).clip(0.0, 1.0)
    realized = np.zeros(len(desired), dtype=float)
    equity = peak = 1.0
    previous = 0.0
    killed = False
    activations = 0
    friction_rate = (cfg.fee_bps + cfg.slippage_bps) / 10_000.0
    for i, want in enumerate(desired.to_numpy(float)):
        position = 0.0 if killed else want
        r = float(future_open_return.iloc[i]) if np.isfinite(future_open_return.iloc[i]) else 0.0
        net = position * r - abs(position - previous) * friction_rate
        realized[i] = position
        equity *= 1.0 + net
        peak = max(peak, equity)
        if not killed and 1.0 - equity / peak >= cfg.max_drawdown:
            # Fail closed for the remainder of an experiment. A production
            # reset would require an explicit, externally audited decision.
            killed = True
            activations += 1
        previous = position
    return pd.Series(realized, index=desired.index), activations


def evaluate_rule_strategies(
    frame: pd.DataFrame,
    config: StrategyLabConfig | None = None,
    *,
    evaluation_start: str | None = None,
    evaluation_end: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = config or StrategyLabConfig()
    features = build_strategy_features(frame, cfg)
    positions = generate_strategy_positions(features, cfg)
    # Decision at close[t], fill at open[t+1], return from open[t+1] to open[t+2].
    future_open_return = features["open"].shift(-2) / features["open"].shift(-1) - 1.0
    eval_mask = pd.Series(True, index=features.index)
    if evaluation_start:
        eval_mask &= features["timestamp"] >= pd.Timestamp(evaluation_start, tz="UTC")
    if evaluation_end:
        eval_mask &= features["timestamp"] <= pd.Timestamp(evaluation_end, tz="UTC")
    eval_index = features.index[eval_mask]
    if len(eval_index) < 2:
        raise ValueError("evaluation window contains fewer than two bars")
    rows: list[dict] = []
    guarded = pd.DataFrame(index=eval_index)
    for name in positions:
        if name == "B0":
            position = positions.loc[eval_index, name]
            kills = 0
        else:
            risk_weight = (
                cfg.risk_per_trade / (cfg.stop_atr * features["atr_pct"].replace(0, np.nan))
            ).clip(lower=0.0, upper=cfg.max_asset_weight).fillna(0.0)
            desired = (positions[name] * risk_weight).loc[eval_index]
            position, kills = _apply_drawdown_guard(desired, future_open_return.loc[eval_index], cfg)
        guarded[name] = position
        net, metrics = backtest_positions(
            future_open_return.loc[eval_index].fillna(0.0), position,
            timeframe=cfg.timeframe, fee_bps=cfg.fee_bps, slippage_bps=cfg.slippage_bps,
        )
        row = {"strategy": name, "kill_switch_activations": kills, **metrics}
        rows.append(row)
        features[f"net_{name}"] = np.nan
        features.loc[eval_index, f"net_{name}"] = net.to_numpy()
    return pd.DataFrame(rows).sort_values("sharpe", ascending=False), guarded, features


def build_meta_events(features: pd.DataFrame, config: StrategyLabConfig | None = None) -> pd.DataFrame:
    """Create causal primary events and future-only labels for supervised training."""

    cfg = config or StrategyLabConfig()
    positions = generate_strategy_positions(features, cfg)
    primary_entry = (positions["IRCP"].diff().fillna(positions["IRCP"]) > 0) | (
        positions["C2"].diff().fillna(positions["C2"]) > 0
    )
    regime = (
        (features["close"] > features["ichi_cloud_top_now"])
        & (features["ichi_tenkan"] > features["ichi_kijun"])
        & (features["kijun_slope_3"] >= 0)
        & features["chikou_causal_positive"].eq(1.0)
    )
    candidate = primary_entry & regime & features["cusum_event"].eq(1.0)
    rows: list[dict] = []
    for i in np.flatnonzero(candidate.to_numpy()):
        entry_i = i + 1
        end_i = min(len(features) - 1, entry_i + cfg.meta_horizon - 1)
        if entry_i >= len(features) or end_i <= entry_i:
            continue
        entry = float(features.loc[entry_i, "open"])
        atr = float(features.loc[i, "atr"])
        if not np.isfinite(entry) or not np.isfinite(atr) or atr <= 0:
            continue
        upper = entry + cfg.profit_atr * atr
        lower = entry - cfg.stop_atr * atr
        label = 0
        exit_i = end_i
        exit_price = float(features.loc[end_i, "close"])
        outcome = "TIME"
        for j in range(entry_i, end_i + 1):
            hit_stop = float(features.loc[j, "low"]) <= lower
            hit_profit = float(features.loc[j, "high"]) >= upper
            if hit_stop:  # conservative ordering if both barriers occur in one bar
                exit_i, exit_price, outcome = j, lower, "STOP"
                break
            if hit_profit:
                label, exit_i, exit_price, outcome = 1, j, upper, "PROFIT"
                break
        gross = exit_price / entry - 1.0
        row = {
            "event_index": int(i), "entry_index": int(entry_i), "exit_index": int(exit_i),
            "timestamp": features.loc[i, "timestamp"], "exit_time": features.loc[exit_i, "timestamp"],
            "label": int(label), "gross_return": float(gross), "outcome": outcome,
        }
        row.update({name: features.loc[i, name] for name in META_FEATURES})
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_irgc_s(
    symbol_features: Mapping[str, pd.DataFrame],
    config: StrategyLabConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Expanding walk-forward Logistic meta-label evaluation across symbols."""

    cfg = config or StrategyLabConfig()
    events = []
    for symbol, features in symbol_features.items():
        e = build_meta_events(features, cfg)
        if not e.empty:
            e["symbol"] = symbol
            events.append(e)
    if not events:
        return pd.DataFrame(), pd.DataFrame(), {"decision": "REJECT_NO_EVENTS"}
    all_events = pd.concat(events, ignore_index=True).sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    times = np.array(sorted(pd.unique(all_events["timestamp"])))
    if len(times) < 8:
        return all_events, pd.DataFrame(), {"decision": "REJECT_TOO_FEW_EVENT_TIMES"}
    boundaries = [int(len(times) * p) for p in (0.60, 0.75, 0.875, 1.0)]
    predictions: list[pd.DataFrame] = []
    start = boundaries[0]
    for fold, stop in enumerate(boundaries[1:], 1):
        train_end = times[start - 1]
        test_times = times[start:stop]
        train = all_events[all_events["exit_time"] < train_end].copy()
        test = all_events[all_events["timestamp"].isin(test_times)].copy()
        if len(train) < cfg.min_meta_train_events or test.empty or train["label"].nunique() < 2:
            start = stop
            continue
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.25, max_iter=1500, class_weight="balanced", random_state=314)),
        ])
        model.fit(train[list(META_FEATURES)], train["label"])
        test["probability"] = model.predict_proba(test[list(META_FEATURES)])[:, 1]
        test["selected"] = test["probability"] >= cfg.meta_probability
        test["fold"] = fold
        predictions.append(test)
        start = stop
    if not predictions:
        return all_events, pd.DataFrame(), {"decision": "REJECT_INSUFFICIENT_TRAINING_EVENTS"}
    pred = pd.concat(predictions, ignore_index=True).sort_values(["timestamp", "symbol"])
    cost = 2.0 * (cfg.fee_bps + cfg.slippage_bps) / 10_000.0
    selected = pred[pred["selected"]].copy()
    selected["weight"] = (
        cfg.risk_per_trade / (cfg.stop_atr * selected["atr_pct"].replace(0, np.nan))
    ).clip(lower=0.0, upper=cfg.max_asset_weight).fillna(0.0)
    selected["net_return"] = selected["weight"] * (selected["gross_return"] - cost)
    baseline = pred.copy()
    baseline["weight"] = (
        cfg.risk_per_trade / (cfg.stop_atr * baseline["atr_pct"].replace(0, np.nan))
    ).clip(lower=0.0, upper=cfg.max_asset_weight).fillna(0.0)
    baseline["net_return"] = baseline["weight"] * (baseline["gross_return"] - cost)
    metrics = _trade_metrics(selected["net_return"]) if not selected.empty else {"n": 0}
    baseline_metrics = _trade_metrics(baseline["net_return"])
    positive_folds = int((selected.groupby("fold")["net_return"].sum() > 0).sum()) if not selected.empty else 0
    decision = "PROMOTE_TO_PAPER" if (
        metrics.get("n", 0) >= 30
        and (metrics.get("total_return") or -1.0) > 0
        and (metrics.get("mean_trade_return") or -99.0) > (baseline_metrics.get("mean_trade_return") or -99.0)
        and positive_folds >= 2
    ) else "REJECT_OR_CONTINUE_RESEARCH"
    summary = {
        "strategy": "IRGC-S",
        "decision": decision,
        "config": asdict(cfg),
        "events_total": int(len(all_events)),
        "events_oos": int(len(pred)),
        "events_selected": int(len(selected)),
        "positive_oos_folds": positive_folds,
        "metrics_selected": metrics,
        "metrics_unfiltered_primary": baseline_metrics,
        "live_execution": False,
    }
    return all_events, pred, summary


def _trade_metrics(returns: pd.Series) -> dict:
    """Metrics for irregular event returns; deliberately avoids time-bar annualisation."""

    r = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if r.empty:
        return {"n": 0}
    equity = (1.0 + r).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    gain = float(r[r > 0].sum())
    loss = float(-r[r < 0].sum())
    return {
        "n": int(len(r)),
        "total_return": float(equity.iloc[-1] - 1.0),
        "mean_trade_return": float(r.mean()),
        "median_trade_return": float(r.median()),
        "win_rate": float((r > 0).mean()),
        "profit_factor": float(gain / loss) if loss > 0 else None,
        "max_drawdown": float(drawdown.min()),
    }


def cost_sensitivity(
    frame: pd.DataFrame,
    positions: pd.DataFrame,
    config: StrategyLabConfig | None = None,
) -> pd.DataFrame:
    cfg = config or StrategyLabConfig()
    future_open_return = frame["open"].shift(-2) / frame["open"].shift(-1) - 1.0
    rows = []
    for total_one_way_bps in cfg.stress_cost_bps:
        for name in positions:
            _, metrics = backtest_positions(
                future_open_return.fillna(0.0), positions[name], timeframe=cfg.timeframe,
                fee_bps=total_one_way_bps, slippage_bps=0.0,
            )
            rows.append({"strategy": name, "one_way_cost_bps": total_one_way_bps, **metrics})
    return pd.DataFrame(rows)
