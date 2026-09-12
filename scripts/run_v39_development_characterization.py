from __future__ import annotations

"""Frozen v0.39 development characterization.

Consumes CoinEx / OKX / KuCoin only. Kraken is never instantiated or fetched.
The script implements the preregistered causal mother-strategy learning ladder:
Ridge -> HistGradientBoosting -> shallow MLP. The first / simplest model passing
all frozen gates is the development winner. PAPER/LIVE remain disabled.
"""

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import ccxt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

from research_bot.financial_system_v39 import (
    DEVELOPMENT_VENUES_V39,
    FinancialRiskPolicyV39,
    LearningPolicyV39,
    ValidationPolicyV39,
    allocate_portfolio_risk,
    causal_robust_normalize,
)
from research_bot.mother_strategy_v39 import (
    NEURAL_FEATURES_V39,
    MotherStrategyPolicyV39,
    _structural_stop_fraction,
    build_mother_features_v39,
)

VENUES = tuple(DEVELOPMENT_VENUES_V39)
SYMBOLS = ("BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT")
TIMEFRAME = "4h"
START = "2025-01-01T00:00:00Z"
END = "2026-09-10T23:59:59Z"
STEP = pd.Timedelta(hours=4)
BASE_ROUNDTRIP_BPS = 24.0
STRESS_ROUNDTRIP_BPS = 36.0
TARGET_R = 3.0
MAX_HOLD_BARS = 30
EMBARGO_BARS = 30
PURGED_FOLDS = 5
BOOTSTRAP_SAMPLES = 750
BOOTSTRAP_BLOCK = 20
SEEDS = (314, 1618, 2718)
CANDIDATES = ("V39_RIDGE", "V39_HISTGB", "V39_SHALLOW_MLP")


def _to_ms(ts: str) -> int:
    return int(pd.Timestamp(ts).timestamp() * 1000)


def _exchange(name: str):
    if name == "kraken":
        raise RuntimeError("Kraken is sealed and forbidden in v0.39 development")
    cls = getattr(ccxt, name)
    return cls({"enableRateLimit": True, "timeout": 30_000})


def fetch_fixed_ohlcv(exchange, symbol: str) -> pd.DataFrame:
    start_ms, end_ms = _to_ms(START), _to_ms(END)
    step_ms = int(STEP.total_seconds() * 1000)
    since = start_ms
    rows: list[list[float]] = []
    stalls = 0
    while since <= end_ms:
        batch = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, since=since, limit=1000)
        if not batch:
            break
        rows.extend(batch)
        last = int(batch[-1][0])
        nxt = last + step_ms
        if nxt <= since:
            stalls += 1
            if stalls >= 2:
                raise RuntimeError(f"pagination stalled: {exchange.id} {symbol}")
            nxt = since + step_ms
        else:
            stalls = 0
        since = nxt
        if last >= end_ms:
            break
        time.sleep(max(float(getattr(exchange, "rateLimit", 0)) / 1000.0, 0.01))

    if not rows:
        raise RuntimeError(f"no OHLCV: {exchange.id} {symbol}")
    x = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    x["timestamp"] = pd.to_datetime(x["timestamp"], unit="ms", utc=True)
    for c in ("open", "high", "low", "close", "volume"):
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.dropna().drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    x = x[(x["timestamp"] >= pd.Timestamp(START)) & (x["timestamp"] <= pd.Timestamp(END))].reset_index(drop=True)
    if len(x) < 300:
        raise RuntimeError(f"insufficient bars: {exchange.id} {symbol}: {len(x)}")
    return x


def _event_labels(features: pd.DataFrame, venue: str, symbol: str, cfg: MotherStrategyPolicyV39) -> pd.DataFrame:
    rows: list[dict] = []
    n = len(features)
    for t in range(n - 1):
        if int(features.at[t, "mother_event_v39"]) != 1:
            continue
        side = int(features.at[t, "research_candidate_side_v39"])
        if side not in (-1, 1):
            continue
        entry_i = t + 1
        entry = float(features.at[entry_i, "open"])
        if not np.isfinite(entry) or entry <= 0:
            continue
        stop_fraction = _structural_stop_fraction(features.loc[t], side, cfg)
        if not np.isfinite(stop_fraction) or stop_fraction <= 0:
            continue
        stop_dist = float(entry * stop_fraction)
        stop = entry - side * stop_dist
        target = entry + side * TARGET_R * stop_dist
        last_i = min(n - 1, entry_i + MAX_HOLD_BARS - 1)
        exit_i = last_i
        exit_price = float(features.at[last_i, "close"])
        gross_r = float(side * (exit_price - entry) / stop_dist)
        outcome = "TIME"
        for j in range(entry_i, last_i + 1):
            low, high = float(features.at[j, "low"]), float(features.at[j, "high"])
            hit_stop = low <= stop if side > 0 else high >= stop
            hit_target = high >= target if side > 0 else low <= target
            if hit_stop:
                exit_i, exit_price, gross_r, outcome = j, stop, -1.0, "STOP"
                break
            if hit_target:
                exit_i, exit_price, gross_r, outcome = j, target, TARGET_R, "TARGET"
                break
        base_cost_r = (BASE_ROUNDTRIP_BPS / 10_000.0) / stop_fraction
        stress_cost_r = (STRESS_ROUNDTRIP_BPS / 10_000.0) / stop_fraction
        trend = int(features.at[t, "brooks_trend_regime_v39"])
        range_ = int(features.at[t, "brooks_range_regime_v39"])
        regime = "trend" if trend else ("range" if range_ else "transition")
        rows.append({
            "series_id": f"{venue}::{symbol}",
            "venue": venue,
            "symbol": symbol,
            "signal_time": features.at[t, "timestamp"],
            "entry_time": features.at[entry_i, "timestamp"],
            "exit_time": features.at[exit_i, "timestamp"],
            "side": side,
            "entry": entry,
            "stop_fraction": stop_fraction,
            "outcome": outcome,
            "gross_r": gross_r,
            "base_cost_r": float(base_cost_r),
            "stress_cost_r": float(stress_cost_r),
            "net_r": float(gross_r - base_cost_r),
            "stress_net_r": float(gross_r - stress_cost_r),
            "regime": regime,
        })
    return pd.DataFrame(rows)


def _prepare_panel() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_panels: list[pd.DataFrame] = []
    events: list[pd.DataFrame] = []
    manifest: list[dict] = []
    cfg = MotherStrategyPolicyV39()
    for venue in VENUES:
        ex = _exchange(venue)
        markets = ex.load_markets()
        missing = [s for s in SYMBOLS if s not in markets]
        if missing:
            raise RuntimeError(f"required symbols missing on {venue}: {missing}")
        for symbol in SYMBOLS:
            frame = fetch_fixed_ohlcv(ex, symbol)
            f = build_mother_features_v39(frame, cfg).copy()
            f["venue"] = venue
            f["symbol"] = symbol
            f["series_id"] = f"{venue}::{symbol}"
            feature_panels.append(f)
            e = _event_labels(f, venue, symbol, cfg)
            if not e.empty:
                events.append(e)
            manifest.append({
                "venue": venue,
                "symbol": symbol,
                "timeframe": TIMEFRAME,
                "requested_start": START,
                "requested_end": END,
                "first_bar": frame["timestamp"].iloc[0],
                "last_bar": frame["timestamp"].iloc[-1],
                "bars": int(len(frame)),
                "mother_events": int(f["mother_event_v39"].sum()),
                "candidate_long": int((f["research_candidate_side_v39"] == 1).sum()),
                "candidate_short": int((f["research_candidate_side_v39"] == -1).sum()),
            })
    panel = pd.concat(feature_panels, ignore_index=True, sort=False)
    event_frame = pd.concat(events, ignore_index=True, sort=False) if events else pd.DataFrame()
    normalized = causal_robust_normalize(
        panel,
        NEURAL_FEATURES_V39,
        timestamp_col="timestamp",
        group_col="series_id",
    )
    normalized = normalized.rename(columns={"timestamp": "signal_time"})
    feature_cols: list[str] = []
    for c in NEURAL_FEATURES_V39:
        feature_cols.extend([c, f"{c}__missing"])
    event_frame = event_frame.merge(
        normalized[["signal_time", "series_id", *feature_cols]],
        on=["signal_time", "series_id"],
        how="left",
        validate="many_to_one",
    )
    event_frame = event_frame.replace([np.inf, -np.inf], np.nan)
    event_frame[feature_cols] = event_frame[feature_cols].fillna(0.0).astype("float32")
    return panel, event_frame, pd.DataFrame(manifest)


def _folds(events: pd.DataFrame) -> list[dict]:
    times = pd.Index(sorted(pd.to_datetime(events["signal_time"], utc=True).unique()))
    if len(times) < 100:
        raise RuntimeError("too few unique event timestamps for five purged folds")
    first_test = max(1, int(math.floor(0.40 * len(times))))
    chunks = [c for c in np.array_split(np.arange(first_test, len(times)), PURGED_FOLDS) if len(c)]
    out: list[dict] = []
    for fold_i, idx in enumerate(chunks, start=1):
        test_start = pd.Timestamp(times[int(idx[0])])
        test_end = pd.Timestamp(times[int(idx[-1])])
        pretest_cut = test_start - EMBARGO_BARS * STEP
        history_times = times[times < pretest_cut]
        if len(history_times) < 50:
            continue
        cal_i = max(1, int(math.floor(0.80 * len(history_times))))
        cal_start = pd.Timestamp(history_times[cal_i])
        fit = events[(events["signal_time"] < cal_start) & (events["exit_time"] < cal_start)].copy()
        cal = events[(events["signal_time"] >= cal_start) & (events["signal_time"] < pretest_cut) & (events["exit_time"] < test_start)].copy()
        test = events[(events["signal_time"] >= test_start) & (events["signal_time"] <= test_end)].copy()
        if len(fit) < 200 or len(cal) < 80 or len(test) < 20:
            continue
        out.append({"fold": fold_i, "fit": fit, "cal": cal, "test": test,
                    "cal_start": cal_start, "test_start": test_start, "test_end": test_end})
    if len(out) < PURGED_FOLDS:
        raise RuntimeError(f"insufficient valid purged folds: {len(out)} < {PURGED_FOLDS}")
    return out


def _fit_one(candidate: str, X: np.ndarray, y: np.ndarray, seed: int):
    if candidate == "V39_RIDGE":
        model = Ridge(alpha=10.0)
    elif candidate == "V39_HISTGB":
        model = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=180,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=314,
        )
    elif candidate == "V39_SHALLOW_MLP":
        model = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=3e-4,
            max_iter=80,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=8,
            random_state=seed,
        )
    else:
        raise ValueError(candidate)
    model.fit(X, y)
    return model


def _higher_quantile(a: np.ndarray, q: float) -> float:
    return float(np.quantile(np.asarray(a, dtype=float), q, method="higher"))


def _predict_fold(candidate: str, fold: dict, feature_cols: list[str]) -> tuple[pd.DataFrame, list[float], dict]:
    fit, cal, test = fold["fit"], fold["cal"], fold["test"]
    Xfit = fit[feature_cols].to_numpy(dtype=np.float32)
    Xcal = cal[feature_cols].to_numpy(dtype=np.float32)
    Xtest = test[feature_cols].to_numpy(dtype=np.float32)
    yfit = fit["net_r"].to_numpy(dtype=float)
    ycal = cal["net_r"].to_numpy(dtype=float)

    cal_by_seed: list[np.ndarray] = []
    test_by_seed: list[np.ndarray] = []
    use_seeds = SEEDS if candidate == "V39_SHALLOW_MLP" else (314, 314, 314)
    for seed in use_seeds:
        model = _fit_one(candidate, Xfit, yfit, seed)
        cal_by_seed.append(np.asarray(model.predict(Xcal), dtype=float))
        test_by_seed.append(np.asarray(model.predict(Xtest), dtype=float))

    cal_median = np.median(np.stack(cal_by_seed), axis=0)
    test_median = np.median(np.stack(test_by_seed), axis=0)
    lower_buffer = _higher_quantile(cal_median - ycal, 0.80)
    upper_buffer = _higher_quantile(ycal - cal_median, 0.80)

    out = test.copy()
    out["candidate"] = candidate
    out["fold"] = int(fold["fold"])
    out["expected_r"] = test_median
    out["lower_expected_r"] = test_median - lower_buffer
    out["upper_expected_r"] = test_median + upper_buffer
    out["selected_model"] = (out["expected_r"] > 0.0) & (out["lower_expected_r"] > 0.0)

    seed_expectancies: list[float] = []
    for cal_pred, test_pred in zip(cal_by_seed, test_by_seed):
        q = _higher_quantile(cal_pred - ycal, 0.80)
        sel = (test_pred > 0.0) & ((test_pred - q) > 0.0)
        seed_expectancies.append(float(np.mean(out.loc[sel, "net_r"])) if bool(np.any(sel)) else float("nan"))

    diag = {
        "candidate": candidate,
        "fold": int(fold["fold"]),
        "fit_events": int(len(fit)),
        "calibration_events": int(len(cal)),
        "test_events": int(len(test)),
        "selected_events": int(out["selected_model"].sum()),
        "calibration_start": fold["cal_start"],
        "test_start": fold["test_start"],
        "test_end": fold["test_end"],
        "lower_buffer_r": lower_buffer,
        "upper_buffer_r": upper_buffer,
        "selected_expectancy_r": float(out.loc[out["selected_model"], "net_r"].mean()) if out["selected_model"].any() else None,
    }
    return out, seed_expectancies, diag


def _profit_factor(values: pd.Series) -> float | None:
    x = pd.Series(values, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if x.empty:
        return None
    gains, losses = float(x[x > 0].sum()), float(-x[x < 0].sum())
    return (float("inf") if gains > 0 else None) if losses <= 0 else gains / losses


def _block_ci_low(values: pd.Series) -> float | None:
    x = pd.Series(values, dtype=float).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    n = len(x)
    if n < 20:
        return None
    block = min(BOOTSTRAP_BLOCK, n)
    rng = np.random.default_rng(314)
    means = np.empty(BOOTSTRAP_SAMPLES, dtype=float)
    for b in range(BOOTSTRAP_SAMPLES):
        sampled: list[float] = []
        while len(sampled) < n:
            s = int(rng.integers(0, n))
            sampled.extend(x[(s + np.arange(block)) % n].tolist())
        means[b] = float(np.mean(sampled[:n]))
    return float(np.quantile(means, 0.025))


def _realize_nonoverlap(selected: pd.DataFrame) -> pd.DataFrame:
    if selected.empty:
        return selected.copy()
    rows: list[pd.Series] = []
    for (venue, symbol), group in selected.groupby(["venue", "symbol"], sort=False):
        blocked_until = pd.Timestamp.min.tz_localize("UTC")
        for _, row in group.sort_values(["entry_time", "signal_time"]).iterrows():
            if row["entry_time"] <= blocked_until:
                continue
            rows.append(row)
            blocked_until = row["exit_time"]
    return pd.DataFrame(rows).reset_index(drop=True) if rows else selected.iloc[0:0].copy()


def _financial_simulation(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades.copy()
    p = FinancialRiskPolicyV39()
    x = trades.sort_values(["entry_time", "venue", "symbol"]).copy()
    x["uncertainty_width_r"] = (x["upper_expected_r"] - x["lower_expected_r"]).clip(lower=0.0)
    x["allocated_risk_fraction"] = 0.0
    x["position_weight"] = 0.0
    x["equity_at_entry"] = np.nan
    x["equity_after_exit"] = np.nan
    x["account_return"] = 0.0
    x["executed"] = False

    equity = peak = 1.0
    pending: list[dict] = []
    loss_streak = 0
    cooldown_until: pd.Timestamp | None = None
    entries_by_day: dict[object, int] = {}

    def settle_before(cutoff: pd.Timestamp) -> None:
        nonlocal equity, peak, pending, loss_streak, cooldown_until
        due = sorted([q for q in pending if q["exit_time"] < cutoff], key=lambda q: q["exit_time"])
        keep = [q for q in pending if q["exit_time"] >= cutoff]
        for q in due:
            pnl = q["risk_cash"] * q["net_r"]
            equity += pnl
            peak = max(peak, equity)
            if q["net_r"] < 0:
                loss_streak += 1
                if loss_streak >= 3:
                    cooldown_until = q["exit_time"] + pd.Timedelta(days=2)
            else:
                loss_streak = 0
            x.at[q["index"], "equity_after_exit"] = equity
        pending = keep

    for entry_time, batch in x.groupby("entry_time", sort=True):
        entry_time = pd.Timestamp(entry_time)
        settle_before(entry_time)
        if cooldown_until is not None and entry_time < cooldown_until:
            continue
        day = entry_time.date()
        allowed = max(0, 4 - entries_by_day.get(day, 0))
        if allowed <= 0:
            continue
        batch = batch.sort_values(["lower_expected_r", "expected_r"], ascending=False).head(allowed)
        open_total_cash = sum(q["risk_cash"] for q in pending)
        open_long_cash = sum(q["risk_cash"] for q in pending if q["side"] > 0)
        open_short_cash = sum(q["risk_cash"] for q in pending if q["side"] < 0)
        denom = max(equity, 1e-12)
        alloc = allocate_portfolio_risk(
            batch[["symbol", "side", "lower_expected_r", "uncertainty_width_r", "stop_fraction"]],
            equity=equity,
            peak=peak,
            open_total_risk_fraction=open_total_cash / denom,
            open_long_risk_fraction=open_long_cash / denom,
            open_short_risk_fraction=open_short_cash / denom,
            policy=p,
        )
        for pos, (idx, row) in enumerate(batch.iterrows()):
            risk_fraction = float(alloc.iloc[pos]["allocated_risk_fraction"])
            weight = float(alloc.iloc[pos]["position_weight"])
            if risk_fraction <= 0:
                continue
            risk_cash = equity * risk_fraction
            x.at[idx, "allocated_risk_fraction"] = risk_fraction
            x.at[idx, "position_weight"] = weight
            x.at[idx, "equity_at_entry"] = equity
            x.at[idx, "account_return"] = risk_fraction * float(row["net_r"])
            x.at[idx, "executed"] = True
            pending.append({"index": idx, "exit_time": row["exit_time"], "side": int(row["side"]),
                            "risk_cash": risk_cash, "net_r": float(row["net_r"])})
            entries_by_day[day] = entries_by_day.get(day, 0) + 1

    settle_before(pd.Timestamp.max.tz_localize("UTC"))
    return x.reset_index(drop=True)


def _venue_metrics(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"n": 0, "profit_factor": None, "expectancy_r": None, "positive_asset_fraction": 0.0,
                "block_ci_low": None, "positive_quarter_fraction": 0.0, "stress_profit_factor": None,
                "max_account_drawdown": None}
    x = trades.sort_values(["exit_time", "symbol"]).copy()
    asset = x.groupby("symbol")["net_r"].mean()
    breadth = sum(float(asset.get(s, -np.inf)) > 0 for s in SYMBOLS) / len(SYMBOLS)
    quarters = pd.to_datetime(x["exit_time"], utc=True).dt.to_period("Q")
    q = x.groupby(quarters)["net_r"].sum()
    executed = x[x.get("executed", True).astype(bool)].copy() if "executed" in x else x
    if "equity_after_exit" in executed and executed["equity_after_exit"].notna().any():
        eq = executed.sort_values("exit_time")["equity_after_exit"].dropna().to_numpy(dtype=float)
        peak = np.maximum.accumulate(eq) if len(eq) else np.array([])
        max_dd = float(np.min(eq / peak - 1.0)) if len(eq) else None
    else:
        max_dd = None
    return {
        "n": int(len(x)),
        "profit_factor": _profit_factor(x["net_r"]),
        "expectancy_r": float(x["net_r"].mean()),
        "positive_asset_fraction": float(breadth),
        "block_ci_low": _block_ci_low(x["net_r"]),
        "positive_quarter_fraction": float((q > 0).mean()) if len(q) else 0.0,
        "stress_profit_factor": _profit_factor(x["stress_net_r"]),
        "max_account_drawdown": max_dd,
    }


def _venue_gate(m: dict) -> dict:
    def ge(v, t): return v is not None and (math.isinf(float(v)) or float(v) >= t)
    def gt(v, t): return v is not None and np.isfinite(float(v)) and float(v) > t
    checks = {
        "n_ge_200": int(m["n"]) >= 200,
        "pf_ge_1_05": ge(m["profit_factor"], 1.05),
        "expectancy_gt_0": gt(m["expectancy_r"], 0.0),
        "breadth_ge_0_60": ge(m["positive_asset_fraction"], 0.60),
        "block_ci_low_gt_0": gt(m["block_ci_low"], 0.0),
        "positive_quarters_ge_0_60": ge(m["positive_quarter_fraction"], 0.60),
        "stress_pf_ge_1_00": ge(m["stress_profit_factor"], 1.00),
        "max_dd_le_0_05": m["max_account_drawdown"] is not None and float(m["max_account_drawdown"]) >= -0.05 - 1e-12,
    }
    return {**checks, "venue_pass": bool(all(checks.values()))}


def _worst_groups(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["venue", "quarter", "regime", "n", "expectancy_r", "profit_factor"])
    x = trades.copy()
    x["quarter"] = pd.to_datetime(x["exit_time"], utc=True).dt.to_period("Q").astype(str)
    rows = []
    for keys, g in x.groupby(["venue", "quarter", "regime"]):
        rows.append({"venue": keys[0], "quarter": keys[1], "regime": keys[2], "n": int(len(g)),
                     "expectancy_r": float(g["net_r"].mean()), "profit_factor": _profit_factor(g["net_r"])})
    return pd.DataFrame(rows).sort_values(["expectancy_r", "n"], ascending=[True, False])


def _jsonable(obj):
    if isinstance(obj, dict): return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list): return [_jsonable(v) for v in obj]
    if isinstance(obj, tuple): return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): obj = float(obj)
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return "Infinity" if math.isinf(obj) and obj > 0 else ("-Infinity" if math.isinf(obj) else None)
    if isinstance(obj, pd.Timestamp): return obj.isoformat()
    return obj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    panel, events, data_manifest = _prepare_panel()
    if events.empty:
        raise RuntimeError("v0.39 generated no candidate events")
    folds = _folds(events)
    feature_cols: list[str] = []
    for c in NEURAL_FEATURES_V39:
        feature_cols.extend([c, f"{c}__missing"])

    candidate_summaries: list[dict] = []
    fold_rows: list[dict] = []
    all_oos: list[pd.DataFrame] = []
    winner: str | None = None
    winner_trades = pd.DataFrame()
    winner_venue_results: dict[str, dict] = {}

    for candidate in CANDIDATES:
        oos_parts: list[pd.DataFrame] = []
        seed_fold_expectancies = {s: [] for s in SEEDS}
        fold_expectancies: list[float] = []
        for fold in folds:
            pred, seed_exps, diag = _predict_fold(candidate, fold, feature_cols)
            oos_parts.append(pred)
            fold_rows.append(diag)
            sel = pred[pred["selected_model"]]
            fold_expectancies.append(float(sel["net_r"].mean()) if not sel.empty else float("nan"))
            for s, value in zip(SEEDS, seed_exps):
                seed_fold_expectancies[s].append(value)

        oos = pd.concat(oos_parts, ignore_index=True).sort_values("signal_time")
        all_oos.append(oos)
        selected = oos[oos["selected_model"]].copy()
        realized = _realize_nonoverlap(selected)
        financially_realized = _financial_simulation(realized)
        executed = financially_realized[financially_realized["executed"]].copy() if not financially_realized.empty else financially_realized

        venue_results = {}
        for venue in VENUES:
            vt = executed[executed["venue"] == venue].copy() if not executed.empty else executed
            m = _venue_metrics(vt)
            g = _venue_gate(m)
            venue_results[venue] = {"metrics": m, "gates": g}

        finite_folds = [x for x in fold_expectancies if np.isfinite(x)]
        positive_fold_fraction = float(np.mean(np.asarray(finite_folds) > 0)) if finite_folds else 0.0
        seed_expectancies = []
        for s in SEEDS:
            vals = [v for v in seed_fold_expectancies[s] if np.isfinite(v)]
            seed_expectancies.append(float(np.mean(vals)) if vals else float("nan"))
        positive_seed_fraction = float(np.mean(np.asarray([v for v in seed_expectancies if np.isfinite(v)]) > 0)) if any(np.isfinite(seed_expectancies)) else 0.0
        cross_ok = positive_fold_fraction >= 0.60 and positive_seed_fraction >= (2.0 / 3.0)
        venue_ok = all(venue_results[v]["gates"]["venue_pass"] for v in VENUES)
        passed = bool(cross_ok and venue_ok)

        summary = {
            "candidate": candidate,
            "oos_events": int(len(oos)),
            "model_selected_events": int(len(selected)),
            "nonoverlap_events": int(len(realized)),
            "financially_executed_events": int(len(executed)),
            "positive_fold_fraction": positive_fold_fraction,
            "seed_expectancies_r": seed_expectancies,
            "positive_seed_fraction": positive_seed_fraction,
            "all_venue_gates_pass": venue_ok,
            "cross_fold_seed_gate_pass": cross_ok,
            "candidate_pass": passed,
            "venue_results": venue_results,
        }
        candidate_summaries.append(summary)
        if passed:
            winner = candidate
            winner_trades = executed
            winner_venue_results = venue_results
            break

    oos_all = pd.concat(all_oos, ignore_index=True) if all_oos else pd.DataFrame()
    oos_all.to_csv(outdir / "oos_predictions_v39.csv.gz", index=False, compression="gzip")
    winner_trades.to_csv(outdir / "trades_v39.csv.gz", index=False, compression="gzip")
    pd.DataFrame(fold_rows).to_csv(outdir / "fold_diagnostics_v39.csv", index=False)
    data_manifest.to_csv(outdir / "data_manifest_v39.csv", index=False)
    _worst_groups(winner_trades).to_csv(outdir / "worst_groups_v39.csv", index=False)
    pd.DataFrame([{k: v for k, v in s.items() if k != "venue_results"} for s in candidate_summaries]).to_csv(
        outdir / "candidate_summary_v39.csv", index=False
    )

    decision = {
        "version": "v0.39",
        "experiment": "ROBUST_MOTHER_STRATEGY_DEVELOPMENT_CHARACTERIZATION",
        "decision": "V39_DEVELOPMENT_WINNER" if winner else "V39_DEVELOPMENT_REJECT_OR_INSUFFICIENT_EVIDENCE",
        "winner": winner,
        "candidate_order": list(CANDIDATES),
        "selection_rule": "first/simplest candidate passing all frozen gates",
        "venues": list(VENUES),
        "symbols": list(SYMBOLS),
        "timeframe": TIMEFRAME,
        "start": START,
        "end": END,
        "purged_folds": PURGED_FOLDS,
        "embargo_bars": EMBARGO_BARS,
        "base_roundtrip_bps": BASE_ROUNDTRIP_BPS,
        "stress_roundtrip_bps": STRESS_ROUNDTRIP_BPS,
        "financial_risk_policy": asdict(FinancialRiskPolicyV39()),
        "learning_policy": asdict(LearningPolicyV39()),
        "validation_policy": asdict(ValidationPolicyV39()),
        "candidate_summaries": candidate_summaries,
        "winner_venue_results": winner_venue_results,
        "kraken_touched": False,
        "paper_execution": False,
        "live_execution": False,
        "post_result_threshold_relaxation": False,
        "post_result_feature_selection": False,
    }
    (outdir / "decision_v39.json").write_text(json.dumps(_jsonable(decision), indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(_jsonable(decision), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
