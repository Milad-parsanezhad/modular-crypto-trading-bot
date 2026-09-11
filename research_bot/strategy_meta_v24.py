from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from statistics import NormalDist
from typing import Iterable
import math

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from research_bot.ml_framework_v23r import (
    make_preprocessor,
    probability_score,
    select_feature_columns,
    supervised_model_registry,
)
from research_bot.ml_meta_v21 import build_labeled_candidate_attempts
from research_bot.multitimeframe_strategies_v19 import StrategySpec, TournamentConfig, moving_block_mean_ci
from research_bot.multitimeframe_strategies_v20 import STRATEGY_REGISTRY_V20, RiskPsychologyPolicy


TARGET_STRATEGIES_V24: tuple[str, ...] = (
    "H4_S6_BREAKOUT",
    "H4_KUMO_TRIANGLE",
    "H4_OB_BOS_RETEST",
    "H4_SUPPLY_DEMAND",
    "H4_CORRELATION_DIVERGENCE",
    "H4_D1_S6_VOL_RISK",
    "H4_D1_OB_BOS_RISK",
)


@dataclass(frozen=True)
class V24Contract:
    timeframe: str = "4h"
    risk_per_trade: float = 0.0025
    fee_bps_each_way: float = 10.0
    slippage_bps_each_way: float = 2.0
    min_validation_selected: int = 200
    min_test_selected: int = 100
    min_test_profit_factor: float = 1.05
    max_test_drawdown: float = 0.05
    min_symbol_uplift_fraction: float = 0.60
    min_strategy_uplift_fraction: float = 0.50
    min_dsr_probability: float = 0.95
    max_validation_pbo: float = 0.50
    threshold_quantiles: tuple[float, ...] = (0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.925, 0.95)
    seeds: tuple[int, ...] = (314, 2718, 1618)
    bootstrap_samples: int = 750
    bootstrap_block: int = 20
    cscv_groups: int = 6
    cusum_vol_window: int = 50
    cusum_threshold_sigma: float = 1.25
    live_execution_authorized: bool = False

    @property
    def roundtrip_bps(self) -> float:
        return 2.0 * (self.fee_bps_each_way + self.slippage_bps_each_way)

    def to_dict(self) -> dict:
        x = asdict(self)
        x["roundtrip_bps"] = self.roundtrip_bps
        x["strategy_names"] = list(TARGET_STRATEGIES_V24)
        return x


def target_specs() -> tuple[StrategySpec, ...]:
    by_name = {s.name: s for s in STRATEGY_REGISTRY_V20}
    missing = [n for n in TARGET_STRATEGIES_V24 if n not in by_name]
    if missing:
        raise RuntimeError(f"v0.24 strategy registry mismatch: {missing}")
    specs = tuple(by_name[n] for n in TARGET_STRATEGIES_V24)
    if any(s.timeframe != "4h" for s in specs):
        raise RuntimeError("v0.24 initial experiment is frozen to 4h candidates")
    return specs


def causal_cusum_features(frame: pd.DataFrame, *, vol_window: int = 50, threshold_sigma: float = 1.25) -> pd.DataFrame:
    """Causal close-to-close CUSUM event state.

    The threshold at bar t uses a volatility estimate shifted by one bar, while the
    return at t is available at the signal close. This makes the event observable at
    signal_time without future information.
    """
    required = {"timestamp", "close"}
    if required - set(frame.columns):
        raise ValueError(f"missing CUSUM columns: {sorted(required - set(frame.columns))}")
    x = frame[["timestamp", "close"]].copy().sort_values("timestamp").reset_index(drop=True)
    x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True, errors="coerce")
    close = pd.to_numeric(x["close"], errors="coerce")
    r = np.log(close.replace(0, np.nan)).diff()
    sigma = r.shift(1).rolling(vol_window, min_periods=max(10, vol_window // 3)).std()
    h = (threshold_sigma * sigma).clip(lower=1e-6)
    event = np.zeros(len(x), dtype=np.int8)
    age = np.full(len(x), np.nan)
    s_pos = 0.0
    s_neg = 0.0
    bars_since = 10_000
    for i in range(len(x)):
        ri = float(r.iloc[i]) if np.isfinite(r.iloc[i]) else np.nan
        hi = float(h.iloc[i]) if np.isfinite(h.iloc[i]) else np.nan
        if not np.isfinite(ri) or not np.isfinite(hi):
            bars_since += 1
            age[i] = bars_since
            continue
        s_pos = max(0.0, s_pos + ri)
        s_neg = min(0.0, s_neg + ri)
        if s_pos > hi:
            event[i] = 1
            s_pos = s_neg = 0.0
            bars_since = 0
        elif s_neg < -hi:
            event[i] = -1
            s_pos = s_neg = 0.0
            bars_since = 0
        else:
            bars_since += 1
        age[i] = bars_since
    return pd.DataFrame({
        "signal_time": x["timestamp"],
        "f_cusum_event": event,
        "f_cusum_abs_event": np.abs(event).astype(np.int8),
        "f_cusum_threshold": h,
        "f_cusum_age": age,
    })


def build_strategy_event_panel(
    spec: StrategySpec,
    frame: pd.DataFrame,
    symbol: str,
    *,
    peer: pd.DataFrame | None = None,
    contract: V24Contract | None = None,
) -> pd.DataFrame:
    """Create auditable strategy events with horizontal/vertical barrier outcomes.

    The inherited bracket simulator is a triple-barrier implementation: stop and
    target are the horizontal barriers, max_hold_bars is the vertical barrier,
    entry is next-open, and same-bar stop/target collisions are stop-first.
    """
    c = contract or V24Contract()
    tournament = TournamentConfig(
        fee_bps=c.fee_bps_each_way,
        slippage_bps=c.slippage_bps_each_way,
        risk_per_trade=c.risk_per_trade,
        min_pretest_trades=1000,
        min_test_trades=200,
    )
    policy = RiskPsychologyPolicy(base_risk_per_trade=c.risk_per_trade)
    x = build_labeled_candidate_attempts(spec, frame, symbol, tournament, policy, peer=peer)
    if x.empty:
        return x
    x = x.copy()
    x["signal_time"] = pd.to_datetime(x["signal_time"], utc=True).astype("datetime64[ns, UTC]")
    x["label_end_time"] = pd.to_datetime(x["exit_time"], utc=True).astype("datetime64[ns, UTC]")
    cusum = causal_cusum_features(frame, vol_window=c.cusum_vol_window, threshold_sigma=c.cusum_threshold_sigma)
    cusum["signal_time"] = pd.to_datetime(cusum["signal_time"], utc=True).astype("datetime64[ns, UTC]")
    x = x.merge(cusum, on="signal_time", how="left", validate="many_to_one")
    x["f_strategy_rr"] = float(spec.rr)
    x["f_strategy_stop_atr"] = float(spec.stop_atr)
    x["f_strategy_max_hold_bars"] = float(spec.max_hold_bars)
    # Meta-label = whether the frozen strategy event made money net of the frozen
    # market-order friction. Outcomes remain labels only and never enter X.
    x["label_meta_execute"] = (x["r_multiple"].astype(float) > 0).astype("int8")
    x["label_triple_barrier_class"] = np.select(
        [x["exit_reason"].eq("target"), x["exit_reason"].eq("stop")],
        [1, -1],
        default=0,
    ).astype("int8")
    x["label_vertical_barrier"] = x["exit_reason"].eq("timeout").astype("int8")
    # Segment from v0.19 is discarded later; v0.24 performs one pooled purged split
    # across all symbols/strategies so a timestamp cannot leak across segments.
    return x


def _stress_r_multiple(df: pd.DataFrame, roundtrip_bps: float) -> pd.Series:
    stop_fraction = (pd.to_numeric(df["entry"], errors="coerce") - pd.to_numeric(df["stop"], errors="coerce")).abs() / pd.to_numeric(df["entry"], errors="coerce").replace(0, np.nan)
    gross = pd.to_numeric(df["gross_return"], errors="coerce")
    return (gross - roundtrip_bps / 10_000.0) / stop_fraction.replace(0, np.nan)


def economic_metrics(
    rows: pd.DataFrame,
    selected: Iterable[bool] | None = None,
    *,
    risk_per_trade: float = 0.0025,
    roundtrip_bps: float | None = None,
) -> dict[str, float | int]:
    if rows.empty:
        return {
            "selected": 0, "coverage": 0.0, "mean_r": np.nan, "profit_factor": np.nan,
            "win_rate": np.nan, "total_return": np.nan, "max_drawdown": np.nan,
            "mean_account_return": np.nan,
        }
    mask = np.ones(len(rows), dtype=bool) if selected is None else np.asarray(list(selected), dtype=bool)
    if len(mask) != len(rows):
        raise ValueError("selection mask length mismatch")
    z = rows.loc[mask].sort_values(["entry_time", "symbol", "strategy"]).copy()
    if z.empty:
        return {
            "selected": 0, "coverage": 0.0, "mean_r": np.nan, "profit_factor": np.nan,
            "win_rate": np.nan, "total_return": 0.0, "max_drawdown": 0.0,
            "mean_account_return": 0.0,
        }
    r = _stress_r_multiple(z, roundtrip_bps) if roundtrip_bps is not None else pd.to_numeric(z["r_multiple"], errors="coerce")
    r = r.replace([np.inf, -np.inf], np.nan).dropna()
    if r.empty:
        return {
            "selected": 0, "coverage": 0.0, "mean_r": np.nan, "profit_factor": np.nan,
            "win_rate": np.nan, "total_return": 0.0, "max_drawdown": 0.0,
            "mean_account_return": 0.0,
        }
    account = risk_per_trade * r
    eq = (1.0 + account).cumprod()
    dd = eq / eq.cummax() - 1.0
    wins = float(account[account > 0].sum())
    losses = float(-account[account < 0].sum())
    pf = wins / losses if losses > 0 else (np.inf if wins > 0 else np.nan)
    return {
        "selected": int(len(r)),
        "coverage": float(len(r) / len(rows)),
        "mean_r": float(r.mean()),
        "profit_factor": float(pf),
        "win_rate": float((r > 0).mean()),
        "total_return": float(eq.iloc[-1] - 1.0),
        "max_drawdown": float(dd.min()),
        "mean_account_return": float(account.mean()),
    }


def choose_meta_threshold(validation: pd.DataFrame, score: np.ndarray, contract: V24Contract | None = None) -> tuple[float, dict]:
    c = contract or V24Contract()
    base = economic_metrics(validation, risk_per_trade=c.risk_per_trade)
    best_t = 1.0
    best: dict = {"validation_objective": -np.inf, "base": base, "filtered": economic_metrics(validation.iloc[0:0], risk_per_trade=c.risk_per_trade)}
    for q in c.threshold_quantiles:
        t = float(np.quantile(score, q))
        selected = np.asarray(score) >= t
        filt = economic_metrics(validation, selected, risk_per_trade=c.risk_per_trade)
        if int(filt["selected"]) < c.min_validation_selected:
            continue
        if not (np.isfinite(float(filt["mean_r"])) and np.isfinite(float(filt["profit_factor"]))):
            continue
        base_pf = float(base["profit_factor"]) if np.isfinite(float(base["profit_factor"])) else 0.0
        uplift_r = float(filt["mean_r"]) - float(base["mean_r"])
        uplift_pf = min(float(filt["profit_factor"]), 5.0) - min(base_pf, 5.0)
        dd_penalty = max(0.0, abs(float(filt["max_drawdown"])) - abs(float(base["max_drawdown"])))
        objective = uplift_r * math.sqrt(int(filt["selected"])) + 0.15 * uplift_pf - 0.75 * dd_penalty
        if objective > float(best["validation_objective"]):
            best_t = t
            best = {
                "validation_objective": float(objective),
                "base": base,
                "filtered": filt,
                "uplift_mean_r": float(uplift_r),
                "uplift_profit_factor": float(uplift_pf),
                "threshold_quantile": float(q),
            }
    return best_t, best


def _sharpe_per_trade(values: Iterable[float]) -> float:
    a = np.asarray(list(values), dtype=float)
    a = a[np.isfinite(a)]
    if len(a) < 3 or np.std(a, ddof=1) <= 0:
        return np.nan
    return float(np.mean(a) / np.std(a, ddof=1))


def deflated_sharpe_probability(
    returns: Iterable[float],
    *,
    trial_sharpes: Iterable[float],
) -> dict[str, float | int]:
    """Bailey/Lopez-de-Prado style DSR diagnostic on per-trade returns.

    The benchmark Sharpe is the expected maximum under the observed dispersion of
    tried validation configurations. This is a multiple-testing diagnostic, not a
    substitute for untouched external replication.
    """
    r = np.asarray(list(returns), dtype=float)
    r = r[np.isfinite(r)]
    trials = np.asarray(list(trial_sharpes), dtype=float)
    trials = trials[np.isfinite(trials)]
    sr = _sharpe_per_trade(r)
    if len(r) < 30 or not np.isfinite(sr) or len(trials) < 2:
        return {"observations": int(len(r)), "trial_count": int(len(trials)), "sharpe": float(sr) if np.isfinite(sr) else np.nan, "benchmark_sharpe": np.nan, "probability": np.nan}
    sigma_trials = float(np.std(trials, ddof=1))
    n_trials = max(2, len(trials))
    nd = NormalDist()
    gamma = 0.5772156649015329
    z1 = nd.inv_cdf(1.0 - 1.0 / n_trials)
    z2 = nd.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
    benchmark = sigma_trials * ((1.0 - gamma) * z1 + gamma * z2)
    centered = r - np.mean(r)
    sd = float(np.std(r, ddof=1))
    skew = float(np.mean(centered ** 3) / (sd ** 3)) if sd > 0 else 0.0
    kurt = float(np.mean(centered ** 4) / (sd ** 4)) if sd > 0 else 3.0
    denom_sq = max(1e-12, 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr * sr)
    z = (sr - benchmark) * math.sqrt(len(r) - 1.0) / math.sqrt(denom_sq)
    return {
        "observations": int(len(r)),
        "trial_count": int(len(trials)),
        "sharpe": float(sr),
        "benchmark_sharpe": float(benchmark),
        "skew": skew,
        "kurtosis": kurt,
        "probability": float(nd.cdf(z)),
    }


def cscv_pbo_diagnostic(return_matrix: pd.DataFrame, *, groups: int = 6) -> dict[str, float | int | str]:
    """CSCV-style probability-of-backtest-overfitting diagnostic.

    Columns are frozen model/seed/threshold configurations and rows are ordered
    validation events. No test data enters this calculation. Full CPCV retraining
    remains mandatory if a v0.24 candidate survives the untouched test gate.
    """
    if return_matrix.empty or return_matrix.shape[1] < 2 or len(return_matrix) < groups * 20:
        return {"decision": "INSUFFICIENT_FOR_PBO", "paths": 0, "pbo": np.nan, "median_oos_rank_percentile": np.nan}
    x = return_matrix.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    blocks = [b for b in np.array_split(np.arange(len(x)), groups) if len(b)]
    k = len(blocks) // 2
    ranks: list[float] = []
    for test_groups in combinations(range(len(blocks)), k):
        test_set = set(test_groups)
        is_idx = np.concatenate([blocks[i] for i in range(len(blocks)) if i not in test_set])
        oos_idx = np.concatenate([blocks[i] for i in range(len(blocks)) if i in test_set])
        is_perf = np.mean(x[is_idx], axis=0)
        winner = int(np.nanargmax(is_perf))
        oos_perf = np.mean(x[oos_idx], axis=0)
        winner_perf = float(oos_perf[winner])
        percentile = (1.0 + float(np.sum(oos_perf < winner_perf))) / float(len(oos_perf) + 1.0)
        ranks.append(percentile)
    pbo = float(np.mean(np.asarray(ranks) <= 0.5)) if ranks else np.nan
    return {
        "decision": "PBO_DIAGNOSTIC_COMPLETE" if ranks else "INSUFFICIENT_FOR_PBO",
        "paths": int(len(ranks)),
        "pbo": pbo,
        "median_oos_rank_percentile": float(np.median(ranks)) if ranks else np.nan,
        "note": "validation-only CSCV ranking diagnostic; a surviving candidate still requires full CPCV/external replication",
    }


def breadth_uplift(rows: pd.DataFrame, selected: np.ndarray, *, group_col: str, risk_per_trade: float) -> dict[str, float | int]:
    values = []
    for key, g in rows.groupby(group_col, sort=True):
        pos = rows.index.get_indexer(g.index)
        mask = selected[pos]
        base = economic_metrics(g, risk_per_trade=risk_per_trade)
        filt = economic_metrics(g, mask, risk_per_trade=risk_per_trade)
        values.append((key, float(filt["total_return"]) - float(base["total_return"])))
    if not values:
        return {"groups": 0, "uplift_fraction": 0.0}
    return {
        "groups": int(len(values)),
        "uplift_fraction": float(np.mean([v > 0 for _, v in values])),
        "positive_uplift_groups": int(sum(v > 0 for _, v in values)),
    }


def paired_block_uplift_ci(rows: pd.DataFrame, selected: np.ndarray, contract: V24Contract | None = None) -> tuple[float, float]:
    c = contract or V24Contract()
    ordered = rows.sort_values(["entry_time", "symbol", "strategy"]).copy()
    selection = pd.Series(selected, index=rows.index).reindex(ordered.index).fillna(False).to_numpy(dtype=bool)
    base = c.risk_per_trade * pd.to_numeric(ordered["r_multiple"], errors="coerce").fillna(0.0).to_numpy()
    filtered = np.where(selection, base, 0.0)
    return moving_block_mean_ci(
        filtered - base,
        samples=c.bootstrap_samples,
        block=c.bootstrap_block,
        seed=314159,
    )


def cost_stress_table(rows: pd.DataFrame, selected: np.ndarray, contract: V24Contract | None = None) -> pd.DataFrame:
    c = contract or V24Contract()
    records = []
    for bps in (c.roundtrip_bps, 36.0, 60.0):
        base = economic_metrics(rows, risk_per_trade=c.risk_per_trade, roundtrip_bps=bps)
        filt = economic_metrics(rows, selected, risk_per_trade=c.risk_per_trade, roundtrip_bps=bps)
        records.append({
            "roundtrip_bps": float(bps),
            **{f"base_{k}": v for k, v in base.items()},
            **{f"filtered_{k}": v for k, v in filt.items()},
        })
    return pd.DataFrame(records)


def promotion_decision(
    test_rows: pd.DataFrame,
    selected: np.ndarray,
    *,
    validation_pbo: dict,
    dsr: dict,
    bootstrap_ci: tuple[float, float],
    cost_stress: pd.DataFrame,
    contract: V24Contract | None = None,
) -> dict:
    c = contract or V24Contract()
    base = economic_metrics(test_rows, risk_per_trade=c.risk_per_trade)
    filt = economic_metrics(test_rows, selected, risk_per_trade=c.risk_per_trade)
    symbol_breadth = breadth_uplift(test_rows.reset_index(drop=True), selected, group_col="symbol", risk_per_trade=c.risk_per_trade)
    strategy_breadth = breadth_uplift(test_rows.reset_index(drop=True), selected, group_col="strategy", risk_per_trade=c.risk_per_trade)
    stress36 = cost_stress.loc[np.isclose(cost_stress["roundtrip_bps"], 36.0)].iloc[0].to_dict()
    pbo = float(validation_pbo.get("pbo", np.nan))
    dsr_p = float(dsr.get("probability", np.nan))
    gates = {
        "min_test_selected": int(filt["selected"]) >= c.min_test_selected,
        "positive_filtered_expectancy": np.isfinite(float(filt["mean_r"])) and float(filt["mean_r"]) > 0,
        "profit_factor": np.isfinite(float(filt["profit_factor"])) and float(filt["profit_factor"]) >= c.min_test_profit_factor,
        "incremental_total_return": float(filt["total_return"]) > float(base["total_return"]),
        "max_drawdown": abs(float(filt["max_drawdown"])) <= c.max_test_drawdown,
        "paired_bootstrap_uplift": np.isfinite(bootstrap_ci[0]) and float(bootstrap_ci[0]) > 0,
        "symbol_breadth": float(symbol_breadth["uplift_fraction"]) >= c.min_symbol_uplift_fraction,
        "strategy_breadth": float(strategy_breadth["uplift_fraction"]) >= c.min_strategy_uplift_fraction,
        "deflated_sharpe": np.isfinite(dsr_p) and dsr_p >= c.min_dsr_probability,
        "validation_pbo": np.isfinite(pbo) and pbo <= c.max_validation_pbo,
        "stress_36bps": np.isfinite(float(stress36["filtered_profit_factor"])) and float(stress36["filtered_profit_factor"]) >= 1.0 and float(stress36["filtered_mean_r"]) > 0,
    }
    passed = bool(all(gates.values()))
    return {
        "decision": "INTERNAL_META_RESEARCH_CANDIDATE" if passed else "NO_META_MODEL_PROMOTED",
        "all_gates_passed": passed,
        "gates": gates,
        "base_test": base,
        "filtered_test": filt,
        "symbol_breadth": symbol_breadth,
        "strategy_breadth": strategy_breadth,
        "paired_bootstrap_uplift_ci": {"low": bootstrap_ci[0], "high": bootstrap_ci[1]},
        "validation_pbo": validation_pbo,
        "deflated_sharpe": dsr,
        "requires_full_cpcv_retraining": True,
        "requires_external_venue_replication": True,
        "requires_strategy_level_portfolio_risk_backtest": True,
        "forward_paper_authorized": False,
        "paper_replacement_authorized": False,
        "live_execution_authorized": False,
    }


def fit_meta_models(
    development: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    *,
    contract: V24Contract | None = None,
) -> tuple[pd.DataFrame, dict, Pipeline, pd.DataFrame, pd.DataFrame]:
    """Fit on development, select family/seed/threshold on validation, inspect test once."""
    c = contract or V24Contract()
    numeric, categorical = select_feature_columns(development, include_context=True, include_identity=False)
    xcols = numeric + categorical
    validation_rows = []
    fitted: dict[tuple[str, int], Pipeline] = {}
    config_returns: dict[str, np.ndarray] = {}
    trial_sharpes: list[float] = []
    base_val_account = c.risk_per_trade * pd.to_numeric(validation["r_multiple"], errors="coerce").fillna(0.0).to_numpy()

    for seed in c.seeds:
        for name, estimator in supervised_model_registry(seed).items():
            pipe = Pipeline([("prep", make_preprocessor(numeric, categorical)), ("model", clone(estimator))])
            try:
                pipe.fit(development[xcols], development["label_meta_execute"].astype(int))
                score = probability_score(pipe, validation[xcols])
                threshold, val_eval = choose_meta_threshold(validation, score, c)
                fitted[(name, seed)] = pipe
                validation_rows.append({
                    "model": name,
                    "seed": seed,
                    "status": "ok",
                    "threshold": threshold,
                    "validation_objective": float(val_eval["validation_objective"]),
                    "validation_uplift_mean_r": float(val_eval.get("uplift_mean_r", np.nan)),
                    "validation_uplift_profit_factor": float(val_eval.get("uplift_profit_factor", np.nan)),
                    **{f"validation_base_{k}": v for k, v in val_eval["base"].items()},
                    **{f"validation_filtered_{k}": v for k, v in val_eval["filtered"].items()},
                })
                for q in c.threshold_quantiles:
                    qt = float(np.quantile(score, q))
                    selected = score >= qt
                    vector = np.where(selected, base_val_account, 0.0)
                    key = f"{name}|seed={seed}|q={q:.3f}"
                    config_returns[key] = vector
                    trial_sharpes.append(_sharpe_per_trade(vector))
            except Exception as exc:
                validation_rows.append({"model": name, "seed": seed, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})

    board = pd.DataFrame(validation_rows)
    ok = board[board["status"] == "ok"].copy()
    if ok.empty:
        raise RuntimeError("all v0.24 meta-label models failed")
    family = ok.groupby("model", as_index=False).agg(
        validation_objective_mean=("validation_objective", "mean"),
        validation_objective_std=("validation_objective", "std"),
        validation_uplift_mean_r_mean=("validation_uplift_mean_r", "mean"),
        validation_filtered_pf_mean=("validation_filtered_profit_factor", "mean"),
        seeds=("seed", "nunique"),
    ).sort_values(["validation_objective_mean", "validation_objective_std"], ascending=[False, True])
    champion_name = str(family.iloc[0]["model"])
    seed_row = ok[ok["model"] == champion_name].sort_values("validation_objective", ascending=False).iloc[0]
    champion_seed = int(seed_row["seed"])
    threshold = float(seed_row["threshold"])
    champion = fitted[(champion_name, champion_seed)]

    # Untouched test is scored only after model family, seed and threshold freeze.
    test_score = probability_score(champion, test[xcols])
    selected = np.asarray(test_score) >= threshold
    test_predictions = test[["signal_time", "entry_time", "exit_time", "strategy", "symbol", "side", "r_multiple"]].copy()
    test_predictions["meta_score"] = test_score
    test_predictions["meta_selected"] = selected

    pbo_matrix = pd.DataFrame(config_returns, index=validation.index)
    pbo = cscv_pbo_diagnostic(pbo_matrix, groups=c.cscv_groups)
    bootstrap_ci = paired_block_uplift_ci(test, selected, c)
    stress = cost_stress_table(test, selected, c)
    selected_account = c.risk_per_trade * pd.to_numeric(test.loc[selected, "r_multiple"], errors="coerce").dropna().to_numpy()
    dsr = deflated_sharpe_probability(selected_account, trial_sharpes=trial_sharpes)
    decision = promotion_decision(
        test.reset_index(drop=True), selected,
        validation_pbo=pbo, dsr=dsr, bootstrap_ci=bootstrap_ci,
        cost_stress=stress, contract=c,
    )
    decision.update({
        "champion_family": champion_name,
        "champion_seed": champion_seed,
        "frozen_threshold": threshold,
        "fit_segment": "development_only",
        "selection_segment": "validation_only",
        "test_used_for_selection": False,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "search_trials": int(len(config_returns)),
    })
    return board, decision, champion, test_predictions, stress
