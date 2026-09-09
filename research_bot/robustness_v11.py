from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from math import ceil
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from .backtest import performance_metrics
from .oos_tournament_v10 import (
    TournamentConfig,
    _make_model,
    _portfolio_from_scores,
    _time_folds,
    build_tournament_panel,
)
from .regime import REGIME_COLUMNS

LEARNED_VARIANTS = ("logistic", "hgb", "random_forest")
BASELINE_VARIANTS = ("momentum_baseline", "ichimoku_baseline", "equal_weight_market")


@dataclass(frozen=True)
class RobustnessConfig:
    seeds: tuple[int, ...] = (11, 23, 42, 77, 101)
    bootstrap_samples: int = 1000
    block_length: int = 12
    alpha: float = 0.05
    fdr_alpha: float = 0.10
    min_regime_periods: int = 30
    positive_seed_fraction_required: float = 0.80
    max_allowed_drawdown: float = -0.35
    random_state: int = 11011


def _safe_float(value) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def _json_float(value) -> float | None:
    x = _safe_float(value)
    return x if x is not None else None


def benjamini_hochberg(p_values: Mapping[str, float]) -> dict[str, float]:
    """Benjamini-Hochberg FDR-adjusted q-values.

    Invalid p-values are ignored by the caller; output is monotone in ranked
    p-values and clipped to [0, 1].
    """

    clean = [(str(k), float(v)) for k, v in p_values.items() if np.isfinite(v)]
    if not clean:
        return {}
    clean.sort(key=lambda kv: kv[1])
    m = len(clean)
    adjusted = [min(1.0, p * m / (i + 1)) for i, (_, p) in enumerate(clean)]
    for i in range(m - 2, -1, -1):
        adjusted[i] = min(adjusted[i], adjusted[i + 1])
    return {clean[i][0]: float(max(0.0, min(1.0, adjusted[i]))) for i in range(m)}


def _moving_block_indices(n: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=int)
    b = max(1, min(int(block_length), n))
    n_blocks = int(ceil(n / b))
    starts = rng.integers(0, n, size=n_blocks)
    out: list[int] = []
    for start in starts:
        out.extend(((int(start) + j) % n) for j in range(b))
    return np.asarray(out[:n], dtype=int)


def moving_block_bootstrap_paired(
    candidate_returns: Sequence[float],
    baseline_returns: Sequence[float],
    *,
    samples: int = 1000,
    block_length: int = 12,
    alpha: float = 0.05,
    random_state: int = 11011,
    timeframe: str = "4h",
) -> dict:
    """Paired moving-block bootstrap for candidate-minus-baseline performance.

    The paired resampling preserves synchronous market periods and local serial
    dependence. The p-value is one-sided for positive mean net-return edge.
    """

    c = np.asarray(candidate_returns, dtype=float)
    b = np.asarray(baseline_returns, dtype=float)
    mask = np.isfinite(c) & np.isfinite(b)
    c, b = c[mask], b[mask]
    if len(c) < 10:
        return {"n": int(len(c)), "status": "INSUFFICIENT_OVERLAP"}

    rng = np.random.default_rng(random_state)
    mean_diffs: list[float] = []
    total_return_diffs: list[float] = []
    sharpe_diffs: list[float] = []
    for _ in range(max(1, int(samples))):
        idx = _moving_block_indices(len(c), block_length, rng)
        cc = c[idx]
        bb = b[idx]
        mean_diffs.append(float(np.mean(cc - bb)))
        total_return_diffs.append(float(np.prod(1.0 + cc) - np.prod(1.0 + bb)))
        cm = performance_metrics(pd.Series(cc), timeframe=timeframe)
        bm = performance_metrics(pd.Series(bb), timeframe=timeframe)
        cs = _safe_float(cm.get("sharpe"))
        bs = _safe_float(bm.get("sharpe"))
        if cs is not None and bs is not None:
            sharpe_diffs.append(float(cs - bs))

    lo = 100.0 * (alpha / 2.0)
    hi = 100.0 * (1.0 - alpha / 2.0)
    observed_diff = c - b
    observed_candidate = performance_metrics(pd.Series(c), timeframe=timeframe)
    observed_baseline = performance_metrics(pd.Series(b), timeframe=timeframe)
    p_edge = (1.0 + sum(x <= 0.0 for x in mean_diffs)) / (len(mean_diffs) + 1.0)

    return {
        "status": "OK",
        "n": int(len(c)),
        "block_length": int(max(1, min(block_length, len(c)))),
        "bootstrap_samples": int(len(mean_diffs)),
        "observed_mean_return_diff": float(np.mean(observed_diff)),
        "observed_total_return_diff": float(
            (np.prod(1.0 + c) - 1.0) - (np.prod(1.0 + b) - 1.0)
        ),
        "observed_sharpe_diff": (
            float(observed_candidate["sharpe"] - observed_baseline["sharpe"])
            if _safe_float(observed_candidate.get("sharpe")) is not None
            and _safe_float(observed_baseline.get("sharpe")) is not None
            else None
        ),
        "mean_return_diff_ci": [float(np.percentile(mean_diffs, lo)), float(np.percentile(mean_diffs, hi))],
        "total_return_diff_ci": [
            float(np.percentile(total_return_diffs, lo)),
            float(np.percentile(total_return_diffs, hi)),
        ],
        "sharpe_diff_ci": (
            [float(np.percentile(sharpe_diffs, lo)), float(np.percentile(sharpe_diffs, hi))]
            if sharpe_diffs
            else None
        ),
        "one_sided_p_mean_edge": float(p_edge),
    }


def _classify_regime(row: pd.Series) -> str:
    if float(row.get("regime_high_vol", 0.0)) >= 0.50:
        return "HIGH_VOL"
    up = float(row.get("regime_trend_up", 0.0))
    down = float(row.get("regime_trend_down", 0.0))
    if up >= 0.40 and up > down:
        return "TREND_UP"
    if down >= 0.40 and down > up:
        return "TREND_DOWN"
    return "RANGE"


def _seed_tournament_details(
    symbol_bars: Mapping[str, pd.DataFrame],
    tournament_config: TournamentConfig,
    seed: int,
) -> dict:
    cfg = replace(tournament_config, random_state=int(seed))
    panel, features = build_tournament_panel(symbol_bars, cfg)
    if panel.empty:
        raise ValueError("Empty tournament panel")
    folds = _time_folds(panel, cfg)

    prediction_rows: list[pd.DataFrame] = []
    test_rows: list[pd.DataFrame] = []
    for fold, train_ts, test_ts in folds:
        train = panel[panel["timestamp"].isin(train_ts)].copy()
        test = panel[panel["timestamp"].isin(test_ts)].copy()
        test["fold"] = fold
        test_rows.append(test)
        y = train["target_up"].astype(int)
        if y.nunique() < 2:
            continue
        for name in LEARNED_VARIANTS:
            model = _make_model(name, int(seed))
            model.fit(train[features], y)
            pred = test[["timestamp", "symbol", "future_return", "target_up", "fold"]].copy()
            pred["score"] = model.predict_proba(test[features])[:, 1]
            pred["model"] = name
            prediction_rows.append(pred)

    if not prediction_rows or not test_rows:
        raise ValueError("No OOS predictions were produced")

    predictions = pd.concat(prediction_rows, ignore_index=True)
    test_union = pd.concat(test_rows, ignore_index=True).drop_duplicates(["timestamp", "symbol", "fold"])
    variants: dict[str, pd.DataFrame] = {}
    for name, group in predictions.groupby("model"):
        variants[str(name)] = _portfolio_from_scores(group, "score", cfg)

    baseline = test_union.copy()
    baseline["momentum_score"] = (
        baseline["mom_24"].fillna(0.0)
        + 0.5 * baseline["mom_6"].fillna(0.0)
        - 0.25 * baseline["vol_24"].fillna(baseline["vol_24"].median())
    )
    baseline["ichimoku_score"] = (
        baseline["ichi_tenkan_kijun"].fillna(0.0)
        + baseline["ichi_price_kijun"].fillna(0.0)
        - baseline["ichi_cloud_width"].fillna(0.0)
    )
    variants["momentum_baseline"] = _portfolio_from_scores(baseline, "momentum_score", cfg)
    variants["ichimoku_baseline"] = _portfolio_from_scores(baseline, "ichimoku_score", cfg)
    variants["equal_weight_market"] = _portfolio_from_scores(
        baseline, "momentum_score", cfg, equal_weight=True
    )

    regime_cols = [c for c in REGIME_COLUMNS if c in test_union.columns]
    regime_by_ts = test_union.groupby("timestamp")[regime_cols].mean().reset_index()
    regime_by_ts["regime"] = regime_by_ts.apply(_classify_regime, axis=1)

    summaries: dict[str, dict] = {}
    for name, bt in variants.items():
        m = performance_metrics(bt["net_return"], timeframe=cfg.timeframe)
        m.update(
            {
                "variant": name,
                "gross_total_return": float((1.0 + bt["gross_return"]).prod() - 1.0),
                "net_total_return": float((1.0 + bt["net_return"]).prod() - 1.0),
                "explicit_cost_sum": float(bt["cost"].sum()),
                "turnover_sum": float(bt["turnover"].sum()),
                "rebalances": int(len(bt)),
            }
        )
        if name in LEARNED_VARIANTS:
            pred = predictions[predictions["model"].eq(name)]
            yy = pred["target_up"].astype(int)
            pp = pred["score"].astype(float).clip(1e-6, 1 - 1e-6)
            m["auc"] = float(roc_auc_score(yy, pp)) if yy.nunique() > 1 else None
            m["brier"] = float(brier_score_loss(yy, pp))
        summaries[name] = m

    return {
        "seed": int(seed),
        "config": asdict(cfg),
        "panel_rows": int(len(panel)),
        "panel_timestamps": int(panel["timestamp"].nunique()),
        "features": features,
        "fold_count": len(folds),
        "summaries": summaries,
        "variants": variants,
        "regime_by_ts": regime_by_ts,
    }


def _aggregate_seed_series(seed_details: Sequence[dict], variant: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for d in seed_details:
        bt = d["variants"].get(variant)
        if bt is None or bt.empty:
            continue
        x = bt[["timestamp", "net_return"]].copy()
        x["seed"] = int(d["seed"])
        frames.append(x)
    if not frames:
        return pd.DataFrame(columns=["timestamp", "net_return", "seed_count"])
    all_rows = pd.concat(frames, ignore_index=True)
    return (
        all_rows.groupby("timestamp", as_index=False)
        .agg(net_return=("net_return", "mean"), seed_count=("seed", "nunique"))
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def _seed_stability(seed_details: Sequence[dict], variant: str) -> dict:
    rows = [d["summaries"][variant] for d in seed_details if variant in d["summaries"]]
    returns = np.asarray([float(x.get("net_total_return", np.nan)) for x in rows], dtype=float)
    sharpes = np.asarray([float(x.get("sharpe", np.nan)) for x in rows], dtype=float)
    drawdowns = np.asarray([float(x.get("max_drawdown", np.nan)) for x in rows], dtype=float)
    returns = returns[np.isfinite(returns)]
    sharpes = sharpes[np.isfinite(sharpes)]
    drawdowns = drawdowns[np.isfinite(drawdowns)]
    return {
        "variant": variant,
        "seed_count": int(len(rows)),
        "net_return_median": float(np.median(returns)) if len(returns) else None,
        "net_return_mean": float(np.mean(returns)) if len(returns) else None,
        "net_return_std": float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0 if len(returns) else None,
        "net_return_min": float(np.min(returns)) if len(returns) else None,
        "net_return_max": float(np.max(returns)) if len(returns) else None,
        "positive_seed_fraction": float(np.mean(returns > 0.0)) if len(returns) else None,
        "sharpe_median": float(np.median(sharpes)) if len(sharpes) else None,
        "sharpe_mean": float(np.mean(sharpes)) if len(sharpes) else None,
        "sharpe_std": float(np.std(sharpes, ddof=1)) if len(sharpes) > 1 else 0.0 if len(sharpes) else None,
        "max_drawdown_median": float(np.median(drawdowns)) if len(drawdowns) else None,
    }


def _coverage_matched_pair(candidate: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    if candidate.empty or baseline.empty:
        return pd.DataFrame()
    return candidate.rename(columns={"net_return": "candidate_return"}).merge(
        baseline.rename(columns={"net_return": "baseline_return"})[["timestamp", "baseline_return"]],
        on="timestamp",
        how="inner",
        validate="one_to_one",
    )


def _regime_diagnostics(
    aggregate_series: Mapping[str, pd.DataFrame],
    regime_by_ts: pd.DataFrame,
    *,
    timeframe: str,
    min_periods: int,
) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for variant, series in aggregate_series.items():
        z = series.merge(regime_by_ts[["timestamp", "regime"]], on="timestamp", how="left")
        rows: list[dict] = []
        for regime, g in z.groupby("regime", dropna=False):
            n = int(len(g))
            if n < min_periods:
                rows.append({"regime": str(regime), "n": n, "status": "INSUFFICIENT_PERIODS"})
                continue
            m = performance_metrics(g["net_return"], timeframe=timeframe)
            rows.append(
                {
                    "regime": str(regime),
                    "n": n,
                    "status": "OK",
                    "total_return": _json_float(m.get("total_return")),
                    "sharpe": _json_float(m.get("sharpe")),
                    "sortino": _json_float(m.get("sortino")),
                    "max_drawdown": _json_float(m.get("max_drawdown")),
                    "cvar_95_loss": _json_float(m.get("cvar_95_loss")),
                }
            )
        out[variant] = rows
    return out


def run_robustness_v11(
    symbol_bars: Mapping[str, pd.DataFrame],
    tournament_config: TournamentConfig | None = None,
    robustness_config: RobustnessConfig | None = None,
) -> dict:
    """Run multi-seed OOS robustness/inference without authorizing execution."""

    tcfg = tournament_config or TournamentConfig()
    rcfg = robustness_config or RobustnessConfig()
    if not rcfg.seeds:
        raise ValueError("At least one seed is required")

    seed_details = [_seed_tournament_details(symbol_bars, tcfg, seed) for seed in rcfg.seeds]
    all_variants = tuple(LEARNED_VARIANTS + BASELINE_VARIANTS)
    seed_stability = {v: _seed_stability(seed_details, v) for v in all_variants}
    aggregate_series = {v: _aggregate_seed_series(seed_details, v) for v in all_variants}

    learned_ranked = sorted(
        [seed_stability[v] for v in LEARNED_VARIANTS],
        key=lambda x: x["sharpe_median"] if x.get("sharpe_median") is not None else -1e9,
        reverse=True,
    )
    provisional_best = learned_ranked[0]["variant"] if learned_ranked else None
    baseline_ranked = sorted(
        [seed_stability[v] for v in BASELINE_VARIANTS],
        key=lambda x: x["sharpe_median"] if x.get("sharpe_median") is not None else -1e9,
        reverse=True,
    )
    strongest_baseline = baseline_ranked[0]["variant"] if baseline_ranked else None

    pairwise: dict[str, dict] = {}
    raw_p: dict[str, float] = {}
    for learned in LEARNED_VARIANTS:
        for baseline in BASELINE_VARIANTS:
            pair = _coverage_matched_pair(aggregate_series[learned], aggregate_series[baseline])
            key = f"{learned}__vs__{baseline}"
            if pair.empty:
                pairwise[key] = {"status": "NO_COMMON_COVERAGE", "n": 0}
                continue
            inference = moving_block_bootstrap_paired(
                pair["candidate_return"],
                pair["baseline_return"],
                samples=rcfg.bootstrap_samples,
                block_length=rcfg.block_length,
                alpha=rcfg.alpha,
                random_state=rcfg.random_state + abs(hash(key)) % 100_000,
                timeframe=tcfg.timeframe,
            )
            inference["candidate"] = learned
            inference["baseline"] = baseline
            inference["coverage_match_fraction"] = float(
                len(pair)
                / max(1, min(len(aggregate_series[learned]), len(aggregate_series[baseline])))
            )
            pairwise[key] = inference
            p = _safe_float(inference.get("one_sided_p_mean_edge"))
            if p is not None:
                raw_p[key] = p

    q_values = benjamini_hochberg(raw_p)
    for key, q in q_values.items():
        pairwise[key]["fdr_q_value"] = float(q)
        pairwise[key]["fdr_significant"] = bool(q <= rcfg.fdr_alpha)

    regime_by_ts = seed_details[0]["regime_by_ts"]
    regime_diagnostics = _regime_diagnostics(
        aggregate_series,
        regime_by_ts,
        timeframe=tcfg.timeframe,
        min_periods=rcfg.min_regime_periods,
    )

    promotion_reasons: list[str] = []
    promotion = "NO_MODEL_PROMOTED"
    if provisional_best is None or strongest_baseline is None:
        promotion_reasons.append("MISSING_CANDIDATE_OR_BASELINE")
    else:
        stability = seed_stability[provisional_best]
        if (stability.get("net_return_median") or 0.0) <= 0.0:
            promotion_reasons.append("NON_POSITIVE_MEDIAN_SEED_RETURN")
        if (stability.get("positive_seed_fraction") or 0.0) < rcfg.positive_seed_fraction_required:
            promotion_reasons.append("INSUFFICIENT_MULTI_SEED_STABILITY")
        mdd = _safe_float(stability.get("max_drawdown_median"))
        if mdd is None or mdd < rcfg.max_allowed_drawdown:
            promotion_reasons.append("DRAWDOWN_TOO_LARGE")

        key = f"{provisional_best}__vs__{strongest_baseline}"
        test = pairwise.get(key, {})
        ci = test.get("mean_return_diff_ci")
        if not ci or float(ci[0]) <= 0.0:
            promotion_reasons.append("BOOTSTRAP_EDGE_CI_INCLUDES_ZERO")
        if float(test.get("fdr_q_value", 1.0)) > rcfg.fdr_alpha:
            promotion_reasons.append("NO_FDR_SIGNIFICANT_EDGE_OVER_STRONGEST_BASELINE")
        if float(test.get("coverage_match_fraction", 0.0)) < 0.95:
            promotion_reasons.append("INSUFFICIENT_COVERAGE_MATCH")

        valid_regimes = [
            x
            for x in regime_diagnostics.get(provisional_best, [])
            if x.get("status") == "OK"
        ]
        positive_regimes = sum(1 for x in valid_regimes if (x.get("total_return") or 0.0) > 0.0)
        if len(valid_regimes) >= 2 and positive_regimes < 2:
            promotion_reasons.append("INSUFFICIENT_REGIME_STABILITY")

        if not promotion_reasons:
            promotion = "PROVISIONAL_ROBUSTNESS_WINNER_STILL_NOT_EXECUTION_APPROVED"

    return {
        "research_status": "V11_ROBUSTNESS_INFERENCE_NOT_TRADING_SIGNAL",
        "tournament_config": asdict(tcfg),
        "robustness_config": asdict(rcfg),
        "symbols": sorted(symbol_bars),
        "seed_count": len(rcfg.seeds),
        "seed_stability": seed_stability,
        "provisional_best_model": provisional_best,
        "strongest_baseline": strongest_baseline,
        "pairwise_bootstrap_inference": pairwise,
        "multiple_testing": {
            "method": "BENJAMINI_HOCHBERG_FDR",
            "fdr_alpha": rcfg.fdr_alpha,
            "tests": len(q_values),
        },
        "regime_conditional_diagnostics": regime_diagnostics,
        "promotion_status": promotion,
        "promotion_reasons": promotion_reasons,
        "decision_contract": (
            "No result from v0.11 is a BUY/SELL instruction. Paper/Testnet promotion remains "
            "closed unless multi-seed, bootstrap, FDR, regime, cost, coverage and risk gates pass."
        ),
        "limitations": [
            "Moving-block bootstrap is an inference diagnostic, not proof of future profitability.",
            "Benjamini-Hochberg controls FDR across the tested model-baseline comparisons, not every research choice made historically.",
            "Regime labels are point-in-time technical regimes and do not yet include macro/on-chain/derivatives regimes.",
            "Live execution remains disabled.",
        ],
    }
