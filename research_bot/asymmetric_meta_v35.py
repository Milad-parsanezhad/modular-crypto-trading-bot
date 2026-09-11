from __future__ import annotations

"""v0.35 preregistered asymmetric regime + cross-sectional meta-labeling.

Scientific motivation
---------------------
The consumed v0.31-v0.33 evidence shows positive aggregate economics but weak
cross-asset/block stability. v0.35 therefore introduces a genuinely new alpha
family instead of retuning the rejected breadth thresholds.

The design is intentionally asymmetric:
- longs require persistent UP-UP market regime;
- shorts may operate in DOWN-DOWN or transition regimes (regime <= 0), reflecting
  the stronger short-side evidence observed in consumed v0.32 attribution;
- entries must also be cross-sectionally strong/weak in 20d+60d relative strength;
- a causal empirical-Bayes meta-label score abstains from historically weak
  side/regime/rank states using only events whose exits were known strictly before
  the new signal time.

CoinEx, OKX and KuCoin are consumed development evidence. Kraken remains sealed as
an untouched future holdout and is never fetched by this module.
"""

from dataclasses import asdict, dataclass
from typing import Any, Iterable
import math

import numpy as np
import pandas as pd

from research_bot.regime_event_alpha_v30 import (
    V30_CANDIDATES,
    _attach_context,
    _prior_extreme,
    _standardized_cusum,
    _volume_confirm,
)


PRIOR_EFFECTIVE_TRIALS_V35 = 114
NEW_TRIALS_V35 = 6
TOTAL_EFFECTIVE_TRIALS_V35 = PRIOR_EFFECTIVE_TRIALS_V35 + NEW_TRIALS_V35
DEVELOPMENT_VENUES_V35 = ("coinex_consumed", "okx_consumed", "kucoin_consumed")
RESERVED_HOLDOUT_VENUE_V35 = "kraken"
MIN_META_HISTORY_V35 = 20


@dataclass(frozen=True)
class V35Candidate:
    name: str
    long_rank_min: float
    short_rank_max: float
    long_meta_min: float
    short_meta_min: float


V35_CANDIDATES: tuple[V35Candidate, ...] = (
    V35Candidate("V35_D1_ASYM_META_L55_S45_M50_45", 0.55, 0.45, 0.50, 0.45),
    V35Candidate("V35_D1_ASYM_META_L60_S45_M50_45", 0.60, 0.45, 0.50, 0.45),
    V35Candidate("V35_D1_ASYM_META_L60_S40_M50_45", 0.60, 0.40, 0.50, 0.45),
    V35Candidate("V35_D1_ASYM_META_L60_S40_M55_45", 0.60, 0.40, 0.55, 0.45),
    V35Candidate("V35_D1_ASYM_META_L65_S40_M55_45", 0.65, 0.40, 0.55, 0.45),
    V35Candidate("V35_D1_ASYM_META_L65_S35_M55_50", 0.65, 0.35, 0.55, 0.50),
)


def preregistration_manifest_v35() -> dict[str, Any]:
    return {
        "version": "v0.35",
        "experiment": "ASYMMETRIC_REGIME_CROSS_SECTIONAL_META_LABELING",
        "base_event_family": "CUSUM breakout, newly asymmetric regime semantics",
        "candidate_count": len(V35_CANDIDATES),
        "candidates": [asdict(x) for x in V35_CANDIDATES],
        "development_venues": list(DEVELOPMENT_VENUES_V35),
        "reserved_holdout_venue": RESERVED_HOLDOUT_VENUE_V35,
        "prior_effective_trials": PRIOR_EFFECTIVE_TRIALS_V35,
        "new_trials": NEW_TRIALS_V35,
        "total_effective_trials": TOTAL_EFFECTIVE_TRIALS_V35,
        "min_meta_history": MIN_META_HISTORY_V35,
        "meta_prior_probability": 0.50,
        "meta_prior_strength": 4.0,
        "meta_outcome": "positive post-cost R-multiple",
        "strict_meta_availability": "historical exit_time < current signal_time",
        "long_regime": "persistent UP-UP only",
        "short_regime": "persistent DOWN-DOWN or transition (regime <= 0)",
        "relative_strength_horizons_days": [20, 60],
        "dispersion_reference": "lagged rolling 80th percentile",
        "min_trades_per_venue": 200,
        "min_profit_factor": 1.05,
        "min_expectancy_r": 0.0,
        "min_positive_asset_fraction": 0.60,
        "max_drawdown": 0.05,
        "min_block_ci_low": 0.0,
        "round_trip_cost_fraction": 0.0024,
        "aggregate_open_risk_cap": 0.02,
        "directional_open_risk_cap": 0.015,
        "threshold_relaxation": False,
        "rejected_candidate_retuning": False,
        "kraken_touched": False,
        "paper_replacement_authorized": False,
        "live_execution_authorized": False,
    }


def cross_sectional_context_v35(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Causal 20d/60d relative-strength ranks and lagged dispersion state."""
    parts: list[pd.DataFrame] = []
    for symbol, frame in sorted(frames.items()):
        x = frame[["timestamp", "close"]].copy().sort_values("timestamp").reset_index(drop=True)
        x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True, errors="raise")
        x["mom20_v35"] = pd.to_numeric(x["close"], errors="coerce").pct_change(20)
        x["mom60_v35"] = pd.to_numeric(x["close"], errors="coerce").pct_change(60)
        x["symbol"] = symbol
        parts.append(x[["timestamp", "symbol", "mom20_v35", "mom60_v35"]])
    if not parts:
        return pd.DataFrame(columns=[
            "timestamp", "symbol", "rank20_v35", "rank60_v35", "rs_score_v35",
            "dispersion_v35", "dispersion_cap_v35", "dispersion_ratio_v35",
        ])
    long = pd.concat(parts, ignore_index=True).sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    long["rank20_v35"] = long.groupby("timestamp", sort=False)["mom20_v35"].rank(pct=True, method="average")
    long["rank60_v35"] = long.groupby("timestamp", sort=False)["mom60_v35"].rank(pct=True, method="average")
    long["rs_score_v35"] = 0.5 * (long["rank20_v35"] + long["rank60_v35"])
    dispersion = long.groupby("timestamp", sort=True)["mom20_v35"].std(ddof=0)
    cap = dispersion.rolling(90, min_periods=30).quantile(0.80).shift(1)
    d = pd.DataFrame({
        "timestamp": dispersion.index,
        "dispersion_v35": dispersion.to_numpy(),
        "dispersion_cap_v35": cap.to_numpy(),
    })
    d["dispersion_ratio_v35"] = d["dispersion_v35"] / d["dispersion_cap_v35"].replace(0, np.nan)
    return long.merge(d, on="timestamp", how="left", validate="many_to_one")


def generate_asymmetric_direction_v35(
    frame: pd.DataFrame,
    *,
    market_frame: pd.DataFrame,
    dispersion: pd.DataFrame | None,
) -> tuple[pd.Series, pd.DataFrame]:
    """New asymmetric CUSUM-breakout hypothesis with frozen v0.30 economics."""
    base = next(x for x in V30_CANDIDATES if x.name == "V30_D1_CUSUM_BREAKOUT_3")
    f = _attach_context(frame, timeframe="1d", market_frame=market_frame, dispersion=dispersion)
    pos, neg = _standardized_cusum(f)
    high_break = _prior_extreme(f["high"], base.breakout_lookback, kind="max")
    low_break = _prior_extreme(f["low"], base.breakout_lookback, kind="min")
    volume_ok = _volume_confirm(f, base.volume_multiplier)
    regime = pd.to_numeric(f["market_regime_v30"], errors="coerce").fillna(0)
    dispersion_ok = f["low_dispersion_v30"].fillna(False)

    long = (
        (pos >= float(base.cusum_threshold))
        & (f["close"] > high_break)
        & volume_ok
        & regime.eq(1)
        & dispersion_ok
    )
    short = (
        (neg <= -float(base.cusum_threshold))
        & (f["close"] < low_break)
        & volume_ok
        & regime.le(0)
        & dispersion_ok
    )
    direction = pd.Series(0, index=f.index, dtype="int8")
    direction.loc[long.fillna(False)] = 1
    direction.loc[short.fillna(False)] = -1
    direction.loc[long.fillna(False) & short.fillna(False)] = 0
    f["cusum_positive_v35"] = pos
    f["cusum_negative_v35"] = neg
    f["asym_long_regime_ok_v35"] = regime.eq(1)
    f["asym_short_regime_ok_v35"] = regime.le(0)
    return direction, f


def attach_event_context_v35(events: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    required = {"signal_time", "symbol", "entry_time", "exit_time", "side", "r_multiple"}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"v0.35 event panel missing columns: {sorted(missing)}")
    x = events.copy()
    x["signal_time"] = pd.to_datetime(x["signal_time"], utc=True, errors="raise")
    x["entry_time"] = pd.to_datetime(x["entry_time"], utc=True, errors="raise")
    x["exit_time"] = pd.to_datetime(x["exit_time"], utc=True, errors="raise")
    c = context.copy()
    c["timestamp"] = pd.to_datetime(c["timestamp"], utc=True, errors="raise")
    c = c.rename(columns={"timestamp": "signal_time"})
    keep = ["signal_time", "symbol", "rank20_v35", "rank60_v35", "rs_score_v35", "dispersion_ratio_v35"]
    x = x.merge(c[keep], on=["signal_time", "symbol"], how="left", validate="many_to_one")
    x["rs_bucket_v35"] = pd.cut(
        x["rs_score_v35"], bins=[-np.inf, 1/3, 2/3, np.inf], labels=[0, 1, 2]
    ).astype("float")
    x["dispersion_bucket_v35"] = (pd.to_numeric(x["dispersion_ratio_v35"], errors="coerce") > 1.0).astype("int8")
    # Base generator already stored the market regime in the ledger only indirectly;
    # infer the asymmetric state from side semantics for grouping while keeping the
    # causal signal rule itself in generate_asymmetric_direction_v35.
    x["regime_group_v35"] = np.where(x["side"].astype(int) > 0, 1, -1).astype("int8")
    return x


def _duration_weight(row: pd.Series) -> float:
    delta = pd.Timestamp(row["exit_time"]) - pd.Timestamp(row["entry_time"])
    days = max(1.0, float(delta.total_seconds() / 86400.0))
    return float(1.0 / math.sqrt(days))


def causal_empirical_bayes_meta_score_v35(events: pd.DataFrame) -> pd.DataFrame:
    """Score event quality using only strictly settled historical events.

    The score is a duration-weighted empirical-Bayes positive-R probability.
    Exact side/regime/rank/dispersion states are used when sufficiently populated,
    with deterministic fallback to side+regime, side, and global history.
    """
    if events.empty:
        x = events.copy()
        x["meta_score_v35"] = np.nan
        x["meta_history_n_v35"] = 0
        x["meta_history_level_v35"] = "none"
        return x
    x = events.sort_values(["signal_time", "symbol"], kind="mergesort").reset_index(drop=True).copy()
    settled = x.sort_values(["exit_time", "symbol"], kind="mergesort").reset_index().rename(columns={"index": "event_index_v35"})

    stats: dict[tuple[Any, ...], list[float]] = {}

    def keys(row: pd.Series) -> list[tuple[Any, ...]]:
        side = int(row["side"])
        regime = int(row["regime_group_v35"])
        rank = int(row["rs_bucket_v35"]) if np.isfinite(row["rs_bucket_v35"]) else -1
        disp = int(row["dispersion_bucket_v35"])
        return [
            ("exact", side, regime, rank, disp),
            ("side_regime", side, regime),
            ("side", side),
            ("global",),
        ]

    scores: list[float] = []
    histories: list[int] = []
    levels: list[str] = []
    j = 0
    for _, row in x.iterrows():
        signal_time = pd.Timestamp(row["signal_time"])
        while j < len(settled) and pd.Timestamp(settled.iloc[j]["exit_time"]) < signal_time:
            old = settled.iloc[j]
            y = 1.0 if float(old["r_multiple"]) > 0.0 else 0.0
            w = _duration_weight(old)
            for key in keys(old):
                slot = stats.setdefault(key, [0.0, 0.0, 0.0])  # weight, weighted wins, count
                slot[0] += w
                slot[1] += w * y
                slot[2] += 1.0
            j += 1

        chosen = None
        for key, minimum in zip(keys(row), (12, 20, 30, 40)):
            slot = stats.get(key)
            if slot is not None and int(slot[2]) >= minimum:
                chosen = (key, slot)
                break
        if chosen is None:
            slot = stats.get(("global",), [0.0, 0.0, 0.0])
            level = "prior" if int(slot[2]) < MIN_META_HISTORY_V35 else "global"
        else:
            key, slot = chosen
            level = str(key[0])
        weight, wins, count = slot
        prior_strength = 4.0
        score = (wins + prior_strength * 0.50) / (weight + prior_strength)
        scores.append(float(score))
        histories.append(int(count))
        levels.append(level)

    x["meta_score_v35"] = scores
    x["meta_history_n_v35"] = histories
    x["meta_history_level_v35"] = levels
    return x


def select_candidate_events_v35(events: pd.DataFrame, candidate: V35Candidate) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    x = events.copy()
    rs = pd.to_numeric(x["rs_score_v35"], errors="coerce")
    meta = pd.to_numeric(x["meta_score_v35"], errors="coerce")
    history_ok = pd.to_numeric(x["meta_history_n_v35"], errors="coerce").fillna(0).ge(MIN_META_HISTORY_V35)
    long_ok = (
        x["side"].astype(int).eq(1)
        & rs.ge(candidate.long_rank_min)
        & meta.ge(candidate.long_meta_min)
        & history_ok
    )
    short_ok = (
        x["side"].astype(int).eq(-1)
        & rs.le(candidate.short_rank_max)
        & meta.ge(candidate.short_meta_min)
        & history_ok
    )
    out = x.loc[long_ok | short_ok].copy()
    out["strategy"] = candidate.name
    out["family"] = "v35_asymmetric_regime_cross_sectional_meta"
    out["source_basis"] = "Preregistered asymmetric regime + cross-sectional relative strength + causal empirical-Bayes meta-label"
    return out.sort_values(["entry_time", "symbol"], kind="mergesort").reset_index(drop=True)


def _num(m: dict[str, Any], key: str, default: float = np.nan) -> float:
    try:
        v = float(m.get(key, default))
    except (TypeError, ValueError):
        return default
    return v if np.isfinite(v) else default


def screen_three_venues_v35(metrics_by_venue: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = [metrics_by_venue[v] for v in DEVELOPMENT_VENUES_V35]
    trades = [int(r.get("trades", 0) or 0) for r in rows]
    pf = [_num(r, "profit_factor") for r in rows]
    exp = [_num(r, "expectancy_r") for r in rows]
    breadth = [_num(r, "positive_asset_fraction") for r in rows]
    dd = [abs(_num(r, "max_drawdown")) for r in rows]
    ci = [_num(r, "block_ci_low") for r in rows]
    eligible = (
        all(n >= 200 for n in trades)
        and all(np.isfinite(v) and v >= 1.05 for v in pf)
        and all(np.isfinite(v) and v > 0.0 for v in exp)
        and all(np.isfinite(v) and v >= 0.60 for v in breadth)
        and all(np.isfinite(v) and v <= 0.05 for v in dd)
        and all(np.isfinite(v) and v > 0.0 for v in ci)
    )
    return {
        "development_eligible_v35": bool(eligible),
        "robust_min_trades_v35": min(trades),
        "robust_floor_profit_factor_v35": min(pf),
        "robust_floor_expectancy_r_v35": min(exp),
        "robust_floor_breadth_v35": min(breadth),
        "robust_floor_block_ci_low_v35": min(ci),
        "robust_worst_drawdown_v35": max(dd),
    }


def select_v35_winner(rows: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [dict(r) for r in rows if bool(r.get("development_eligible_v35", False))]
    if not eligible:
        return None
    eligible = sorted(eligible, key=lambda r: str(r.get("strategy", "")))
    return max(eligible, key=lambda r: (
        float(r["robust_floor_block_ci_low_v35"]),
        float(r["robust_floor_breadth_v35"]),
        float(r["robust_floor_profit_factor_v35"]),
        float(r["robust_floor_expectancy_r_v35"]),
        int(r["robust_min_trades_v35"]),
        -float(r["robust_worst_drawdown_v35"]),
    ))


def v35_decision(winner: dict[str, Any] | None) -> dict[str, Any]:
    state = "V35_ASYMMETRIC_META_CANDIDATE_LOCKED_FOR_FRESH_HOLDOUT" if winner else "NO_V35_ROBUST_DEVELOPMENT_CANDIDATE"
    return {
        "version": "v0.35",
        "decision": state,
        "winner": None if winner is None else winner.get("strategy"),
        "timeframe": "1d" if winner is not None else None,
        "development_venues": list(DEVELOPMENT_VENUES_V35),
        "reserved_holdout_venue": RESERVED_HOLDOUT_VENUE_V35,
        "kraken_touched": False,
        "threshold_relaxation": False,
        "rejected_candidate_retuning": False,
        "winner_reselection": False,
        "forward_paper_candidate_authorized": False,
        "paper_replacement_authorized": False,
        "live_execution_authorized": False,
    }
