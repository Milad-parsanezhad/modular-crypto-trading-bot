from __future__ import annotations

"""Prospectively frozen, causal overlap arbitration for v0.51.

The policy may rank only candidates that become executable at the same
``entry_time`` while the venue/symbol slot is flat.  A later signal never
pre-empts an already admitted trade.  Realized outcomes are deliberately not
part of the priority key.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


ARBITRATION_POLICY_V51 = "SIMULTANEOUS_MAX_EXPECTED_R_NO_PREEMPTION"
PREREGISTRATION_COMMIT_V51 = "d8ee4576aaf55750dd5910cc0d3b2efcbba3f5b2"
PREREGISTRATION_TIME_V51 = pd.Timestamp("2026-09-13T04:49:28Z")
PROSPECTIVE_START_V51 = pd.Timestamp("2026-09-13T08:00:00Z")
BLOCK_DAYS_V51 = 30
N_BLOCKS_V51 = 5
ALLOWED_VENUES_V51 = frozenset({"coinex", "okx", "kucoin"})
ALLOWED_ASSETS_V51 = frozenset({"BTC", "ETH", "SOL", "XRP", "DOGE"})
REQUIRED_COLUMNS_V51 = {
    "venue",
    "symbol",
    "series_id",
    "signal_time",
    "entry_time",
    "exit_time",
    "expected_r_v47",
    "selected_model",
}


@dataclass(frozen=True)
class ArbitrationAuditV51:
    input_candidates: int
    admitted_candidates: int
    simultaneous_conflict_cohorts: int
    simultaneous_candidates: int
    blocked_later_candidates: int


def assign_prospective_blocks_v51(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach one of the five immutable 30-day prospective block numbers."""

    if "signal_time" not in frame or "venue" not in frame or "symbol" not in frame:
        raise ValueError("v0.51 prospective frame requires signal_time, venue and symbol")
    x = frame.copy()
    x["signal_time"] = pd.to_datetime(x["signal_time"], utc=True, errors="coerce")
    if x["signal_time"].isna().any():
        raise ValueError("v0.51 prospective frame has invalid signal_time")
    end = PROSPECTIVE_START_V51 + pd.Timedelta(days=BLOCK_DAYS_V51 * N_BLOCKS_V51)
    if (x["signal_time"] < PROSPECTIVE_START_V51).any() or (x["signal_time"] >= end).any():
        raise ValueError("v0.51 evidence lies outside the frozen prospective window")
    venues = set(x["venue"].astype(str).str.lower())
    assets = set(x["symbol"].astype(str).str.upper().str.replace("USDT", "", regex=False))
    if not venues.issubset(ALLOWED_VENUES_V51) or not assets.issubset(ALLOWED_ASSETS_V51):
        raise ValueError("v0.51 evidence contains a forbidden venue or asset")
    elapsed_days = (x["signal_time"] - PROSPECTIVE_START_V51).dt.total_seconds() / 86_400.0
    x["prospective_block_v51"] = (elapsed_days // BLOCK_DAYS_V51).astype("int8") + 1
    return x


def route_decision_v51(
    *,
    conflict_counts: list[int],
    expectancy_deltas: list[float],
    aggregate_expectancy_a0: float,
    aggregate_expectancy_a1: float,
    profit_factor_a0: float,
    profit_factor_a1: float,
    stress_profit_factor_a0: float,
    stress_profit_factor_a1: float,
    worst_drawdown_a1: float,
    causal_checks_passed: bool,
) -> str:
    """Apply only the frozen v0.51 advancement gate; never authorize trading."""

    if (
        len(conflict_counts) != N_BLOCKS_V51
        or len(expectancy_deltas) != N_BLOCKS_V51
        or sum(conflict_counts) < 100
        or any(n < 10 for n in conflict_counts)
    ):
        return "V51_INSUFFICIENT_PROSPECTIVE_SUPPORT"
    numeric = np.asarray(
        expectancy_deltas
        + [aggregate_expectancy_a0, aggregate_expectancy_a1, profit_factor_a0,
           profit_factor_a1, stress_profit_factor_a0, stress_profit_factor_a1,
           worst_drawdown_a1],
        dtype=float,
    )
    if not causal_checks_passed or np.isnan(numeric).any():
        return "V51_ARBITRATION_NOT_SUPPORTED"
    supported = (
        sum(delta > 0.0 for delta in expectancy_deltas) >= 3
        and float(np.median(expectancy_deltas)) > 0.0
        and aggregate_expectancy_a1 > aggregate_expectancy_a0
        and profit_factor_a1 >= profit_factor_a0
        and stress_profit_factor_a1 >= stress_profit_factor_a0
        and worst_drawdown_a1 >= -0.05
    )
    return "V51_ARBITRATION_ADVANCEMENT_SUPPORTED" if supported else "V51_ARBITRATION_NOT_SUPPORTED"


def _validated_candidates(selected: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(REQUIRED_COLUMNS_V51.difference(selected.columns))
    if missing:
        raise ValueError(f"v0.51 arbitration missing columns: {missing}")
    x = selected.copy()
    for col in ("signal_time", "entry_time", "exit_time"):
        x[col] = pd.to_datetime(x[col], utc=True, errors="coerce")
        if x[col].isna().any():
            raise ValueError(f"v0.51 arbitration has invalid {col}")
    x["expected_r_v47"] = pd.to_numeric(x["expected_r_v47"], errors="coerce")
    if not np.isfinite(x["expected_r_v47"].to_numpy(dtype=float)).all():
        raise ValueError("v0.51 arbitration requires finite expected_r_v47")
    if x["selected_model"].isna().any() or not x["selected_model"].map(type).eq(bool).all():
        raise ValueError("v0.51 arbitration requires boolean selected_model")
    if (~x["selected_model"]).any() or (x["expected_r_v47"] <= 0.0).any():
        raise ValueError("v0.51 arbitration accepts only frozen positive-admission candidates")
    if (x["signal_time"] >= x["entry_time"]).any():
        raise ValueError("v0.51 arbitration requires signal_time < entry_time")
    if (x["exit_time"] < x["entry_time"]).any():
        raise ValueError("v0.51 arbitration requires exit_time >= entry_time")

    identity = ["venue", "symbol", "series_id", "signal_time", "entry_time"]
    if x.duplicated(identity, keep=False).any():
        raise ValueError("v0.51 arbitration candidate identity is not unique")
    return x


def arbitrate_overlaps_v51(selected: pd.DataFrame) -> tuple[pd.DataFrame, ArbitrationAuditV51]:
    """Apply the frozen causal v0.51 policy and return admitted rows plus audit.

    Ranking is confined to a same-entry-time cohort.  The highest frozen
    ``expected_r_v47`` wins; ties are resolved solely by decision-time identity.
    Once admitted, a trade blocks later entries through its recorded exit.
    """

    if selected.empty:
        return selected.copy(), ArbitrationAuditV51(0, 0, 0, 0, 0)
    x = _validated_candidates(selected)
    admitted: list[pd.Series] = []
    conflict_cohorts = simultaneous_candidates = blocked_later = 0

    for (_, _), group in x.groupby(["venue", "symbol"], sort=True):
        blocked_until = pd.Timestamp.min.tz_localize("UTC")
        for entry_time, cohort in group.groupby("entry_time", sort=True):
            if entry_time <= blocked_until:
                blocked_later += int(len(cohort))
                continue
            if len(cohort) > 1:
                conflict_cohorts += 1
                simultaneous_candidates += int(len(cohort))
            ranked = cohort.sort_values(
                ["expected_r_v47", "signal_time", "series_id"],
                ascending=[False, True, True],
                kind="mergesort",
            )
            winner = ranked.iloc[0]
            admitted.append(winner)
            blocked_until = winner["exit_time"]

    out = pd.DataFrame(admitted, columns=x.columns).reset_index(drop=True)
    audit = ArbitrationAuditV51(
        input_candidates=int(len(x)),
        admitted_candidates=int(len(out)),
        simultaneous_conflict_cohorts=conflict_cohorts,
        simultaneous_candidates=simultaneous_candidates,
        blocked_later_candidates=blocked_later,
    )
    return out, audit
