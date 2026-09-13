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
