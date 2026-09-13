from __future__ import annotations

"""Fail-closed integrity layer for the amended v0.51 prospective experiment.

The original arbitration implementation remains frozen. This module enforces
identity/support/numerical invariants around it and assigns the amended
prospective calendar after the production predictor identity was made unique.
"""

import hashlib
import json

import numpy as np
import pandas as pd

from research_bot.overlap_arbitration_v51 import (
    ALLOWED_VENUES_V51,
    N_BLOCKS_V51,
    route_decision_v51,
)

AMENDED_PROSPECTIVE_START_V51 = pd.Timestamp("2026-09-13T12:00:00Z")
BLOCK_DAYS_V51 = 30
IDENTITY_COLUMNS_V51 = ("venue", "symbol", "series_id", "signal_time", "entry_time")


def assign_amended_prospective_blocks_v51(frame: pd.DataFrame) -> pd.DataFrame:
    if "signal_time" not in frame:
        raise ValueError("signal_time required")
    x = frame.copy()
    x["signal_time"] = pd.to_datetime(x["signal_time"], utc=True, errors="coerce")
    if x["signal_time"].isna().any():
        raise ValueError("invalid signal_time")
    end = AMENDED_PROSPECTIVE_START_V51 + pd.Timedelta(days=BLOCK_DAYS_V51 * N_BLOCKS_V51)
    if (x["signal_time"] < AMENDED_PROSPECTIVE_START_V51).any() or (x["signal_time"] >= end).any():
        raise ValueError("evidence outside amended v0.51 prospective window")
    elapsed = (x["signal_time"] - AMENDED_PROSPECTIVE_START_V51).dt.total_seconds() / 86_400.0
    x["prospective_block_v51"] = (elapsed // BLOCK_DAYS_V51).astype("int8") + 1
    return x


def candidate_identity_digest_v51(frame: pd.DataFrame) -> str:
    missing = sorted(set(IDENTITY_COLUMNS_V51).difference(frame.columns))
    if missing:
        raise ValueError(f"candidate identity missing columns: {missing}")
    x = frame.loc[:, IDENTITY_COLUMNS_V51].copy()
    x["venue"] = x["venue"].astype(str).str.lower()
    x["symbol"] = x["symbol"].astype(str).str.upper()
    x["series_id"] = x["series_id"].astype(str)
    for col in ("signal_time", "entry_time"):
        x[col] = pd.to_datetime(x[col], utc=True, errors="coerce")
        if x[col].isna().any():
            raise ValueError(f"invalid {col}")
        x[col] = x[col].map(lambda ts: ts.isoformat())
    if x.duplicated(list(IDENTITY_COLUMNS_V51), keep=False).any():
        raise ValueError("candidate identity contains duplicates")
    records = x.sort_values(list(IDENTITY_COLUMNS_V51), kind="mergesort").to_dict("records")
    raw = json.dumps(records, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def assert_common_candidate_set_v51(a0: pd.DataFrame, a1: pd.DataFrame) -> str:
    d0 = candidate_identity_digest_v51(a0)
    d1 = candidate_identity_digest_v51(a1)
    if d0 != d1:
        raise ValueError("A0/A1 candidate identity digest mismatch")
    return d0


def validate_venue_block_support_v51(support: pd.DataFrame) -> None:
    required = {"venue", "prospective_block_v51", "evidence_present", "conflict_cohorts"}
    missing = sorted(required.difference(support.columns))
    if missing:
        raise ValueError(f"support matrix missing columns: {missing}")
    x = support.copy()
    x["venue"] = x["venue"].astype(str).str.lower()
    x["prospective_block_v51"] = pd.to_numeric(x["prospective_block_v51"], errors="coerce")
    x["conflict_cohorts"] = pd.to_numeric(x["conflict_cohorts"], errors="coerce")
    if x[["prospective_block_v51", "conflict_cohorts"]].isna().any().any():
        raise ValueError("support matrix contains non-numeric cells")
    if not np.isfinite(x["conflict_cohorts"].to_numpy(dtype=float)).all() or (x["conflict_cohorts"] < 0).any():
        raise ValueError("support matrix has invalid conflict counts")
    if x["evidence_present"].isna().any() or not x["evidence_present"].map(type).eq(bool).all():
        raise ValueError("support matrix requires boolean evidence_present")
    expected = {(venue, block) for venue in ALLOWED_VENUES_V51 for block in range(1, N_BLOCKS_V51 + 1)}
    observed = set(zip(x["venue"], x["prospective_block_v51"].astype(int)))
    if x.duplicated(["venue", "prospective_block_v51"], keep=False).any():
        raise ValueError("support matrix contains duplicate venue/block rows")
    if observed != expected:
        raise ValueError("support matrix does not contain all frozen venue/block cells")
    if (~x["evidence_present"]).any():
        raise ValueError("missing venue/block evidence fails closed")


def safe_route_decision_v51(**kwargs) -> str:
    """Wrap the frozen routing rule with a non-finite-metric guard."""
    numeric_names = (
        "expectancy_deltas",
        "aggregate_expectancy_a0",
        "aggregate_expectancy_a1",
        "profit_factor_a0",
        "profit_factor_a1",
        "stress_profit_factor_a0",
        "stress_profit_factor_a1",
        "worst_drawdown_a1",
    )
    values: list[float] = []
    for name in numeric_names:
        value = kwargs[name]
        if isinstance(value, (list, tuple, np.ndarray)):
            values.extend(float(v) for v in value)
        else:
            values.append(float(value))
    if not np.isfinite(np.asarray(values, dtype=float)).all():
        return "V51_ARBITRATION_NOT_SUPPORTED"
    return route_decision_v51(**kwargs)
