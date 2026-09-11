from __future__ import annotations

"""Portfolio concurrency diagnostics for the post-causality audit.

v0.24 does not change signal logic, thresholds, or promotion gates.  It measures
how much *already-open* risk a candidate accumulates when several symbols overlap.
This is the first diagnostic required after v0.23 showed that entry-ordered risk
accounting materially distorted the historical tournament.

Risk-at-stop is expressed in normalized cash units:
    entry_equity * risk_fraction
and exposure is measured immediately after all accepted entries at a timestamp,
while positions with ``exit_time == timestamp`` are still considered open. This
matches the v0.23 causal contract: same-bar exits are not known at entry.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExposureSnapshot:
    segment: str
    timestamp: pd.Timestamp
    realized_equity: float
    open_positions: int
    open_symbols: int
    open_risk_cash: float
    open_risk_fraction_of_equity: float


def exposure_path_v24(causal_ledger: pd.DataFrame) -> pd.DataFrame:
    """Build post-entry exposure snapshots from an executed v0.23 ledger."""
    if causal_ledger.empty:
        return pd.DataFrame(
            columns=[
                "segment", "timestamp", "realized_equity", "open_positions",
                "open_symbols", "open_risk_cash", "open_risk_fraction_of_equity",
            ]
        )

    required = {
        "symbol", "entry_time", "exit_time", "executed_v23",
        "risk_fraction_v23", "equity_at_entry_v23",
    }
    missing = required - set(causal_ledger.columns)
    if missing:
        raise ValueError(f"v0.24 exposure audit missing columns: {sorted(missing)}")

    x = causal_ledger[causal_ledger["executed_v23"] == True].copy()  # noqa: E712
    if x.empty:
        return exposure_path_v24(pd.DataFrame())
    x["entry_time"] = pd.to_datetime(x["entry_time"], utc=True, errors="raise")
    x["exit_time"] = pd.to_datetime(x["exit_time"], utc=True, errors="raise")
    x["risk_fraction_v23"] = pd.to_numeric(x["risk_fraction_v23"], errors="raise")
    x["equity_at_entry_v23"] = pd.to_numeric(x["equity_at_entry_v23"], errors="raise")
    x["risk_cash_v24"] = x["equity_at_entry_v23"] * x["risk_fraction_v23"]
    if bool((x["risk_cash_v24"] < 0).any()) or bool((~np.isfinite(x["risk_cash_v24"])).any()):
        raise ValueError("invalid risk cash in causal ledger")

    if "segment" not in x:
        x["segment"] = "all"

    rows: list[dict[str, Any]] = []
    segment_order = [s for s in ("development", "validation", "test") if s in set(x["segment"].astype(str))]
    segment_order += sorted(set(x["segment"].astype(str)) - set(segment_order))

    for segment in segment_order:
        part = x[x["segment"].astype(str) == segment].sort_values(["entry_time", "symbol"], kind="mergesort")
        active = pd.DataFrame(columns=part.columns)
        for timestamp, entering in part.groupby("entry_time", sort=True):
            # Strict causality: an exit stamped on this same bar is still open
            # when entries on this bar are sized and accepted.
            if not active.empty:
                active = active[active["exit_time"] >= timestamp].copy()
            active = pd.concat([active, entering], ignore_index=True)

            equities = entering["equity_at_entry_v23"].to_numpy(dtype=float)
            if not np.allclose(equities, equities[0], rtol=0.0, atol=1e-12):
                raise ValueError("same-timestamp entries disagree on realized entry equity")
            realized_equity = float(equities[0])
            open_risk_cash = float(active["risk_cash_v24"].sum())
            risk_fraction = float(open_risk_cash / realized_equity) if realized_equity > 0 else np.inf
            rows.append(
                {
                    "segment": segment,
                    "timestamp": timestamp,
                    "realized_equity": realized_equity,
                    "open_positions": int(len(active)),
                    "open_symbols": int(active["symbol"].astype(str).nunique()),
                    "open_risk_cash": open_risk_cash,
                    "open_risk_fraction_of_equity": risk_fraction,
                }
            )

    return pd.DataFrame(rows)


def summarize_exposure_v24(path: pd.DataFrame) -> dict[str, float | int]:
    if path.empty:
        return {
            "exposure_snapshots": 0,
            "max_open_positions": 0,
            "p95_open_positions": 0.0,
            "max_open_symbols": 0,
            "max_open_risk_fraction": 0.0,
            "p95_open_risk_fraction": 0.0,
            "snapshots_open_risk_gt_1pct": 0,
            "snapshots_open_risk_gt_2pct": 0,
            "snapshots_open_risk_gt_5pct": 0,
        }

    risk = pd.to_numeric(path["open_risk_fraction_of_equity"], errors="raise").to_numpy(dtype=float)
    positions = pd.to_numeric(path["open_positions"], errors="raise").to_numpy(dtype=float)
    symbols = pd.to_numeric(path["open_symbols"], errors="raise").to_numpy(dtype=float)
    return {
        "exposure_snapshots": int(len(path)),
        "max_open_positions": int(np.max(positions)),
        "p95_open_positions": float(np.quantile(positions, 0.95)),
        "max_open_symbols": int(np.max(symbols)),
        "max_open_risk_fraction": float(np.max(risk)),
        "p95_open_risk_fraction": float(np.quantile(risk, 0.95)),
        "snapshots_open_risk_gt_1pct": int(np.sum(risk > 0.01)),
        "snapshots_open_risk_gt_2pct": int(np.sum(risk > 0.02)),
        "snapshots_open_risk_gt_5pct": int(np.sum(risk > 0.05)),
    }
