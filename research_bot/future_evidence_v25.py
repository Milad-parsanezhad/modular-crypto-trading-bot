from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from research_bot.v24c_external_plan import EXTERNAL_BYBIT_SYMBOLS_V24C


@dataclass(frozen=True)
class V25FutureEvidenceContract:
    future_start_utc: str = "2026-09-11T12:00:00Z"
    venues: tuple[str, ...] = ("okx", "kucoin")
    symbols: tuple[str, ...] = EXTERNAL_BYBIT_SYMBOLS_V24C
    context_days: int = 300
    max_bars: int = 2600
    min_context_bars: int = 1200
    max_missing_fraction: float = 0.02
    min_usable_symbols_per_venue: int = 5
    min_elapsed_hours: int = 168
    min_future_events_per_venue: int = 200
    min_ranked_accepted_per_venue: int = 50
    min_profit_factor: float = 1.05
    max_mtm_drawdown: float = 0.05
    min_incremental_return: float = 0.0
    required_cost_stress_bps: float = 36.0
    forward_paper_authorized: bool = False
    paper_replacement_authorized: bool = False
    live_execution_authorized: bool = False

    @property
    def start(self) -> pd.Timestamp:
        return pd.Timestamp(self.future_start_utc)

    def to_dict(self) -> dict:
        return asdict(self)


def last_safe_completed_4h_open(now: pd.Timestamp | None = None) -> pd.Timestamp:
    """Return the opening timestamp of the latest certainly completed 4h bar.

    Public OHLCV APIs timestamp candles by bar open. At 12:xx UTC, for example,
    the 08:00 candle is complete while the 12:00 candle is in progress. Returning
    floor(now, 4h) - 4h prevents an in-progress bar from entering a future read.
    """
    t = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return t.floor("4h") - pd.Timedelta(hours=4)


def context_start(contract: V25FutureEvidenceContract | None = None) -> pd.Timestamp:
    c = contract or V25FutureEvidenceContract()
    return c.start - pd.Timedelta(days=c.context_days)


def estimated_missing_fraction(frame: pd.DataFrame, *, start: pd.Timestamp, end: pd.Timestamp) -> float:
    if frame.empty:
        return 1.0
    x = frame.copy()
    x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True)
    clipped = x[(x["timestamp"] >= start) & (x["timestamp"] <= end)].drop_duplicates("timestamp")
    expected = max(1, int((end - start) / pd.Timedelta(hours=4)) + 1)
    return float(max(0.0, 1.0 - len(clipped) / expected))


def future_event_slice(
    events: pd.DataFrame,
    *,
    last_completed_bar_open: pd.Timestamp,
    contract: V25FutureEvidenceContract | None = None,
) -> pd.DataFrame:
    c = contract or V25FutureEvidenceContract()
    if events.empty:
        return events.copy()
    x = events.copy()
    for col in ("signal_time", "entry_time", "exit_time"):
        x[col] = pd.to_datetime(x[col], utc=True)
    # A signal must itself be born after the prospective boundary. The entire
    # exit bar must also be completed before the event may enter the evidence set.
    x = x[(x["signal_time"] >= c.start) & (x["exit_time"] <= last_completed_bar_open)].copy()
    return x.sort_values(["signal_time", "strategy", "symbol"], kind="mergesort").reset_index(drop=True)


def assert_no_preboundary_evidence(events: pd.DataFrame, contract: V25FutureEvidenceContract | None = None) -> None:
    c = contract or V25FutureEvidenceContract()
    if events.empty:
        return
    signal = pd.to_datetime(events["signal_time"], utc=True)
    if bool((signal < c.start).any()):
        raise RuntimeError("V25_PREBOUNDARY_EVIDENCE_DETECTED")


def maturity_state(
    *,
    venue_results: Mapping[str, Mapping],
    last_completed_bar_open: pd.Timestamp,
    contract: V25FutureEvidenceContract | None = None,
) -> dict:
    c = contract or V25FutureEvidenceContract()
    if last_completed_bar_open < c.start:
        return {
            "state": "WAITING_FOR_FUTURE_BOUNDARY",
            "elapsed_hours": 0.0,
            "mature": False,
            "reasons": ["future boundary has not produced a completed 4h bar yet"],
        }
    elapsed = float((last_completed_bar_open - c.start) / pd.Timedelta(hours=1))
    reasons: list[str] = []
    if elapsed < c.min_elapsed_hours:
        reasons.append(f"elapsed_hours<{c.min_elapsed_hours}")
    for venue in c.venues:
        r = venue_results.get(venue, {})
        if int(r.get("usable_symbols", 0)) < c.min_usable_symbols_per_venue:
            reasons.append(f"{venue}:usable_symbols<{c.min_usable_symbols_per_venue}")
        if int(r.get("future_events", 0)) < c.min_future_events_per_venue:
            reasons.append(f"{venue}:future_events<{c.min_future_events_per_venue}")
        if int(r.get("ranked_accepted", 0)) < c.min_ranked_accepted_per_venue:
            reasons.append(f"{venue}:ranked_accepted<{c.min_ranked_accepted_per_venue}")
    return {
        "state": "FUTURE_SAMPLE_MATURE_FOR_SCIENTIFIC_GATE" if not reasons else "INSUFFICIENT_FUTURE_SAMPLE",
        "elapsed_hours": elapsed,
        "mature": not reasons,
        "reasons": reasons,
    }


def allocator_gate(
    baseline: Mapping,
    ranked: Mapping,
    *,
    contract: V25FutureEvidenceContract | None = None,
) -> dict:
    """Frozen portfolio gate used only after the prospective sample matures."""
    c = contract or V25FutureEvidenceContract()
    ranked_pf = float(ranked.get("profit_factor", np.nan))
    ranked_return = float(ranked.get("total_realized_return", np.nan))
    base_return = float(baseline.get("total_realized_return", np.nan))
    dd = float(ranked.get("max_intrabar_stress_drawdown", np.nan))
    gates = {
        "profit_factor": np.isfinite(ranked_pf) and ranked_pf >= c.min_profit_factor,
        "positive_return": np.isfinite(ranked_return) and ranked_return > 0,
        "incremental_return": np.isfinite(ranked_return) and np.isfinite(base_return) and ranked_return > base_return + c.min_incremental_return,
        "mtm_drawdown": np.isfinite(dd) and abs(dd) <= c.max_mtm_drawdown,
        "hard_kill_not_triggered": ranked.get("hard_mtm_kill_triggered") is False,
    }
    return {"passed": bool(all(gates.values())), "gates": gates}
