from __future__ import annotations

import pandas as pd

from research_bot.future_evidence_v25 import (
    V25FutureEvidenceContract,
    allocator_gate,
    assert_no_preboundary_evidence,
    future_event_slice,
    last_safe_completed_4h_open,
    maturity_state,
)


def test_last_safe_completed_bar_excludes_in_progress_candle():
    assert last_safe_completed_4h_open(pd.Timestamp("2026-09-11T12:37:00Z")) == pd.Timestamp("2026-09-11T08:00:00Z")
    assert last_safe_completed_4h_open(pd.Timestamp("2026-09-11T16:01:00Z")) == pd.Timestamp("2026-09-11T12:00:00Z")


def test_future_slice_never_accepts_preboundary_signal():
    c = V25FutureEvidenceContract()
    rows = pd.DataFrame([
        {"strategy": "H4_S6_BREAKOUT", "symbol": "ATOM/USDT", "signal_time": "2026-09-11T08:00:00Z", "entry_time": "2026-09-11T12:00:00Z", "exit_time": "2026-09-11T16:00:00Z"},
        {"strategy": "H4_S6_BREAKOUT", "symbol": "ATOM/USDT", "signal_time": "2026-09-11T12:00:00Z", "entry_time": "2026-09-11T16:00:00Z", "exit_time": "2026-09-11T20:00:00Z"},
        {"strategy": "H4_S6_BREAKOUT", "symbol": "ATOM/USDT", "signal_time": "2026-09-11T16:00:00Z", "entry_time": "2026-09-11T20:00:00Z", "exit_time": "2026-09-12T04:00:00Z"},
    ])
    out = future_event_slice(rows, last_completed_bar_open=pd.Timestamp("2026-09-11T20:00:00Z"), contract=c)
    assert len(out) == 1
    assert out.iloc[0]["signal_time"] == pd.Timestamp("2026-09-11T12:00:00Z")
    assert_no_preboundary_evidence(out, c)


def test_maturity_fails_closed_before_minimums():
    c = V25FutureEvidenceContract()
    state = maturity_state(
        venue_results={
            "okx": {"usable_symbols": 11, "future_events": 30, "ranked_accepted": 10},
            "kucoin": {"usable_symbols": 11, "future_events": 31, "ranked_accepted": 9},
        },
        last_completed_bar_open=pd.Timestamp("2026-09-12T12:00:00Z"),
        contract=c,
    )
    assert state["mature"] is False
    assert state["state"] == "INSUFFICIENT_FUTURE_SAMPLE"


def test_allocator_gate_requires_incremental_portfolio_value_and_dd():
    c = V25FutureEvidenceContract()
    baseline = {"total_realized_return": 0.01}
    ranked = {
        "profit_factor": 1.20,
        "total_realized_return": 0.02,
        "max_intrabar_stress_drawdown": -0.04,
        "hard_mtm_kill_triggered": False,
    }
    assert allocator_gate(baseline, ranked, contract=c)["passed"] is True
    ranked_bad = dict(ranked, max_intrabar_stress_drawdown=-0.051)
    assert allocator_gate(baseline, ranked_bad, contract=c)["passed"] is False
