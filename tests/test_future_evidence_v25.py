from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd

from research_bot.ccxt_external_v24c import fetch_ccxt_spot_ohlcv
from research_bot.future_evidence_v25 import (
    V25FutureEvidenceContract,
    allocator_gate,
    assert_no_preboundary_evidence,
    future_event_slice,
    future_state_is_terminal,
    last_safe_completed_4h_open,
    maturity_state,
)
from research_bot.prospective_chain_v25 import merge_append_only, stress_r_multiple


def test_last_safe_completed_bar_excludes_in_progress_candle():
    assert last_safe_completed_4h_open(pd.Timestamp("2026-09-11T12:37:00Z")) == pd.Timestamp("2026-09-11T08:00:00Z")
    assert last_safe_completed_4h_open(pd.Timestamp("2026-09-11T16:01:00Z")) == pd.Timestamp("2026-09-11T12:00:00Z")
    assert last_safe_completed_4h_open(pd.Timestamp("2026-09-11T20:23:00Z")) == pd.Timestamp("2026-09-11T16:00:00Z")


def test_future_slice_never_accepts_preboundary_signal():
    c = V25FutureEvidenceContract()
    rows = pd.DataFrame([
        {"strategy": "H4_S6_BREAKOUT", "symbol": "ATOM/USDT", "signal_time": "2026-09-11T12:00:00Z", "entry_time": "2026-09-11T16:00:00Z", "exit_time": "2026-09-11T20:00:00Z"},
        {"strategy": "H4_S6_BREAKOUT", "symbol": "ATOM/USDT", "signal_time": "2026-09-11T16:00:00Z", "entry_time": "2026-09-11T20:00:00Z", "exit_time": "2026-09-12T00:00:00Z"},
        {"strategy": "H4_S6_BREAKOUT", "symbol": "ATOM/USDT", "signal_time": "2026-09-11T20:00:00Z", "entry_time": "2026-09-12T00:00:00Z", "exit_time": "2026-09-12T08:00:00Z"},
    ])
    out = future_event_slice(rows, last_completed_bar_open=pd.Timestamp("2026-09-12T00:00:00Z"), contract=c)
    assert len(out) == 1
    assert out.iloc[0]["signal_time"] == pd.Timestamp("2026-09-11T16:00:00Z")
    assert_no_preboundary_evidence(out, c)


def test_maturity_fails_closed_before_minimums_without_peeking_at_economics():
    c = V25FutureEvidenceContract()
    state = maturity_state(
        venue_results={
            "okx": {"usable_symbols": 11, "future_events": 30},
            "kucoin": {"usable_symbols": 11, "future_events": 31},
        },
        last_completed_bar_open=pd.Timestamp("2026-09-12T16:00:00Z"),
        contract=c,
    )
    assert state["mature"] is False
    assert state["state"] == "INSUFFICIENT_FUTURE_SAMPLE"
    assert all("accepted" not in reason for reason in state["reasons"])


def test_maturity_allows_first_look_only_after_time_coverage_and_event_counts():
    c = V25FutureEvidenceContract()
    state = maturity_state(
        venue_results={
            "okx": {"usable_symbols": 10, "future_events": 220},
            "kucoin": {"usable_symbols": 9, "future_events": 205},
        },
        last_completed_bar_open=pd.Timestamp("2026-09-18T16:00:00Z"),
        contract=c,
    )
    assert state["mature"] is True
    assert state["state"] == "FUTURE_SAMPLE_MATURE_FOR_FIRST_LOOK"


def test_allocator_gate_requires_portfolio_cost_cvar_breadth_and_paired_uncertainty():
    c = V25FutureEvidenceContract()
    baseline = {"total_realized_return": 0.01}
    ranked = {
        "accepted": 65,
        "profit_factor": 1.20,
        "total_realized_return": 0.02,
        "max_intrabar_stress_drawdown": -0.04,
        "max_rolling_cvar": 0.012,
        "hard_mtm_kill_triggered": False,
    }
    stress36 = {"profit_factor": 1.08, "mean_r_accepted": 0.05}
    ci = {"low": 0.00001, "high": 0.0009}
    assert allocator_gate(baseline, ranked, ranked_cost_stress=stress36, paired_uplift_ci=ci, contract=c)["passed"] is True

    ranked_bad_dd = dict(ranked, max_intrabar_stress_drawdown=-0.051)
    assert allocator_gate(baseline, ranked_bad_dd, ranked_cost_stress=stress36, paired_uplift_ci=ci, contract=c)["passed"] is False

    ranked_bad_cvar = dict(ranked, max_rolling_cvar=0.021)
    assert allocator_gate(baseline, ranked_bad_cvar, ranked_cost_stress=stress36, paired_uplift_ci=ci, contract=c)["passed"] is False

    bad_stress = {"profit_factor": 0.99, "mean_r_accepted": -0.01}
    assert allocator_gate(baseline, ranked, ranked_cost_stress=bad_stress, paired_uplift_ci=ci, contract=c)["passed"] is False

    bad_ci = {"low": -0.00001, "high": 0.001}
    assert allocator_gate(baseline, ranked, ranked_cost_stress=stress36, paired_uplift_ci=bad_ci, contract=c)["passed"] is False


def test_terminal_future_state_prevents_repeated_first_look():
    assert future_state_is_terminal({"state": "FUTURE_ALLOCATOR_GATE_PASS_PRE_SEARCH_AUDIT"}) is True
    assert future_state_is_terminal({"state": "FUTURE_ALLOCATOR_GATE_FAIL"}) is True
    assert future_state_is_terminal({"state": "INSUFFICIENT_FUTURE_SAMPLE"}) is False


def test_append_only_chain_preserves_first_observed_bars_and_flags_restatements():
    previous = pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-09-11T00:00:00Z", "2026-09-11T04:00:00Z"]),
        "open": [100.0, 101.0], "high": [102.0, 103.0], "low": [99.0, 100.0],
        "close": [101.0, 102.0], "volume": [10.0, 11.0], "source": ["old", "old"],
    })
    fresh = pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-09-11T00:00:00Z", "2026-09-11T04:00:00Z", "2026-09-11T08:00:00Z"]),
        "open": [100.5, 101.0, 102.0], "high": [102.0, 103.0, 104.0], "low": [99.0, 100.0, 101.0],
        "close": [101.5, 102.2, 103.0], "volume": [10.0, 11.0, 12.0], "source": ["new", "new", "new"],
    })
    merged, revisions = merge_append_only(previous, fresh)
    assert revisions == 2
    assert len(merged) == 3
    assert merged.loc[0, "open"] == 100.0
    assert merged.loc[1, "close"] == 102.0
    assert merged.loc[2, "timestamp"] == pd.Timestamp("2026-09-11T08:00:00Z")


def test_cost_stress_is_monotone_and_recomputes_r_from_gross_return():
    rows = pd.DataFrame({
        "entry": [100.0, 100.0],
        "stop": [98.0, 102.0],
        "gross_return": [0.02, 0.02],
    })
    r24 = stress_r_multiple(rows, 24.0).to_numpy()
    r36 = stress_r_multiple(rows, 36.0).to_numpy()
    r60 = stress_r_multiple(rows, 60.0).to_numpy()
    assert np.all(r24 > r36)
    assert np.all(r36 > r60)
    np.testing.assert_allclose(r24, [(0.02 - 0.0024) / 0.02] * 2)


def test_ccxt_pagination_advances_by_full_candle_boundary(monkeypatch):
    calls: list[int] = []

    class FakeExchange:
        rateLimit = 0
        timeout = 20_000

        def __init__(self, _config):
            pass

        def load_markets(self):
            return {"ATOM/USDT": {"spot": True}}

        def parse_timeframe(self, timeframe):
            assert timeframe == "4h"
            return 4 * 60 * 60

        def fetch_ohlcv(self, symbol, timeframe, since, limit):
            assert symbol == "ATOM/USDT"
            assert timeframe == "4h"
            calls.append(int(since))
            if len(calls) == 1:
                return [
                    [0, 100, 101, 99, 100, 10],
                    [14_400_000, 100, 102, 99, 101, 11],
                ]
            return [[28_800_000, 101, 103, 100, 102, 12]]

        def close(self):
            return None

    fake_ccxt = types.SimpleNamespace(fake=FakeExchange)
    monkeypatch.setitem(sys.modules, "ccxt", fake_ccxt)
    monkeypatch.setattr("research_bot.ccxt_external_v24c.time.sleep", lambda *_args, **_kwargs: None)

    out = fetch_ccxt_spot_ohlcv(
        "fake",
        "ATOM/USDT",
        timeframe="4h",
        start_ms=0,
        end_ms=28_800_000,
        max_bars=10,
        page_limit=2,
        max_pages=3,
    )
    assert calls == [0, 28_800_000]
    assert len(out) == 3
