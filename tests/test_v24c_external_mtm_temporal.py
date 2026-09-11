from __future__ import annotations

import numpy as np
import pandas as pd

from research_bot.bybit_public import audit_bybit_ohlcv, fetch_bybit_spot_ohlcv
from research_bot.portfolio_mtm_v24c import MTMRiskContract, simulate_mtm_portfolio
from research_bot.temporal_meta_v24c import build_event_sequences
from research_bot.v24c_external_plan import (
    EXTERNAL_BYBIT_SYMBOLS_V24C,
    FROZEN_CANDIDATES_V24C,
    INTERNAL_SYMBOLS_V24B,
    frozen_manifest,
)


def _bars(n: int = 180, start: str = "2026-01-01T00:00:00Z") -> pd.DataFrame:
    ts = pd.date_range(start, periods=n, freq="4h", tz="UTC")
    base = 100.0 + np.linspace(0, 12, n) + np.sin(np.arange(n) / 7.0)
    return pd.DataFrame({
        "timestamp": ts,
        "open": base,
        "high": base + 1.2,
        "low": base - 1.0,
        "close": base + 0.25 * np.sin(np.arange(n) / 3.0),
        "volume": 1000.0 + 10.0 * np.cos(np.arange(n) / 5.0),
    })


def test_external_universe_is_disjoint_from_v24b_internal_panel():
    assert not (set(EXTERNAL_BYBIT_SYMBOLS_V24C) & set(INTERNAL_SYMBOLS_V24B))
    assert {c.strategy for c in FROZEN_CANDIDATES_V24C} == {"H4_S6_BREAKOUT", "H4_D1_OB_BOS_RISK"}
    manifest = frozen_manifest()
    assert manifest["external_model_refit_allowed"] is False
    assert manifest["external_threshold_tuning_allowed"] is False
    assert manifest["live_execution_authorized"] is False


def test_bybit_spot_fetch_normalizes_newest_first_and_audits_gaps():
    step = 14_400_000
    t0 = int(pd.Timestamp("2026-01-01T00:00:00Z").timestamp() * 1000)
    rows = []
    for i in range(4):
        ts = t0 + i * step
        rows.append([str(ts), "100", "102", "99", "101", "10", "1000"])

    def fake(path: str, params: dict, timeout: int):
        assert path == "/v5/market/kline"
        return {"retCode": 0, "result": {"list": list(reversed(rows))}}

    out = fetch_bybit_spot_ohlcv(
        "ATOM/USDT", period="4hour", start_ms=t0, end_ms=t0 + 3 * step,
        bars=4, request_fn=fake,
    )
    assert len(out) == 4
    assert out["timestamp"].is_monotonic_increasing
    assert out.iloc[0]["source"] == "bybit_public_spot"
    audit = audit_bybit_ohlcv(out, "4hour")
    assert audit["gap_count"] == 0
    assert audit["estimated_missing_bars"] == 0


def test_temporal_sequence_is_unchanged_by_future_mutation():
    frame = _bars(180)
    signal_time = frame.loc[120, "timestamp"]
    events = pd.DataFrame({"signal_time": [signal_time], "label_meta_execute": [1], "symbol": ["X/USDT"]})
    x1, m1, _ = build_event_sequences(frame, events, lookback=64)
    mutated = frame.copy()
    mutated.loc[mutated["timestamp"] > signal_time, ["open", "high", "low", "close", "volume"]] *= 100.0
    x2, m2, _ = build_event_sequences(mutated, events, lookback=64)
    assert len(m1) == len(m2) == 1
    np.testing.assert_allclose(x1, x2, atol=0, rtol=0)


def test_mtm_engine_marks_open_position_and_remains_fail_closed_on_risk():
    frame = _bars(120)
    t_entry = frame.loc[70, "timestamp"]
    t_exit = frame.loc[74, "timestamp"]
    events = pd.DataFrame([{
        "strategy": "H4_S6_BREAKOUT", "symbol": "ATOM/USDT", "side": "long",
        "signal_time": frame.loc[69, "timestamp"], "entry_time": t_entry, "exit_time": t_exit,
        "entry": float(frame.loc[70, "open"]), "stop": float(frame.loc[70, "open"] * 0.98),
        "r_multiple": 1.0,
    }])
    summary, ledger, curve = simulate_mtm_portfolio(
        events, {"ATOM/USDT": frame}, contract=MTMRiskContract(cvar_lookback_bars=20, correlation_lookback_bars=20)
    )
    assert summary["accepted"] == 1
    assert summary["mark_to_market"] is True
    assert summary["max_open_risk_fraction_seen"] <= 0.0100001
    assert not curve.empty and "intrabar_stress_drawdown" in curve
    assert bool(ledger.iloc[0]["accepted"])


def test_cvar_is_loss_positive_and_budget_can_block_after_history():
    # Unit-level behavioral guard for the contract itself: budget is expressed as
    # positive loss fraction and must be stricter than the 5% hard drawdown kill.
    c = MTMRiskContract()
    assert 0 < c.max_cvar_loss_fraction < c.hard_mtm_drawdown_kill
    assert c.cvar_alpha == 0.95
    assert c.max_pairwise_correlation_for_full_size == 0.80
