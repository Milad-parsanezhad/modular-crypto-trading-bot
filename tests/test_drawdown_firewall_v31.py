from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from research_bot.drawdown_firewall_v31 import (
    TOTAL_EFFECTIVE_TRIALS_V31,
    DrawdownFirewallV31,
    _drawdown_headroom_cash,
    apply_drawdown_firewall_v31,
    preregistration_manifest_v31,
)
from research_bot.multitimeframe_strategies_v20 import RiskPsychologyPolicy
from research_bot.portfolio_allocator_v25 import PortfolioRiskBudgetV25


@dataclass(frozen=True)
class Spec:
    timeframe: str = "1d"


def ledger(rows: list[dict]) -> pd.DataFrame:
    base = pd.Timestamp("2025-01-01", tz="UTC")
    out = []
    for i, row in enumerate(rows):
        entry_time = row.get("entry_time", base + pd.Timedelta(days=i * 2))
        exit_time = row.get("exit_time", entry_time + pd.Timedelta(days=1))
        out.append(
            {
                "symbol": row.get("symbol", f"S{i}/USDT"),
                "entry_time": entry_time,
                "exit_time": exit_time,
                "r_multiple": row.get("r_multiple", -1.024),
                "side": row.get("side", 1),
                "entry": row.get("entry", 100.0),
                "stop": row.get("stop", 99.0),
            }
        )
    return pd.DataFrame(out)


def test_preregistration_keeps_five_percent_gate_and_kucoin_reserved() -> None:
    m = preregistration_manifest_v31()
    assert m["candidate_count"] == 12
    assert m["total_effective_trials"] == TOTAL_EFFECTIVE_TRIALS_V31 == 108
    assert m["hard_drawdown_cap"] == 0.05
    assert m["aggregate_open_risk_cap"] == 0.02
    assert m["directional_open_risk_cap"] == 0.015
    assert m["reserved_holdout_venue"] == "kucoin"
    assert "kucoin" not in m["development_venues"]
    assert m["alpha_parameter_retuning"] is False
    assert m["qualification_threshold_relaxation"] is False
    assert m["live_execution_authorized"] is False


def test_headroom_formula_is_relative_to_realized_peak() -> None:
    assert np.isclose(_drawdown_headroom_cash(equity=0.97, peak=1.0, cap=0.05), 0.02)
    assert np.isclose(_drawdown_headroom_cash(equity=1.10, peak=1.10, cap=0.05), 0.055)
    assert _drawdown_headroom_cash(equity=0.94, peak=1.0, cap=0.05) == 0.0


def test_firewall_prevents_sequential_stop_losses_from_crossing_five_percent() -> None:
    rows = [{"r_multiple": -1.024, "symbol": f"S{i}/USDT"} for i in range(40)]
    out = apply_drawdown_firewall_v31(ledger(rows), Spec())
    settled = pd.to_numeric(out["settlement_drawdown_v25"], errors="coerce").dropna()
    assert len(settled) > 0
    assert float(settled.min()) >= -0.05 - 1e-12
    assert out["reject_reason_v25"].astype(str).isin(["DRAWDOWN_FIREWALL_EXHAUSTED", "HARD_DRAWDOWN_KILL", ""]).any()


def test_near_floor_entry_is_scaled_by_remaining_headroom() -> None:
    # Four prior stop losses move realized equity close to the hard floor; the
    # next proposal must be smaller than its nominal v0.25 risk request.
    rows = [{"r_multiple": -1.024, "symbol": f"L{i}/USDT"} for i in range(5)]
    out = apply_drawdown_firewall_v31(ledger(rows), Spec())
    executed = out[out["executed_v25"] == True]  # noqa: E712
    assert len(executed) >= 2
    assert bool(executed["firewall_constrained_v31"].fillna(False).any())
    constrained = executed[executed["firewall_constrained_v31"] == True]  # noqa: E712
    assert (constrained["portfolio_allocation_scale_v25"] < 1.0).all()


def test_simultaneous_batch_is_symbol_order_invariant() -> None:
    t = pd.Timestamp("2025-03-01", tz="UTC")
    rows = [
        {"symbol": "A/USDT", "entry_time": t, "exit_time": t + pd.Timedelta(days=1), "r_multiple": -1.024, "side": 1},
        {"symbol": "B/USDT", "entry_time": t, "exit_time": t + pd.Timedelta(days=1), "r_multiple": -1.024, "side": 1},
        {"symbol": "C/USDT", "entry_time": t, "exit_time": t + pd.Timedelta(days=1), "r_multiple": -1.024, "side": -1},
    ]
    a = apply_drawdown_firewall_v31(ledger(rows), Spec()).set_index("symbol")
    b = apply_drawdown_firewall_v31(ledger(list(reversed(rows))), Spec()).set_index("symbol")
    for symbol in ["A/USDT", "B/USDT", "C/USDT"]:
        assert np.isclose(a.loc[symbol, "allocated_stop_loss_budget_cash_v25"], b.loc[symbol, "allocated_stop_loss_budget_cash_v25"])


def test_future_rows_do_not_change_past_allocations() -> None:
    x = ledger([{"r_multiple": -1.024} for _ in range(12)])
    cutoff = x.loc[6, "entry_time"]
    a = apply_drawdown_firewall_v31(x, Spec())
    y = x.copy()
    mask = y["entry_time"] > cutoff
    y.loc[mask, "r_multiple"] = 3.0
    y.loc[mask, "stop"] = 95.0
    b = apply_drawdown_firewall_v31(y, Spec())
    cols = [
        "executed_v25",
        "allocated_stop_loss_budget_cash_v25",
        "allocated_risk_fraction_v25",
        "drawdown_before_v25",
        "firewall_available_cash_before_v31",
    ]
    am = a["entry_time"] <= cutoff
    bm = b["entry_time"] <= cutoff
    pd.testing.assert_frame_equal(
        a.loc[am, cols].reset_index(drop=True),
        b.loc[bm, cols].reset_index(drop=True),
    )


def test_firewall_cap_must_match_frozen_risk_policy() -> None:
    with np.testing.assert_raises_regex(ValueError, "must equal the frozen"):
        apply_drawdown_firewall_v31(
            ledger([{}]),
            Spec(),
            risk_policy=RiskPsychologyPolicy(hard_drawdown_kill=0.05),
            budget=PortfolioRiskBudgetV25(),
            firewall=DrawdownFirewallV31(hard_drawdown_cap=0.06),
        )
