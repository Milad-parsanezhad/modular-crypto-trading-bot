from __future__ import annotations

import numpy as np
import pandas as pd

from research_bot.portfolio_allocator_v25 import PortfolioRiskBudgetV25, apply_portfolio_allocator_v25


def _trade(
    symbol: str,
    entry_time: str,
    exit_time: str,
    *,
    side: int = 1,
    r: float = 1.0,
    entry: float = 100.0,
    stop_fraction: float = 0.01,
    vol_scale: float = 1.0,
) -> dict:
    stop = entry * (1.0 - stop_fraction) if side > 0 else entry * (1.0 + stop_fraction)
    return {
        "strategy": "TEST",
        "family": "test",
        "timeframe": "4h",
        "symbol": symbol,
        "entry_time": entry_time,
        "exit_time": exit_time,
        "side": side,
        "entry": entry,
        "stop": stop,
        "r_multiple": r,
        "risk_scale_volatility": vol_scale,
        "segment": "validation",
    }


def test_cost_adjusted_stop_budget_includes_frozen_24bps_friction() -> None:
    ledger = pd.DataFrame([
        _trade("BTC/USDT", "2026-01-01T00:00:00Z", "2026-01-01T08:00:00Z", stop_fraction=0.01)
    ])
    out = apply_portfolio_allocator_v25(ledger, "4h")
    row = out.iloc[0]

    # 1% price stop + 0.24% round-trip friction => 1.24R stop loss.
    assert np.isclose(float(row["stop_loss_multiple_abs_v25"]), 1.24)
    assert np.isclose(float(row["proposed_risk_fraction_v25"]), 0.0025)
    assert np.isclose(float(row["proposed_stop_loss_budget_cash_v25"]), 0.0025 * 1.24)


def test_simultaneous_batch_respects_aggregate_and_directional_caps() -> None:
    rows = []
    for i in range(8):
        rows.append(_trade(f"L{i}/USDT", "2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z", side=1))
    for i in range(8):
        rows.append(_trade(f"S{i}/USDT", "2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z", side=-1))
    out = apply_portfolio_allocator_v25(pd.DataFrame(rows), "4h")

    assert bool(out["executed_v25"].all())
    assert float(out["open_risk_fraction_after_v25"].max()) <= 0.020 + 1e-12
    longs = out[out["side"] == 1]
    shorts = out[out["side"] == -1]
    assert float(longs["open_directional_risk_fraction_after_v25"].max()) <= 0.015 + 1e-12
    assert float(shorts["open_directional_risk_fraction_after_v25"].max()) <= 0.015 + 1e-12


def test_same_timestamp_allocation_is_permutation_invariant() -> None:
    rows = [
        _trade(f"C{i}/USDT", "2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z", side=1)
        for i in range(10)
    ]
    a = apply_portfolio_allocator_v25(pd.DataFrame(rows), "4h")
    b = apply_portfolio_allocator_v25(pd.DataFrame(list(reversed(rows))), "4h")

    aa = a.set_index("symbol")["allocated_stop_loss_budget_cash_v25"].sort_index()
    bb = b.set_index("symbol")["allocated_stop_loss_budget_cash_v25"].sort_index()
    assert np.allclose(aa.to_numpy(dtype=float), bb.to_numpy(dtype=float))


def test_unresolved_future_outcome_cannot_change_current_allocation() -> None:
    base = pd.DataFrame(
        [
            _trade("BTC/USDT", "2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z", r=-1.0),
            _trade("ETH/USDT", "2026-01-01T04:00:00Z", "2026-01-01T12:00:00Z", r=1.0),
        ]
    )
    alt = base.copy()
    alt.loc[0, "r_multiple"] = 3.0

    a = apply_portfolio_allocator_v25(base, "4h")
    b = apply_portfolio_allocator_v25(alt, "4h")
    eth_a = a[a["symbol"] == "ETH/USDT"].iloc[0]
    eth_b = b[b["symbol"] == "ETH/USDT"].iloc[0]

    assert np.isclose(float(eth_a["allocated_risk_fraction_v25"]), float(eth_b["allocated_risk_fraction_v25"]))
    assert np.isclose(float(eth_a["open_risk_fraction_before_v25"]), float(eth_b["open_risk_fraction_before_v25"]))


def test_exit_on_same_bar_is_still_open_for_entry_budget() -> None:
    ledger = pd.DataFrame(
        [
            _trade("BTC/USDT", "2026-01-01T00:00:00Z", "2026-01-01T04:00:00Z"),
            _trade("ETH/USDT", "2026-01-01T04:00:00Z", "2026-01-01T08:00:00Z"),
        ]
    )
    out = apply_portfolio_allocator_v25(ledger, "4h")
    eth = out[out["symbol"] == "ETH/USDT"].iloc[0]
    assert int(eth["open_positions_before_v25"]) == 1
    assert float(eth["open_risk_fraction_before_v25"]) > 0.0


def test_budget_scales_risk_instead_of_using_symbol_first_come_priority() -> None:
    ledger = pd.DataFrame(
        [
            _trade(f"A{i}/USDT", "2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z", side=1)
            for i in range(10)
        ]
    )
    out = apply_portfolio_allocator_v25(ledger, "4h")
    scales = out["portfolio_allocation_scale_v25"].to_numpy(dtype=float)

    assert np.all(scales > 0.0)
    assert np.all(scales < 1.0)
    assert np.allclose(scales, scales[0])


def test_inherited_directional_overage_freezes_new_same_side_entries() -> None:
    # Fill the long directional budget, keep those longs open, then realize a
    # short loss. Equity falls while long stop-risk cash is unchanged, so the
    # pre-existing long bucket can temporarily exceed 1.5% of new equity. The
    # allocator must not resize history or crash; it must allocate zero new long risk.
    rows = [
        _trade(f"L{i}/USDT", "2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z", side=1, r=1.0)
        for i in range(6)
    ]
    rows.append(
        _trade("SHORT/USDT", "2026-01-01T00:00:00Z", "2026-01-01T03:59:00Z", side=-1, r=-1.24)
    )
    rows.append(
        _trade("NEWLONG/USDT", "2026-01-01T04:00:00Z", "2026-01-02T04:00:00Z", side=1, r=1.0)
    )
    out = apply_portfolio_allocator_v25(pd.DataFrame(rows), "4h")
    row = out[out["symbol"] == "NEWLONG/USDT"].iloc[0]

    assert float(row["open_directional_risk_fraction_before_v25"]) > PortfolioRiskBudgetV25().directional_open_risk_cap
    assert bool(row["executed_v25"]) is False
    assert row["reject_reason_v25"] == "PORTFOLIO_RISK_BUDGET_EXHAUSTED"
