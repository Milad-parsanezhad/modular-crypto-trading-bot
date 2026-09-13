from __future__ import annotations

import numpy as np
import pandas as pd

from research_bot.causal_evaluation_v23 import summarize_trades_v23
from research_bot.risk_causality_v23 import apply_causal_risk_overlay_v23, causality_regression_probe


def _trade(symbol: str, entry: str, exit_: str, r: float, segment: str = "validation") -> dict:
    return {
        "symbol": symbol,
        "entry_time": entry,
        "exit_time": exit_,
        "r_multiple": r,
        "risk_scale_volatility": 1.0,
        "segment": segment,
    }


def test_regression_probe_blocks_unresolved_future_outcome_leakage() -> None:
    result = causality_regression_probe()
    assert result["passed"] is True
    assert result["unresolved_outcome_invariant"] is True
    assert result["resolved_outcome_can_change_future_risk"] is True


def test_exit_on_same_bar_is_not_known_at_that_bars_entry() -> None:
    loss = pd.DataFrame(
        [
            _trade("BTC/USDT", "2026-01-01T00:00:00Z", "2026-01-01T04:00:00Z", -10.0),
            _trade("ETH/USDT", "2026-01-01T04:00:00Z", "2026-01-01T08:00:00Z", 1.0),
        ]
    )
    win = loss.copy()
    win.loc[0, "r_multiple"] = 10.0

    a = apply_causal_risk_overlay_v23(loss, "4h")
    b = apply_causal_risk_overlay_v23(win, "4h")
    eth_a = a.loc[a["symbol"] == "ETH/USDT"].iloc[0]
    eth_b = b.loc[b["symbol"] == "ETH/USDT"].iloc[0]

    assert np.isclose(eth_a["risk_fraction_v23"], eth_b["risk_fraction_v23"])
    assert np.isclose(eth_a["drawdown_before_v23"], eth_b["drawdown_before_v23"])
    assert int(eth_a["open_positions_before_v23"]) == 1


def test_realized_hard_drawdown_kills_only_subsequent_entries() -> None:
    ledger = pd.DataFrame(
        [
            _trade("BTC/USDT", "2026-01-01T00:00:00Z", "2026-01-01T04:00:00Z", -24.0),
            _trade("ETH/USDT", "2026-01-01T02:00:00Z", "2026-01-01T06:00:00Z", 1.0),
            _trade("SOL/USDT", "2026-01-01T08:00:00Z", "2026-01-01T12:00:00Z", 1.0),
        ]
    )
    out = apply_causal_risk_overlay_v23(ledger, "4h")

    btc = out.loc[out["symbol"] == "BTC/USDT"].iloc[0]
    eth = out.loc[out["symbol"] == "ETH/USDT"].iloc[0]
    sol = out.loc[out["symbol"] == "SOL/USDT"].iloc[0]

    # ETH entered before BTC's loss was realized, so rejecting ETH would be lookahead.
    assert bool(btc["executed_v23"]) is True
    assert bool(eth["executed_v23"]) is True
    # By SOL's entry, the >5% realized drawdown is known and the hard gate must fire.
    assert bool(sol["executed_v23"]) is False
    assert sol["reject_reason_v23"] == "HARD_DRAWDOWN_KILL"


def test_same_timestamp_settlements_are_batched() -> None:
    ledger = pd.DataFrame(
        [
            _trade("BTC/USDT", "2026-01-01T00:00:00Z", "2026-01-01T08:00:00Z", 2.0),
            _trade("ETH/USDT", "2026-01-01T04:00:00Z", "2026-01-01T08:00:00Z", -1.0),
            _trade("SOL/USDT", "2026-01-01T12:00:00Z", "2026-01-01T16:00:00Z", 1.0),
        ]
    )
    out = apply_causal_risk_overlay_v23(ledger, "4h")
    closed = out[out["symbol"].isin(["BTC/USDT", "ETH/USDT"])]

    assert set(closed["settlement_batch_size_v23"].astype(int)) == {2}
    assert np.isclose(closed["settlement_equity_v23"].iloc[0], closed["settlement_equity_v23"].iloc[1])
    assert np.isclose(closed["settlement_batch_return_v23"].iloc[0], closed["settlement_batch_return_v23"].iloc[1])


def test_overlapping_positions_settle_fixed_entry_equity_cash_pnl() -> None:
    # Both positions are sized from equity=1 before either outcome is known.
    # With +/-10% entry-normalized P&L, correct cash settlement is exactly flat.
    # Sequential percentage compounding would incorrectly produce 0.99.
    ledger = pd.DataFrame(
        [
            _trade("BTC/USDT", "2026-01-01T00:00:00Z", "2026-01-01T08:00:00Z", 40.0),
            _trade("ETH/USDT", "2026-01-01T04:00:00Z", "2026-01-01T08:00:00Z", -40.0),
        ]
    )
    out = apply_causal_risk_overlay_v23(ledger, "4h")

    assert np.allclose(out["equity_at_entry_v23"].to_numpy(dtype=float), [1.0, 1.0])
    assert np.allclose(np.sort(out["pnl_cash_v23"].to_numpy(dtype=float)), [-0.1, 0.1])
    assert np.allclose(out["settlement_batch_pnl_v23"].to_numpy(dtype=float), [0.0, 0.0])
    assert np.allclose(out["settlement_equity_v23"].to_numpy(dtype=float), [1.0, 1.0])
    assert np.allclose(out["settlement_batch_return_v23"].to_numpy(dtype=float), [0.0, 0.0])

    summary = summarize_trades_v23(out[out["executed_v23"] == True])  # noqa: E712
    assert np.isclose(float(summary["total_return"]), 0.0)
    assert np.isclose(float(summary["max_drawdown"]), 0.0)


def test_first_realized_loss_counts_against_initial_equity_peak() -> None:
    ledger = pd.DataFrame(
        [_trade("BTC/USDT", "2026-01-01T00:00:00Z", "2026-01-01T04:00:00Z", -4.0)]
    )
    out = apply_causal_risk_overlay_v23(ledger, "4h")
    summary = summarize_trades_v23(out[out["executed_v23"] == True])  # noqa: E712
    # base risk 0.25% * -4R = -1% from initial equity.
    assert np.isclose(float(summary["total_return"]), -0.01)
    assert np.isclose(float(summary["max_drawdown"]), -0.01)
