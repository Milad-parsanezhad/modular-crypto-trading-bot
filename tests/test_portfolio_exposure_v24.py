from __future__ import annotations

import numpy as np
import pandas as pd

from research_bot.portfolio_exposure_v24 import exposure_path_v24, summarize_exposure_v24


def _row(symbol: str, entry: str, exit_: str, risk: float, equity: float = 1.0, segment: str = "validation") -> dict:
    return {
        "symbol": symbol,
        "entry_time": entry,
        "exit_time": exit_,
        "executed_v23": True,
        "risk_fraction_v23": risk,
        "equity_at_entry_v23": equity,
        "segment": segment,
    }


def test_same_bar_exit_remains_open_for_same_bar_entry_exposure() -> None:
    ledger = pd.DataFrame(
        [
            _row("BTC/USDT", "2026-01-01T00:00:00Z", "2026-01-01T04:00:00Z", 0.0025),
            _row("ETH/USDT", "2026-01-01T04:00:00Z", "2026-01-01T08:00:00Z", 0.0025),
        ]
    )
    path = exposure_path_v24(ledger)
    second = path.iloc[1]
    assert int(second["open_positions"]) == 2
    assert int(second["open_symbols"]) == 2
    assert np.isclose(float(second["open_risk_fraction_of_equity"]), 0.005)


def test_strictly_earlier_exit_is_removed_before_new_entry() -> None:
    ledger = pd.DataFrame(
        [
            _row("BTC/USDT", "2026-01-01T00:00:00Z", "2026-01-01T03:59:00Z", 0.0025),
            _row("ETH/USDT", "2026-01-01T04:00:00Z", "2026-01-01T08:00:00Z", 0.0025),
        ]
    )
    path = exposure_path_v24(ledger)
    second = path.iloc[1]
    assert int(second["open_positions"]) == 1
    assert np.isclose(float(second["open_risk_fraction_of_equity"]), 0.0025)


def test_risk_cash_is_fixed_from_each_positions_entry_equity() -> None:
    ledger = pd.DataFrame(
        [
            _row("BTC/USDT", "2026-01-01T00:00:00Z", "2026-01-01T08:00:00Z", 0.01, equity=1.0),
            _row("ETH/USDT", "2026-01-01T04:00:00Z", "2026-01-01T12:00:00Z", 0.01, equity=0.8),
        ]
    )
    path = exposure_path_v24(ledger)
    second = path.iloc[1]
    # Open risk cash = 0.01*1.0 + 0.01*0.8 = 0.018; current realized equity=0.8.
    assert np.isclose(float(second["open_risk_cash"]), 0.018)
    assert np.isclose(float(second["open_risk_fraction_of_equity"]), 0.0225)


def test_summary_reports_tail_concurrency_and_risk_threshold_counts() -> None:
    path = pd.DataFrame(
        {
            "segment": ["validation"] * 4,
            "timestamp": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"], utc=True),
            "realized_equity": [1.0] * 4,
            "open_positions": [1, 2, 4, 8],
            "open_symbols": [1, 2, 4, 8],
            "open_risk_cash": [0.005, 0.015, 0.025, 0.060],
            "open_risk_fraction_of_equity": [0.005, 0.015, 0.025, 0.060],
        }
    )
    summary = summarize_exposure_v24(path)
    assert summary["max_open_positions"] == 8
    assert summary["max_open_symbols"] == 8
    assert np.isclose(float(summary["max_open_risk_fraction"]), 0.06)
    assert summary["snapshots_open_risk_gt_1pct"] == 3
    assert summary["snapshots_open_risk_gt_2pct"] == 2
    assert summary["snapshots_open_risk_gt_5pct"] == 1
