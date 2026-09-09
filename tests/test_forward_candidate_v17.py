from __future__ import annotations

import numpy as np
import pandas as pd

from research_bot.forward_candidate_v17 import s6_candidate_signal


def bars(n: int = 260) -> pd.DataFrame:
    close = np.linspace(100.0, 150.0, n)
    close[-1] = 155.0
    ts = pd.date_range("2025-01-01", periods=n, freq="4h", tz="UTC")
    return pd.DataFrame({
        "timestamp": ts,
        "open": close - 0.1,
        "high": close + 0.5,
        "low": close - 0.5,
        "close": close,
        "volume": np.full(n, 1_000.0),
    })


def test_s6_forward_candidate_is_paper_only_and_risk_capped():
    signal = s6_candidate_signal(bars(), currently_long=False)
    assert signal.action == "BUY_CANDIDATE"
    assert signal.paper_only is True
    assert signal.live_execution is False
    assert 0.0 < signal.risk_weight <= 0.35
    assert signal.limits["max_drawdown"] == 0.05


def test_s6_forward_candidate_ignores_future_rows():
    frame = bars()
    first = s6_candidate_signal(frame.iloc[:240], currently_long=False)
    changed = frame.copy()
    changed.loc[240:, "close"] *= 4
    changed.loc[240:, "high"] = changed.loc[240:, "close"] + 1
    changed.loc[240:, "low"] = changed.loc[240:, "close"] - 1
    second = s6_candidate_signal(changed.iloc[:240], currently_long=False)
    assert first == second
