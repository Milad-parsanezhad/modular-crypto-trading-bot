from __future__ import annotations

import numpy as np
import pandas as pd

from research_bot.fast_scanner import FastScanConfig, scan_symbol, scan_universe


def _bars(seed: int, n: int = 320) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ret = rng.normal(0.0005, 0.006, n)
    close = 100 * np.cumprod(1 + ret)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=n, freq="4h", tz="UTC"),
            "open": np.r_[close[0], close[:-1]],
            "high": close * (1 + rng.uniform(0.001, 0.008, n)),
            "low": close * (1 - rng.uniform(0.001, 0.008, n)),
            "close": close,
            "volume": rng.uniform(100, 1000, n),
        }
    )


def test_short_history_is_rejected_before_feature_scoring():
    result = scan_symbol("BTC/USDT", _bars(1, 100), FastScanConfig(min_bars=240))
    assert result.status == "REJECT"
    assert "INSUFFICIENT_HISTORY" in result.rejection_reasons


def test_scanner_ranks_dynamic_symbols_without_confirmed_entry_claim():
    data = {
        "BTC/USDT": _bars(1),
        "ETH/USDT": _bars(2),
        "SOL/USDT": _bars(3),
        "XRP/USDT": _bars(4),
        "ADA/USDT": _bars(5),
    }
    result = scan_universe(data, FastScanConfig(candidate_quantile=0.8))
    assert len(result) == 5
    assert set(result["status"]).issubset({"SCANNED", "DEEP_ANALYSIS_CANDIDATE", "REJECT"})
    assert "CONFIRMED_ENTRY" not in set(result["status"])
    scanned = result[result["raw_score"].notna()]
    assert not scanned.empty
    assert scanned["score_percentile"].between(0, 1).all()
