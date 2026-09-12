from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FAST = ROOT / "scripts" / "run_v42_breadth_cluster_characterization_fast.py"
spec = importlib.util.spec_from_file_location("v42_fast_cache_test", FAST)
assert spec is not None and spec.loader is not None
fast = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fast)


def _events() -> pd.DataFrame:
    start = pd.Timestamp("2025-01-01", tz="UTC")
    return pd.DataFrame(
        {
            "x": [0.1, -0.2, 0.3],
            "event_family_v41": ["ICT_MSS", "BROOKS_H2L2", "SMC_FVG_STRUCTURE"],
            "regime": ["trend", "range", "transition"],
            "side": [1, -1, 1],
            "entry_time": [start, start + pd.Timedelta(hours=4), start + pd.Timedelta(hours=8)],
            "exit_time": [
                start + pd.Timedelta(hours=8),
                start + pd.Timedelta(hours=16),
                start + pd.Timedelta(hours=24),
            ],
            "outcome": ["TARGET", "STOP", "TIME"],
            "net_r": [3.0, -1.0, 0.15],
        }
    )


def test_cached_event_design_is_numerically_identical() -> None:
    events = _events()
    original = fast._ORIGINAL_EVENT(events, ("x",))
    cached1 = fast._cached_event(events, ("x",))
    cached2 = fast._cached_event(events, ("x",))
    assert np.array_equal(original, cached1)
    assert cached1 is cached2


def test_cached_hazard_table_is_numerically_identical_and_reused() -> None:
    events = _events()
    policy = fast.cr.CompetingRiskPolicyV41(max_hold_bars=5)
    original = fast._ORIGINAL_HAZARD(events, ("x",), policy)
    cached1 = fast._cached_hazard(events, ("x",), policy)
    cached2 = fast._cached_hazard(events, ("x",), policy)
    for a, b in zip(original, cached1):
        assert np.array_equal(a, b)
    for a, b in zip(cached1, cached2):
        assert a is b
