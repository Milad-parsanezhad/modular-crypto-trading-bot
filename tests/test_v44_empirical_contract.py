from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_v44_fold_from_prepared.py"
spec = importlib.util.spec_from_file_location("v44_fold_contract", SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load v0.44 fold runner")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def _events() -> pd.DataFrame:
    signal = pd.date_range("2026-01-01", periods=120, freq="4h", tz="UTC")
    x = pd.DataFrame({
        "signal_time": signal,
        "exit_time": signal + pd.Timedelta(hours=8),
        "v44_sample_s0": True,
        "v44_sample_s1": [(i % 2) == 0 for i in range(120)],
        "v44_sample_s2": [(i % 3) == 0 for i in range(120)],
    })
    return x


def test_variant_masks_cannot_move_common_fold_boundaries():
    events = _events()
    row = pd.Series({
        "cal_start": pd.Timestamp("2026-01-10T00:00:00Z"),
        "pretest_cut": pd.Timestamp("2026-01-13T00:00:00Z"),
        "test_start": pd.Timestamp("2026-01-18T00:00:00Z"),
        "test_end": pd.Timestamp("2026-01-20T20:00:00Z"),
    })

    base_fit, base_cal, base_test = mod._window_slice(events, row)
    for col in ("v44_sample_s1", "v44_sample_s2"):
        filtered = events.loc[events[col]].copy()
        fit, cal, test = mod._window_slice(filtered, row)
        assert set(fit["signal_time"]).issubset(set(base_fit["signal_time"]))
        assert set(cal["signal_time"]).issubset(set(base_cal["signal_time"]))
        assert set(test["signal_time"]).issubset(set(base_test["signal_time"]))
        assert (fit["signal_time"] < row["cal_start"]).all()
        assert ((cal["signal_time"] >= row["cal_start"]) & (cal["signal_time"] < row["pretest_cut"])).all()
        assert ((test["signal_time"] >= row["test_start"]) & (test["signal_time"] <= row["test_end"])).all()


def test_fit_and_calibration_require_settled_labels_before_boundaries():
    events = _events()
    # Force two rows to settle after their allowed boundary; they must be excluded.
    events.loc[10, "exit_time"] = pd.Timestamp("2026-01-11T00:00:00Z")
    events.loc[70, "exit_time"] = pd.Timestamp("2026-01-19T00:00:00Z")
    row = pd.Series({
        "cal_start": pd.Timestamp("2026-01-10T00:00:00Z"),
        "pretest_cut": pd.Timestamp("2026-01-13T00:00:00Z"),
        "test_start": pd.Timestamp("2026-01-18T00:00:00Z"),
        "test_end": pd.Timestamp("2026-01-20T20:00:00Z"),
    })
    fit, cal, _ = mod._window_slice(events, row)
    assert (fit["exit_time"] < row["cal_start"]).all()
    assert (cal["exit_time"] < row["test_start"]).all()
