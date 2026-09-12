from __future__ import annotations

import numpy as np
import pandas as pd

from research_bot.label_representation_diagnostic_v45 import (
    assign_common_test_folds_v45,
    brier_decomposition_v45,
    cause_age_hazard_v45,
    feature_information_v45,
)


def test_brier_decomposition_perfect_forecast_reconstructs_zero():
    y = np.array([0, 0, 1, 1], dtype=float)
    p = y.copy()
    metrics, bins = brier_decomposition_v45(y, p)
    assert metrics["n"] == 4
    assert abs(float(metrics["brier"])) < 1e-12
    assert abs(float(metrics["reliability"])) < 1e-12
    assert abs(float(metrics["resolution"]) - float(metrics["uncertainty"])) < 1e-12
    assert abs(float(metrics["reconstruction_residual"])) < 1e-12
    assert int(bins["n"].sum()) == 4


def test_brier_decomposition_constant_prevalence_has_zero_resolution():
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=float)
    p = np.full(len(y), 0.5)
    metrics, _ = brier_decomposition_v45(y, p)
    assert abs(float(metrics["reliability"])) < 1e-12
    assert abs(float(metrics["resolution"])) < 1e-12
    assert abs(float(metrics["brier"]) - 0.25) < 1e-12
    assert abs(float(metrics["uncertainty"]) - 0.25) < 1e-12


def test_common_test_fold_assignment_never_rebuilds_boundaries():
    timestamps = pd.date_range("2026-01-01", periods=60, freq="4h", tz="UTC")
    events = pd.DataFrame({
        "signal_time": timestamps,
        "entry_time": timestamps,
        "exit_time": timestamps + pd.Timedelta(hours=4),
        "v44_sample_s0": True,
        "symbol": "BTC/USDT",
        "venue": "okx",
        "outcome": "TIME",
        "net_r": 0.0,
    })
    starts = [timestamps[0], timestamps[10], timestamps[20], timestamps[30], timestamps[40]]
    ends = [timestamps[4], timestamps[14], timestamps[24], timestamps[34], timestamps[44]]
    folds = pd.DataFrame({"fold": [1, 2, 3, 4, 5], "test_start": starts, "test_end": ends})
    out = assign_common_test_folds_v45(events, folds)
    assert len(out) == 25
    assert out["fold"].value_counts().sort_index().tolist() == [5, 5, 5, 5, 5]
    assert set(out["signal_time"]).issubset(set(timestamps))


def test_cause_age_hazard_conserves_terminal_events_and_risk_set_falls():
    base = pd.Timestamp("2026-01-01T00:00:00Z")
    durations = np.array([1, 2, 3, 5, 5, 10, 30])
    outcomes = ["TARGET", "STOP", "TIME", "TARGET", "STOP", "TIME", "TIME"]
    entry = [base] * len(durations)
    # duration_bars = round(delta/4h)+1, therefore delta=(duration-1)*4h.
    exit_ = [base + pd.Timedelta(hours=4 * (int(d) - 1)) for d in durations]
    frame = pd.DataFrame({"entry_time": entry, "exit_time": exit_, "outcome": outcomes})
    table = cause_age_hazard_v45(frame)
    risk = table["at_risk_n"].to_numpy(dtype=int)
    assert np.all(np.diff(risk) <= 0)
    terminal = table[["target_terminal_n", "stop_terminal_n", "time_terminal_n"]].sum(axis=1).sum()
    assert int(terminal) == len(frame)
    assert int(table.iloc[0]["at_risk_n"]) == len(frame)


def test_feature_information_is_descriptive_and_support_gated():
    n = 240
    target = np.array(([0] * 120) + ([1] * 120), dtype=int)
    outcome = np.where(target == 1, "TARGET", "STOP")
    frame = pd.DataFrame({
        "fold": np.ones(n, dtype=int),
        "outcome": outcome,
        "f_signal": target.astype(float),
    })
    result = feature_information_v45(frame, ["f_signal"])
    overall = result[(result["scope"] == "overall") & (result["feature"] == "f_signal")].iloc[0]
    assert bool(overall["target_supported"])
    assert bool(overall["stop_supported"])
    assert abs(float(overall["target_auc"]) - 1.0) < 1e-12
    assert abs(float(overall["stop_auc"]) - 0.0) < 1e-12
    assert abs(float(overall["target_auc_abs_edge"]) - 0.5) < 1e-12
