from __future__ import annotations

import numpy as np
import pandas as pd

from research_bot.two_stage_hurdle_v40 import (
    HurdlePolicyV40,
    V40_SEEDS,
    fit_predict_seed_v40,
    median_seed_prediction_v40,
    preregistration_manifest_v40,
    split_probability_and_conformal_calibration,
)


def _events(n: int, start: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = pd.date_range(start, periods=n, freq="4h", tz="UTC")
    f1 = rng.normal(size=n)
    f2 = rng.normal(size=n)
    edge = 0.55 * f1 - 0.30 * f2 + rng.normal(scale=0.75, size=n)
    net_r = np.where(edge > 0.0, 0.55 + 0.35 * np.abs(edge), -(0.45 + 0.25 * np.abs(edge)))
    duration = rng.integers(1, 18, size=n)
    regime = np.array(["trend", "range", "transition"])[np.arange(n) % 3]
    return pd.DataFrame(
        {
            "series_id": "okx::BTC/USDT",
            "venue": "okx",
            "symbol": "BTC/USDT",
            "signal_time": t,
            "entry_time": t + pd.Timedelta(hours=4),
            "exit_time": t + pd.to_timedelta((duration + 1) * 4, unit="h"),
            "regime": regime,
            "net_r": net_r,
            "f1": f1.astype("float32"),
            "f2": f2.astype("float32"),
        }
    )


def test_v40_manifest_keeps_holdout_and_execution_disabled() -> None:
    m = preregistration_manifest_v40()
    assert m["reserved_holdout"] == "kraken"
    assert m["kraken_touched"] is False
    assert m["paper_execution"] is False
    assert m["live_execution"] is False
    assert m["stage1_target"] == "1(net_r > 0)"


def test_calibration_split_is_chronological_and_disjoint() -> None:
    cal = _events(100, "2025-03-01", 1)
    prob, conf = split_probability_and_conformal_calibration(cal)
    assert len(prob) == 50
    assert len(conf) == 50
    assert prob["signal_time"].max() < conf["signal_time"].min()
    assert set(prob.index).isdisjoint(set(conf.index)) is False  # indices reset by splitter copies
    assert not set(prob["signal_time"]).intersection(set(conf["signal_time"]))


def test_target_labels_cannot_change_v40_predictions() -> None:
    fit = _events(300, "2025-01-01", 2)
    prob_cal = _events(100, "2025-04-01", 3)
    conf_cal = _events(120, "2025-05-01", 4)
    target = _events(90, "2025-06-01", 5)
    cols = ["f1", "f2"]

    a, _ = fit_predict_seed_v40(
        "V40_LOGIT_HUBER_HURDLE", fit, prob_cal, conf_cal, target, cols, 314
    )
    changed = target.copy()
    changed["net_r"] = np.linspace(-999.0, 999.0, len(changed))
    b, _ = fit_predict_seed_v40(
        "V40_LOGIT_HUBER_HURDLE", fit, prob_cal, conf_cal, changed, cols, 314
    )

    for col in (
        "p_win_v40",
        "expected_r_v40",
        "lower_expected_r_v40",
        "upper_expected_r_v40",
        "predicted_duration_bars_v40",
    ):
        np.testing.assert_allclose(a[col].to_numpy(), b[col].to_numpy(), rtol=0, atol=1e-12)
    assert np.array_equal(a["selected_v40"].to_numpy(), b["selected_v40"].to_numpy())


def test_selected_rows_obey_frozen_hurdle_rule_and_three_seed_median() -> None:
    fit = _events(300, "2025-01-01", 12)
    prob_cal = _events(100, "2025-04-01", 13)
    conf_cal = _events(120, "2025-05-01", 14)
    target = _events(90, "2025-06-01", 15)
    outputs = []
    for seed in V40_SEEDS:
        out, diag = fit_predict_seed_v40(
            "V40_LOGIT_HUBER_HURDLE", fit, prob_cal, conf_cal, target, ["f1", "f2"], seed
        )
        assert np.isfinite(out[["p_win_v40", "expected_r_v40", "lower_expected_r_v40"]].to_numpy()).all()
        assert 0.0 <= diag["brier"] <= 1.0
        outputs.append(out)

    combined = median_seed_prediction_v40(outputs)
    p = HurdlePolicyV40()
    selected = combined["selected_v40"]
    assert ((combined.loc[selected, "p_win_v40"] >= p.stage1_probability_floor)).all()
    assert (combined.loc[selected, "expected_r_v40"] > 0.0).all()
    assert (combined.loc[selected, "lower_expected_r_v40"] > 0.0).all()
    assert combined["seed"].eq("MEDIAN_3").all()
