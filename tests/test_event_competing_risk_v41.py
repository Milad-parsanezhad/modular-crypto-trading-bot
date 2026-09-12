from __future__ import annotations

import numpy as np
import pandas as pd

from research_bot.event_competing_risk_v41 import (
    CompetingRiskPolicyV41,
    V41_SEEDS,
    assign_event_family_v41,
    build_hazard_training_table_v41,
    calibrate_expected_r_bounds_v41,
    fit_competing_risk_bundle_v41,
    median_seed_prediction_v41,
    predict_competing_risks_v41,
    preregistration_manifest_v41,
)


def _events(n: int, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = pd.date_range("2025-01-01", periods=n, freq="4h", tz="UTC")
    f1 = rng.normal(size=n).astype("float32")
    f2 = rng.normal(size=n).astype("float32")
    latent = 0.7 * f1 - 0.3 * f2 + rng.normal(scale=0.9, size=n)
    outcome = np.where(latent > 0.55, "TARGET", np.where(latent < -0.45, "STOP", "TIME"))
    duration = rng.integers(1, 12, size=n)
    net_r = np.where(outcome == "TARGET", 2.92, np.where(outcome == "STOP", -1.08, 0.25 * latent - 0.08))
    family = np.array(["ICT_MSS", "BROOKS_H2L2", "ICHIMOKU_BREAKOUT"])[np.arange(n) % 3]
    regime = np.array(["trend", "range", "transition"])[np.arange(n) % 3]
    side = np.where(np.arange(n) % 2 == 0, 1, -1)
    return pd.DataFrame(
        {
            "series_id": "okx::BTC/USDT",
            "venue": "okx",
            "symbol": "BTC/USDT",
            "signal_time": t,
            "entry_time": t + pd.Timedelta(hours=4),
            "exit_time": t + pd.to_timedelta(duration * 4, unit="h"),
            "outcome": outcome,
            "net_r": net_r,
            "base_cost_r": np.full(n, 0.08),
            "side": side,
            "regime": regime,
            "event_family_v41": family,
            "f1": f1,
            "f2": f2,
        }
    )


def test_v41_manifest_is_fail_closed() -> None:
    m = preregistration_manifest_v41()
    assert m["reserved_holdout"] == "kraken"
    assert m["kraken_touched"] is False
    assert m["paper_execution"] is False
    assert m["live_execution"] is False
    assert m["causes"] == ["TARGET", "STOP"]


def test_event_family_priority_is_deterministic() -> None:
    cols = {
        "mother_event_v39": [1],
        "ict_recent_sweep_down_v39": [1], "ict_recent_sweep_up_v39": [0],
        "canonical_bull_displacement": [1], "canonical_bear_displacement": [0],
        "brooks_failed_breakdown_v39": [1], "brooks_failed_breakout_v39": [0],
        "smc_bull_ob_retest_v39": [1], "smc_bear_ob_retest_v39": [0],
        "ichimoku_pullback_long_v39": [1], "ichimoku_pullback_short_v39": [0],
        "ichimoku_breakout_up_v39": [1], "ichimoku_breakout_down_v39": [0],
        "brooks_h2_v39": [1], "brooks_l2_v39": [0],
        "ict_mss_up_v39": [1], "ict_mss_down_v39": [0],
    }
    x = pd.DataFrame(cols)
    assert assign_event_family_v41(x).iloc[0] == "ICT_MSS"


def test_discrete_time_table_has_single_terminal_cause() -> None:
    x = _events(20, 2)
    X, yt, ys, owner = build_hazard_training_table_v41(x, ["f1", "f2"])
    assert len(X) == len(yt) == len(ys) == len(owner)
    assert not np.any((yt == 1) & (ys == 1))
    for i, outcome in enumerate(x["outcome"]):
        mask = owner == i
        assert int(yt[mask].sum()) == int(outcome == "TARGET")
        assert int(ys[mask].sum()) == int(outcome == "STOP")


def test_target_labels_do_not_change_competing_risk_prediction() -> None:
    fit = _events(500, 3)
    target = _events(80, 4)
    bundle = fit_competing_risk_bundle_v41(
        "V41_LOGIT_CAUSE_SPECIFIC", fit, ["f1", "f2"], 314
    )
    a = predict_competing_risks_v41(bundle, target)
    changed = target.copy()
    changed["net_r"] = np.linspace(-999, 999, len(changed))
    changed["outcome"] = np.where(np.arange(len(changed)) % 2, "STOP", "TARGET")
    b = predict_competing_risks_v41(bundle, changed)
    for col in ("p_target_v41", "p_stop_v41", "p_timeout_v41", "expected_r_v41"):
        np.testing.assert_allclose(a[col], b[col], rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        a[["p_target_v41", "p_stop_v41", "p_timeout_v41"]].sum(axis=1),
        np.ones(len(a)),
        rtol=0,
        atol=1e-8,
    )


def test_conformal_selection_and_three_seed_median() -> None:
    fit = _events(500, 5)
    cal = _events(180, 6)
    test = _events(100, 7)
    outputs = []
    for seed in V41_SEEDS:
        bundle = fit_competing_risk_bundle_v41(
            "V41_LOGIT_CAUSE_SPECIFIC", fit, ["f1", "f2"], seed
        )
        cal_pred = predict_competing_risks_v41(bundle, cal)
        test_pred = predict_competing_risks_v41(bundle, test)
        outputs.append(calibrate_expected_r_bounds_v41(cal, cal_pred, test_pred))
    combined = median_seed_prediction_v41(outputs)
    assert combined["seed"].eq("MEDIAN_3").all()
    selected = combined["selected_v41"]
    assert combined.loc[selected, "lower_expected_r_v41"].gt(0).all()
    assert combined.loc[selected, "p_target_v41"].gt(combined.loc[selected, "p_stop_v41"]).all()
