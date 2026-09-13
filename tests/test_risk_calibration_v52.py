import numpy as np

from research_bot.risk_calibration_v52 import ConformalRiskConfig, calibrate_one_sided_var


def test_fails_closed_with_too_little_history():
    result = calibrate_one_sided_var(
        realized_losses=[0.01] * 20,
        predicted_var=[0.012] * 20,
        current_predicted_var=0.015,
        config=ConformalRiskConfig(min_observations=40),
    )
    assert result.approved_for_research_use is False
    assert result.reason == "INSUFFICIENT_CALIBRATION_HISTORY"
    assert result.calibrated_var == 0.015


def test_positive_underprediction_scores_add_a_buffer():
    losses = [0.01] * 45 + [0.03] * 15
    forecasts = [0.012] * 60
    result = calibrate_one_sided_var(
        realized_losses=losses,
        predicted_var=forecasts,
        current_predicted_var=0.02,
        config=ConformalRiskConfig(alpha=0.95, decay=1.0, min_observations=40, min_effective_sample_size=20),
    )
    assert result.approved_for_research_use is True
    assert result.additive_buffer > 0
    assert result.calibrated_var >= 0.02


def test_regime_localization_can_fail_closed_on_low_ess():
    n = 60
    regimes = np.column_stack([np.linspace(-10, 10, n), np.zeros(n)])
    result = calibrate_one_sided_var(
        realized_losses=[0.01] * n,
        predicted_var=[0.012] * n,
        current_predicted_var=0.015,
        historical_regimes=regimes.tolist(),
        current_regime=[20.0, 0.0],
        config=ConformalRiskConfig(
            alpha=0.95,
            decay=0.97,
            regime_bandwidth=0.1,
            min_observations=40,
            min_effective_sample_size=15,
        ),
    )
    assert result.approved_for_research_use is False
    assert result.reason == "INSUFFICIENT_EFFECTIVE_SAMPLE_SIZE"


def test_invalid_current_var_is_rejected():
    try:
        calibrate_one_sided_var(
            realized_losses=[0.01] * 50,
            predicted_var=[0.012] * 50,
            current_predicted_var=float("nan"),
        )
    except ValueError as exc:
        assert "current_predicted_var" in str(exc)
    else:
        raise AssertionError("non-finite current VaR must fail closed")
