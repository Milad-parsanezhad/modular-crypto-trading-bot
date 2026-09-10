import numpy as np
import pandas as pd
import pytest

from research_bot.ml_rebuild_v23 import (
    MLRebuildConfig,
    assert_unique_columns,
    attach_development_only_unsupervised,
    audit_labeled_dataset,
    choose_classifier_champion,
    portfolio_economic_metrics,
    safe_feature_columns,
    train_classifier_tournament,
)


def make_dataset(n_dev=180, n_val=90, n_test=90):
    rng = np.random.default_rng(314)
    parts = []
    start = pd.Timestamp("2024-01-01", tz="UTC")
    offset = 0
    for seg, n in (("development", n_dev), ("validation", n_val), ("test", n_test)):
        t = pd.date_range(start + pd.Timedelta(hours=4 * offset), periods=n, freq="4h")
        x1 = rng.normal(size=n)
        x2 = rng.normal(size=n)
        latent = 0.8 * x1 - 0.3 * x2 + rng.normal(scale=0.7, size=n)
        y = (latent > 0).astype(int)
        r = np.where(y == 1, 1.2 + 0.2 * rng.normal(size=n), -1.0 + 0.2 * rng.normal(size=n))
        parts.append(pd.DataFrame({
            "signal_time": t,
            "segment": seg,
            "strategy": "SYNTH",
            "family": "test",
            "timeframe": "4h",
            "symbol": "BTC/USDT",
            "side": 1,
            "f_x1": x1,
            "f_x2": x2,
            "label_profitable_net": y,
            "label_target_hit": y,
            "label_ge_1r": (r >= 1).astype(int),
            "label_ge_2r": (r >= 2).astype(int),
            "label_outcome_3class": np.where(r <= 0, -1, np.where(r >= 1, 1, 0)),
            "label_r_multiple": r,
        }))
        offset += n
    return pd.concat(parts, ignore_index=True)


def test_duplicate_columns_are_fatal():
    x = pd.DataFrame(np.ones((4, 2)), columns=["f_x", "f_x"])
    with pytest.raises(ValueError, match="duplicate columns"):
        assert_unique_columns(x)


def test_future_named_feature_is_rejected():
    x = make_dataset()
    x["f_future_return"] = 0.0
    with pytest.raises(ValueError, match="suspicious"):
        safe_feature_columns(x)


def test_chronological_overlap_is_rejected():
    x = make_dataset()
    val_idx = x.index[x["segment"].eq("validation")][0]
    x.loc[val_idx, "signal_time"] = pd.Timestamp("2024-01-01", tz="UTC")
    with pytest.raises(ValueError, match="chronological split overlap"):
        audit_labeled_dataset(x)


def test_development_only_unsupervised_fit_ignores_test_distribution():
    a = make_dataset()
    b = a.copy()
    mask = b["segment"].isin(["validation", "test"])
    b.loc[mask, "f_x1"] += 10000.0
    b.loc[mask, "f_x2"] -= 10000.0

    _, bank_a = attach_development_only_unsupervised(a, seed=314)
    _, bank_b = attach_development_only_unsupervised(b, seed=314)

    np.testing.assert_allclose(bank_a.scaler.center_, bank_b.scaler.center_)
    np.testing.assert_allclose(bank_a.kmeans.cluster_centers_, bank_b.kmeans.cluster_centers_)


def test_concurrent_risk_is_capped_per_timestamp():
    t = pd.Timestamp("2026-01-01", tz="UTC")
    rows = pd.DataFrame({
        "signal_time": [t] * 10,
        "label_r_multiple": [1.0] * 10,
    })
    cfg = MLRebuildConfig(base_risk_per_trade=0.0025, max_risk_per_timestamp=0.01)
    m = portfolio_economic_metrics(rows, np.ones(10, dtype=bool), cfg)
    assert m["selected"] == 10
    assert m["timestamps"] == 1
    assert m["total_return"] == pytest.approx(0.01, abs=1e-12)


def test_post_cost_r_is_not_charged_a_second_time():
    rows = pd.DataFrame({
        "signal_time": [pd.Timestamp("2026-01-01", tz="UTC")],
        "label_r_multiple": [1.0],
    })
    cfg = MLRebuildConfig(base_risk_per_trade=0.0025, max_risk_per_timestamp=0.01)
    m = portfolio_economic_metrics(rows, np.array([True]), cfg)
    assert m["total_return"] == pytest.approx(0.0025, abs=1e-12)


def test_validation_threshold_is_independent_of_test_mutation(tmp_path):
    a = make_dataset()
    b = a.copy()
    test_mask = b["segment"].eq("test")
    b.loc[test_mask, "f_x1"] *= -100.0
    b.loc[test_mask, "f_x2"] += 500.0
    b.loc[test_mask, "label_profitable_net"] = 1 - b.loc[test_mask, "label_profitable_net"]
    b.loc[test_mask, "label_r_multiple"] *= -1.0

    cfg = MLRebuildConfig(
        min_selected_validation=20,
        min_selected_test=10,
        bootstrap_resamples=50,
        bootstrap_block=4,
        require_nonnegative_bootstrap_lower=False,
    )
    board_a, _, _, _ = train_classifier_tournament(
        a, tmp_path / "a", cfg, include_unsupervised=False, model_names=["logistic"]
    )
    board_b, _, _, _ = train_classifier_tournament(
        b, tmp_path / "b", cfg, include_unsupervised=False, model_names=["logistic"]
    )
    aa = board_a.iloc[0]
    bb = board_b.iloc[0]
    assert aa["threshold"] == pytest.approx(bb["threshold"])
    assert aa["validation_validation_objective"] == pytest.approx(bb["validation_validation_objective"])


def test_small_supervised_smoke_tournament_runs(tmp_path):
    x = make_dataset()
    cfg = MLRebuildConfig(
        min_selected_validation=20,
        min_selected_test=10,
        bootstrap_resamples=40,
        bootstrap_block=4,
        require_nonnegative_bootstrap_lower=False,
    )
    board, val, test, meta = train_classifier_tournament(
        x,
        tmp_path,
        cfg,
        include_unsupervised=True,
        model_names=["dummy_prior", "logistic", "random_forest"],
    )
    assert set(board["model"]) == {"dummy_prior", "logistic", "random_forest"}
    assert (board["status"] == "ok").all()
    assert not val.empty and not test.empty
    assert meta["unsupervised_fitted_on"] == "development only"


def test_promotion_gate_fails_closed_on_negative_bootstrap_bound():
    board = pd.DataFrame([{
        "model": "logistic",
        "status": "ok",
        "threshold": 0.7,
        "validation_validation_objective": 10.0,
        "test_selected": 300,
        "test_mean_r": 0.4,
        "test_profit_factor_r": 1.8,
        "test_max_drawdown": -0.02,
        "test_bootstrap_ci_low": -0.00001,
    }])
    decision = choose_classifier_champion(board, MLRebuildConfig())
    assert decision["decision"] == "NO_MODEL_PROMOTED"
    assert decision["live_execution_authorized"] is False


def test_promotion_gate_can_only_yield_forward_paper_candidate():
    board = pd.DataFrame([{
        "model": "logistic",
        "status": "ok",
        "threshold": 0.7,
        "validation_validation_objective": 10.0,
        "test_selected": 300,
        "test_mean_r": 0.4,
        "test_profit_factor_r": 1.8,
        "test_max_drawdown": -0.02,
        "test_bootstrap_ci_low": 0.00001,
    }])
    decision = choose_classifier_champion(board, MLRebuildConfig())
    assert decision["decision"] == "FORWARD_PAPER_CANDIDATE"
    assert decision["paper_replacement_authorized"] is False
    assert decision["live_execution_authorized"] is False
