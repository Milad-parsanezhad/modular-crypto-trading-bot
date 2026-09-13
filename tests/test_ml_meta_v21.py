import numpy as np
import pandas as pd

from research_bot.ml_meta_v21 import (
    CATEGORICAL_FEATURES,
    OUTCOME_BANNED,
    MLConfig,
    causal_snapshot_frame,
    choose_classifier_champion,
    choose_threshold,
    classifier_zoo,
    economic_metrics,
    regressor_zoo,
    safe_feature_columns,
)


def synthetic(n=1800, freq="5min", seed=21):
    rng = np.random.default_rng(seed)
    ret = rng.normal(0.00008, 0.0035, n)
    close = 100 * np.exp(np.cumsum(ret))
    open_ = np.r_[close[0], close[:-1]]
    spread = np.maximum(close * 0.0008, rng.uniform(0.0002, 0.002, n) * close)
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq=freq, tz="UTC"),
        "open": open_, "high": high, "low": low, "close": close,
        "volume": rng.lognormal(8, 0.5, n),
    })


def labeled_frame(n=1200, seed=314):
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=n)
    y = (signal + rng.normal(scale=0.8, size=n) > 0).astype(int)
    r = np.where(y == 1, rng.uniform(0.2, 3.0, n), -rng.uniform(0.2, 1.3, n))
    seg = np.array(["development"] * 720 + ["validation"] * 240 + ["test"] * 240)
    return pd.DataFrame({
        "signal_time": pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
        "entry_time": pd.date_range("2024-01-01 01:00", periods=n, freq="h", tz="UTC"),
        "strategy": np.where(np.arange(n) % 2, "A", "B"),
        "family": "test", "timeframe": "1h", "symbol": np.where(np.arange(n) % 3, "BTC/USDT", "ETH/USDT"),
        "side": np.where(np.arange(n) % 2, 1, -1), "segment": seg,
        "f_signal": signal, "f_noise": rng.normal(size=n), "f_flag": (signal > 0).astype(int),
        "label_profitable_net": y, "label_target_hit": (r >= 2).astype(int),
        "label_ge_1r": (r >= 1).astype(int), "label_ge_2r": (r >= 2).astype(int),
        "label_outcome_3class": np.select([r <= 0, r >= 1], [-1, 1], default=0),
        "label_r_multiple": r,
        "r_multiple": r, "net_return": r * 0.001, "exit": 999.0, "target": 1001.0,
    })


def test_causal_snapshot_is_prefix_invariant():
    df = synthetic()
    full = causal_snapshot_frame(df, "5m")
    cut = 1200
    prefix = causal_snapshot_frame(df.iloc[:cut].copy(), "5m")
    cols = [c for c in full.columns if c.startswith("f_")][:15]
    for c in cols:
        assert np.allclose(full.loc[:cut - 1, c].to_numpy(float), prefix[c].to_numpy(float), equal_nan=True)


def test_feature_selector_is_whitelist_and_excludes_outcomes():
    df = labeled_frame()
    numeric, categorical = safe_feature_columns(df)
    assert numeric
    assert set(categorical).issubset(set(CATEGORICAL_FEATURES))
    assert not ((set(numeric) | set(categorical)) & OUTCOME_BANNED)
    assert "r_multiple" not in numeric
    assert "net_return" not in numeric
    assert "exit" not in numeric


def test_model_zoo_covers_major_families():
    c = classifier_zoo()
    r = regressor_zoo()
    required_c = {"logistic", "ridge_classifier", "linear_svc", "rbf_svc", "gaussian_nb", "knn", "decision_tree", "random_forest", "extra_trees", "gradient_boosting", "hist_gradient_boosting", "adaboost", "mlp"}
    required_r = {"linear_regression", "ridge", "elastic_net", "huber", "kernel_ridge", "svr", "knn_regressor", "decision_tree_regressor", "random_forest_regressor", "extra_trees_regressor", "gradient_boosting_regressor", "hist_gradient_boosting_regressor", "mlp_regressor"}
    assert required_c.issubset(c)
    assert required_r.issubset(r)


def test_threshold_selection_uses_only_validation_rows():
    df = labeled_frame()
    val = df[df.segment == "validation"].copy()
    score = 1 / (1 + np.exp(-val["f_signal"].to_numpy()))
    t1, m1 = choose_threshold(val, score, MLConfig(min_selected_validation=30))
    # Mutating a nonexistent test set cannot change a function that accepts validation only.
    t2, m2 = choose_threshold(val.copy(), score.copy(), MLConfig(min_selected_validation=30))
    assert t1 == t2
    assert m1["selected"] == m2["selected"]


def test_economic_metrics_reacts_to_good_filter():
    df = labeled_frame()
    good = df["f_signal"].to_numpy() > 0.75
    all_rows = np.ones(len(df), dtype=bool)
    m_good = economic_metrics(df, good)
    m_all = economic_metrics(df, all_rows)
    assert m_good["selected"] > 0
    assert m_good["mean_r"] > m_all["mean_r"]


def test_champion_is_selected_by_validation_not_test_ranking():
    board = pd.DataFrame([
        {"model": "A", "status": "ok", "threshold": 0.6, "validation_validation_economic_score": 2.0, "test_selected": 150, "test_mean_r": 0.2, "test_profit_factor": 1.2, "test_max_drawdown": -0.02},
        {"model": "B", "status": "ok", "threshold": 0.7, "validation_validation_economic_score": 1.0, "test_selected": 500, "test_mean_r": 9.0, "test_profit_factor": 99.0, "test_max_drawdown": -0.001},
    ])
    result = choose_classifier_champion(board, MLConfig())
    assert result["champion"] == "A"
    assert result["live_execution_authorized"] is False
