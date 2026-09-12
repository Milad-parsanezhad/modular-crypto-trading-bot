from __future__ import annotations

import numpy as np
import pandas as pd

from research_bot.temporal_regime_diagnostic_v48 import (
    bh_adjust,
    chronological_halves,
    fold_cluster_bootstrap_spearman,
    js_divergence,
    one_sided_sign_p,
    robust_scale_fit,
    standardized_wasserstein,
)


def test_robust_scale_is_fit_only_and_positive():
    x = pd.Series([0.0, 1.0, 2.0, 3.0, 4.0])
    s = robust_scale_fit(x)
    assert s is not None
    assert s > 0


def test_constant_feature_fails_closed():
    assert robust_scale_fit(pd.Series([1.0, 1.0, 1.0])) is None
    assert standardized_wasserstein(pd.Series([0.0, 1.0]), pd.Series([1.0, 2.0]), None) is None


def test_standardized_wasserstein_detects_shift():
    d = standardized_wasserstein(pd.Series([0.0, 0.0, 1.0]), pd.Series([1.0, 1.0, 2.0]), 1.0)
    assert d is not None
    assert d > 0


def test_chronological_halves_do_not_shuffle():
    df = pd.DataFrame({
        "signal_time": pd.date_range("2025-01-01", periods=6, freq="D", tz="UTC"),
        "x": np.arange(6),
    })
    early, late = chronological_halves(df)
    assert early["x"].tolist() == [0, 1, 2]
    assert late["x"].tolist() == [3, 4, 5]


def test_js_divergence_extremes():
    assert js_divergence(["a", "a"], ["a", "a"]) == 0.0
    assert js_divergence(["a", "a"], ["b", "b"]) > 0.99


def test_sign_test_and_bh_are_deterministic():
    k, n, p = one_sided_sign_p([1.0, 2.0, 3.0, -1.0])
    assert (k, n) == (3, 4)
    assert 0 <= p <= 1
    q = bh_adjust([0.01, 0.04, 0.03])
    assert np.all(np.isfinite(q))
    assert np.all(q >= np.array([0.01, 0.04, 0.03]) - 1e-12)


def test_fold_cluster_bootstrap_spearman_fixed_seed():
    df = pd.DataFrame({
        "fold": [1, 1, 2, 2, 3, 3],
        "x": [1, 2, 3, 4, 5, 6],
        "y": [1, 2, 3, 4, 5, 6],
    })
    a = fold_cluster_bootstrap_spearman(df, "x", "y", reps=100, seed=314159)
    b = fold_cluster_bootstrap_spearman(df, "x", "y", reps=100, seed=314159)
    assert a == b
    assert a["rho"] > 0.99
    assert a["ci90_low"] is not None and a["ci90_low"] > 0.99
