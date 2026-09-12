from __future__ import annotations

import numpy as np
import pandas as pd

from research_bot.mother_strategy_v39 import (
    AccountStateV39,
    MOTHER_ENGINE_NAMES_V39,
    build_mother_features_v39,
    build_neural_panel_v39,
    build_trade_proposals_v39,
    mother_strategy_manifest_v39,
    psychology_allows_new_trade_v39,
)


def _synthetic(n: int = 900, seed: int = 314) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2025-01-01", periods=n, freq="4h", tz="UTC")
    drift = np.linspace(0.0, 32.0, n)
    cycle = 5.5 * np.sin(np.linspace(0.0, 22.0 * np.pi, n))
    noise = rng.normal(0.0, 0.45, n).cumsum() * 0.09
    close = 100.0 + drift + cycle + noise
    open_ = np.r_[close[0], close[:-1]] + rng.normal(0.0, 0.22, n)
    spread = np.abs(rng.normal(0.9, 0.18, n)) + 0.10
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = 1000.0 + 200.0 * np.abs(np.sin(np.linspace(0, 13 * np.pi, n))) + rng.uniform(0, 90, n)
    return pd.DataFrame({
        "timestamp": ts,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


def test_manifest_keeps_engines_independent_and_execution_off() -> None:
    m = mother_strategy_manifest_v39()
    assert tuple(m["engines"]) == MOTHER_ENGINE_NAMES_V39
    assert m["hard_k_of_n_gate"] is False
    assert m["kraken_holdout"] == "SEALED"
    assert m["paper_execution"] is False
    assert m["live_execution"] is False
    assert m["psychology"]["martingale_allowed"] is False
    assert m["psychology"]["averaging_down_allowed"] is False


def test_mother_features_are_causal_under_future_perturbation() -> None:
    base = _synthetic()
    cutoff = 620
    changed = base.copy()
    future = changed.index > cutoff
    changed.loc[future, "close"] *= 1.17
    changed.loc[future, "open"] *= 0.91
    changed.loc[future, "high"] = np.maximum(changed.loc[future, "open"], changed.loc[future, "close"]) * 1.03
    changed.loc[future, "low"] = np.minimum(changed.loc[future, "open"], changed.loc[future, "close"]) * 0.97
    changed.loc[future, "volume"] *= 2.0

    a = build_mother_features_v39(base)
    b = build_mother_features_v39(changed)
    cols = [
        "ict_score_v39", "smc_score_v39", "ichimoku_score_v39",
        "brooks_score_v39", "mtf_score_v39", "directional_prior_v39",
        "brooks_h1_v39", "brooks_h2_v39", "mother_event_v39",
        "research_candidate_side_v39",
    ]
    pd.testing.assert_frame_equal(
        a.loc[:cutoff, cols].reset_index(drop=True),
        b.loc[:cutoff, cols].reset_index(drop=True),
        check_dtype=False,
        check_exact=True,
    )


def test_all_independent_engine_scores_exist_and_are_bounded() -> None:
    x = build_mother_features_v39(_synthetic())
    for name in MOTHER_ENGINE_NAMES_V39:
        col = f"{name}_score_v39"
        assert col in x
        finite = pd.to_numeric(x[col], errors="coerce").dropna()
        assert len(finite) > 0
        assert bool((finite.abs() <= 1.0 + 1e-12).all())
    assert set(x["research_candidate_side_v39"].dropna().unique()).issubset({-1, 0, 1})


def test_neural_contract_is_finite_float32() -> None:
    frames = {"BTC/USDT": _synthetic(seed=314), "ETH/USDT": _synthetic(seed=1618)}
    panel, matrix = build_neural_panel_v39(frames)
    assert len(panel) == 1800
    assert matrix.dtype == np.float32
    assert matrix.shape[0] == 1800
    assert matrix.shape[1] > 50
    assert np.isfinite(matrix).all()


def test_psychology_governance_blocks_loss_loop_and_cooldown() -> None:
    ok, reason = psychology_allows_new_trade_v39(AccountStateV39(equity=1.0, peak_equity=1.0, consecutive_losses=3))
    assert not ok and reason == "LOSS_STREAK_LIMIT"
    ok, reason = psychology_allows_new_trade_v39(AccountStateV39(equity=1.0, peak_equity=1.0, cooldown_active=True))
    assert not ok and reason == "COOLDOWN_ACTIVE"


def test_trade_admission_requires_positive_lower_bound_and_respects_risk_caps() -> None:
    t = pd.Timestamp("2026-01-01T00:00:00Z")
    features = pd.DataFrame({
        "timestamp": [t, t],
        "symbol": ["BTC/USDT", "ETH/USDT"],
        "research_candidate_side_v39": [1, 1],
        "mother_event_v39": [1, 1],
        "close": [100.0, 200.0],
        "atr": [2.0, 4.0],
    })
    preds = pd.DataFrame({
        "timestamp": [t, t],
        "symbol": ["BTC/USDT", "ETH/USDT"],
        "expected_r": [0.8, 0.5],
        "lower_expected_r": [0.25, 0.10],
        "upper_expected_r": [1.2, 1.0],
    })
    state = AccountStateV39(equity=1.0, peak_equity=1.0)
    out = build_trade_proposals_v39(features, preds, state)
    assert len(out) == 2
    assert float(out["allocated_risk_fraction"].sum()) <= 0.020 + 1e-12
    assert float(out["allocated_risk_fraction"].max()) <= 0.005 + 1e-12
    assert float(out["position_weight"].sum()) <= 0.70 + 1e-12

    preds_bad = preds.copy()
    preds_bad["lower_expected_r"] = [-0.01, 0.0]
    rejected = build_trade_proposals_v39(features, preds_bad, state)
    assert rejected.empty
