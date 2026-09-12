from __future__ import annotations

import numpy as np
import pandas as pd

from research_bot.canonical_strategy_v39r import (
    CanonicalConfig,
    build_canonical_features,
    canonical_manifest,
    canonical_score,
    reconstruction_metadata,
    risk_weight_from_stop,
)


def _synthetic_ohlcv(n: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(314)
    ts = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    drift = np.linspace(0.0, 25.0, n)
    cycle = 4.0 * np.sin(np.linspace(0.0, 14.0 * np.pi, n))
    noise = rng.normal(0.0, 0.55, n).cumsum() * 0.08
    close = 100.0 + drift + cycle + noise
    open_ = np.r_[close[0], close[:-1]] + rng.normal(0.0, 0.18, n)
    spread = np.abs(rng.normal(0.8, 0.18, n)) + 0.10
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = 1000.0 + 150.0 * np.abs(np.sin(np.linspace(0.0, 9.0 * np.pi, n))) + rng.uniform(0.0, 80.0, n)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def test_manifest_is_explicit_and_kraken_is_not_active() -> None:
    manifest = canonical_manifest()
    assert not manifest.empty
    assert manifest["component"].is_unique
    valid = {
        "FEATURE_ONLY",
        "CANDIDATE_GENERATOR",
        "ACTIVE_IN_RECONSTRUCTION",
        "REJECTED_UNDER_PROTOCOL",
        "SUPERSEDED",
        "UNFORMALIZED",
    }
    assert set(manifest["state"]).issubset(valid)
    kraken = manifest.loc[manifest["component"] == "kraken_holdout", "state"].iloc[0]
    assert kraken != "ACTIVE_IN_RECONSTRUCTION"


def test_canonical_output_is_finite_and_direction_is_discrete() -> None:
    scored = canonical_score(_synthetic_ohlcv())
    assert len(scored) == 500
    assert set(scored["canonical_direction"].dropna().unique()).issubset({-1, 0, 1})
    assert scored["canonical_long_score"].between(0, 7).all()
    assert scored["canonical_short_score"].between(0, 7).all()
    assert scored["canonical_conflict"].isin([0, 1]).all()


def test_reconstruction_is_causal_to_future_perturbation() -> None:
    base = _synthetic_ohlcv(520)
    cutoff = 390
    original = canonical_score(base)

    altered = base.copy()
    future = altered.index > cutoff
    altered.loc[future, "open"] *= 1.30
    altered.loc[future, "close"] *= 1.30
    altered.loc[future, "high"] = np.maximum(altered.loc[future, "open"], altered.loc[future, "close"]) * 1.01
    altered.loc[future, "low"] = np.minimum(altered.loc[future, "open"], altered.loc[future, "close"]) * 0.99
    altered.loc[future, "volume"] *= 4.0
    changed = canonical_score(altered)

    cols = [
        "canonical_cusum_event",
        "canonical_kumo_bull",
        "canonical_kumo_bear",
        "canonical_choch_up",
        "canonical_choch_down",
        "canonical_long_score",
        "canonical_short_score",
        "canonical_direction",
    ]
    pd.testing.assert_frame_equal(
        original.loc[:cutoff, cols].reset_index(drop=True),
        changed.loc[:cutoff, cols].reset_index(drop=True),
        check_dtype=True,
    )


def test_risk_weight_respects_asset_cap() -> None:
    features = build_canonical_features(_synthetic_ohlcv())
    cfg = CanonicalConfig(risk_per_trade=0.0025, max_asset_weight=0.35)
    weight = risk_weight_from_stop(features, cfg)
    assert (weight >= 0.0).all()
    assert (weight <= cfg.max_asset_weight + 1e-12).all()


def test_live_and_paper_execution_remain_disabled() -> None:
    meta = reconstruction_metadata()
    assert meta["version"] == "v0.39R"
    assert meta["live_execution"] is False
    assert meta["paper_execution"] is False
    assert meta["kraken_holdout"] == "SEALED"
    assert meta["promotion_status"] == "NOT_EVALUATED"
