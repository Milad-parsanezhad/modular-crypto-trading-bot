from __future__ import annotations

import numpy as np
import pandas as pd

from research_bot.oos_tournament_v10 import TournamentConfig
from research_bot.robustness_v11 import (
    RobustnessConfig,
    benjamini_hochberg,
    moving_block_bootstrap_paired,
    run_robustness_v11,
)
from research_bot.universe import EligibilityPolicy, MarketListing, evaluate_listing


def test_benjamini_hochberg_known_ordering():
    q = benjamini_hochberg({"a": 0.001, "b": 0.02, "c": 0.04, "d": 0.50})
    assert set(q) == {"a", "b", "c", "d"}
    assert 0 <= q["a"] <= q["b"] <= q["c"] <= q["d"] <= 1
    assert q["a"] <= 0.01
    assert q["d"] == 0.50


def test_moving_block_bootstrap_detects_large_paired_edge():
    rng = np.random.default_rng(7)
    baseline = rng.normal(0.0001, 0.001, 240)
    candidate = baseline + 0.0015 + rng.normal(0.0, 0.00015, 240)
    out = moving_block_bootstrap_paired(
        candidate,
        baseline,
        samples=250,
        block_length=12,
        random_state=99,
        timeframe="4h",
    )
    assert out["status"] == "OK"
    assert out["n"] == 240
    assert out["observed_mean_return_diff"] > 0
    assert out["mean_return_diff_ci"][0] > 0
    assert out["one_sided_p_mean_edge"] < 0.05


def test_usdg_and_other_known_stable_bases_are_rejected():
    policy = EligibilityPolicy(
        min_volume_24h_quote=1.0,
        max_spread_bps=100.0,
        min_history_bars=1,
        max_missing_fraction=1.0,
        max_abnormal_fraction=1.0,
    )
    for base in ("USDG", "USDS", "USDP", "BUSD", "GUSD", "FRAX", "LUSD", "USDD", "EURC"):
        listing = MarketListing(
            exchange="fixture",
            symbol=f"{base}/USDT",
            base=base,
            quote="USDT",
            market_type="spot",
            active=True,
            volume_24h_quote=10_000_000.0,
            spread_bps=1.0,
            history_bars=1000,
            missing_fraction=0.0,
            abnormal_fraction=0.0,
        )
        result = evaluate_listing(listing, policy)
        assert not result.eligible, base
        assert "STABLECOIN_BASE" in result.reasons, (base, result.reasons)


def _bars(seed: int, asset: int, n: int = 360) -> pd.DataFrame:
    rng = np.random.default_rng(seed + asset * 101)
    ts = pd.date_range("2025-01-01", periods=n, freq="4h", tz="UTC")
    market = 0.00025 + 0.0010 * np.sin(np.arange(n) / 19.0)
    cross = (asset - 2.5) * 0.00008 * np.cos(np.arange(n) / 13.0)
    noise = rng.normal(0.0, 0.0045, n)
    returns = market + cross + noise
    close = (100.0 + asset * 7.0) * np.cumprod(1.0 + returns)
    open_ = np.r_[close[0], close[:-1]]
    spread = np.maximum(close * (0.0015 + rng.random(n) * 0.0015), 0.01)
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = 1_000_000.0 + asset * 100_000.0 + rng.lognormal(11.0, 0.25, n)
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


def test_run_robustness_v11_is_fail_closed_and_complete():
    universe = {f"ASSET{i}/USDT": _bars(100, i) for i in range(6)}
    tcfg = TournamentConfig(
        timeframe="4h",
        horizon_bars=1,
        n_folds=2,
        train_fraction=0.55,
        top_quantile=0.34,
        one_way_cost_bps=12.0,
        label_hurdle_bps=12.0,
        min_assets_per_timestamp=5,
        min_train_timestamps=120,
        min_test_timestamps=40,
        random_state=42,
    )
    rcfg = RobustnessConfig(
        seeds=(11, 42),
        bootstrap_samples=40,
        block_length=6,
        min_regime_periods=10,
        positive_seed_fraction_required=1.0,
        random_state=123,
    )
    out = run_robustness_v11(universe, tcfg, rcfg)

    assert out["research_status"] == "V11_ROBUSTNESS_INFERENCE_NOT_TRADING_SIGNAL"
    assert out["seed_count"] == 2
    assert set(out["seed_stability"]) == {
        "logistic",
        "hgb",
        "random_forest",
        "momentum_baseline",
        "ichimoku_baseline",
        "equal_weight_market",
    }
    assert out["provisional_best_model"] in {"logistic", "hgb", "random_forest"}
    assert out["strongest_baseline"] in {
        "momentum_baseline",
        "ichimoku_baseline",
        "equal_weight_market",
    }
    assert out["multiple_testing"]["method"] == "BENJAMINI_HOCHBERG_FDR"
    assert out["multiple_testing"]["tests"] == 9
    assert len(out["pairwise_bootstrap_inference"]) == 9
    assert out["promotion_status"] in {
        "NO_MODEL_PROMOTED",
        "PROVISIONAL_ROBUSTNESS_WINNER_STILL_NOT_EXECUTION_APPROVED",
    }
    assert "BUY_SIGNAL" not in str(out).upper()
    assert "CONFIRMED_ENTRY" not in str(out).upper()
