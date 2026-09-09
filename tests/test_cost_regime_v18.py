import numpy as np
import pandas as pd

from research_bot.cost_regime_v18 import V18Config, moving_block_ci, run_cost_aware_conversion, run_external_regime_replication


def _bars(seed: int, n: int = 900) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2025-01-01", periods=n, freq="4h", tz="UTC")
    drift = 0.0004 + 0.0008 * np.sin(np.arange(n) / 35.0)
    shocks = rng.normal(0, 0.006, n)
    close = 100 * np.exp(np.cumsum(drift + shocks))
    open_ = np.r_[close[0], close[:-1]]
    spread = np.abs(rng.normal(0.004, 0.001, n))
    high = np.maximum(open_, close) * (1 + spread)
    low = np.minimum(open_, close) * (1 - spread)
    volume = rng.lognormal(9, 0.4, n)
    return pd.DataFrame({"timestamp": ts, "open": open_, "high": high, "low": low, "close": close, "volume": volume})


def test_moving_block_ci_returns_interval():
    x = pd.Series(np.linspace(-0.001, 0.002, 200))
    out = moving_block_ci(x, resamples=100, block=12, seed=1)
    assert out["n"] == 200
    assert out["ci_low"] is not None
    assert out["ci_high"] is not None
    assert out["ci_low"] <= out["mean"] <= out["ci_high"]


def test_cost_aware_conversion_is_fail_closed_and_auditable():
    cfg = V18Config(train_fraction=0.65, bootstrap_resamples=100, min_holdout_periods=100)
    out = run_cost_aware_conversion({"BTC/USDT": _bars(1), "ETH/USDT": _bars(2)}, cfg)
    assert out["experiment"] == "A_COST_AWARE_ALPHA_CONVERSION"
    assert out["decision"] in {"COST_AWARE_CONVERSION_SUPPORTED", "NO_COST_AWARE_CONVERSION_EVIDENCE"}
    assert out["claim_scope"].endswith("not live authorization")
    assert out["naive_portfolio"]["n"] >= 100
    assert out["cost_aware_portfolio"]["turnover_sum"] <= out["naive_portfolio"]["turnover_sum"] + 1e-12
    for row in out["per_symbol"]:
        assert row["total_hurdle"] >= row["round_trip_cost_hurdle"]


def test_external_regime_replication_uses_frozen_regimes():
    cfg = V18Config(bootstrap_resamples=100, min_holdout_periods=100)
    out = run_external_regime_replication({"BTC/USDT": _bars(3), "ETH/USDT": _bars(4)}, cfg)
    assert out["experiment"] == "B_EXTERNAL_REGIME_ICHIMOKU_REPLICATION"
    assert out["frozen_favorable_regimes"] == ["HIGH_VOL", "TREND_DOWN"]
    assert out["decision"] in {"EXTERNAL_REGIME_REPLICATION_SUPPORTED", "NO_EXTERNAL_REGIME_REPLICATION_EVIDENCE"}
    assert out["claim_scope"].endswith("not live authorization")
