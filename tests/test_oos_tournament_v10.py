from __future__ import annotations

import numpy as np
import pandas as pd

from research_bot.oos_tournament_v10 import TournamentConfig, build_tournament_panel, run_oos_tournament


def _bars(seed: int, n: int = 720) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Weak autocorrelated component makes the test non-degenerate without
    # asserting profitability or a particular winner.
    eps = rng.normal(0.0, 0.006, n)
    ret = np.zeros(n)
    for i in range(1, n):
        ret[i] = 0.12 * ret[i - 1] + eps[i]
    close = 100.0 * np.exp(np.cumsum(ret))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * (1.0 + rng.uniform(0.001, 0.006, n))
    low = np.minimum(open_, close) * (1.0 - rng.uniform(0.001, 0.006, n))
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=n, freq="4h", tz="UTC"),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.uniform(1_000, 50_000, n),
        }
    )


def _universe(n_assets: int = 8):
    names = ["BTC", "ETH", "SOL", "XRP", "ADA", "DOT", "LINK", "LTC", "AAVE", "UNI"]
    return {f"{names[i]}/USDT": _bars(i + 1) for i in range(n_assets)}


def test_panel_target_is_forward_return_and_cs_features_are_timestamp_local():
    bars = _universe(6)
    cfg = TournamentConfig(horizon_bars=1, min_assets_per_timestamp=4)
    panel, features = build_tournament_panel(bars, cfg)
    btc = panel[panel["symbol"].eq("BTC/USDT")].sort_values("timestamp")
    source = bars["BTC/USDT"].copy()
    expected = source["close"].shift(-1) / source["close"] - 1.0
    merged = btc.merge(source[["timestamp"]].assign(expected=expected), on="timestamp", how="left")
    np.testing.assert_allclose(merged["future_return"].to_numpy(), merged["expected"].to_numpy(), rtol=1e-12, atol=1e-12)
    assert any(x.startswith("cs_") for x in features)
    cs = panel[[c for c in features if c.startswith("cs_")]].stack().dropna()
    assert cs.between(0.0, 1.0).all()


def test_oos_tournament_produces_cost_aware_purged_research_result_only():
    cfg = TournamentConfig(
        n_folds=2,
        train_fraction=0.60,
        top_quantile=0.25,
        one_way_cost_bps=12.0,
        label_hurdle_bps=8.0,
        min_assets_per_timestamp=5,
        min_train_timestamps=220,
        min_test_timestamps=60,
        random_state=7,
    )
    report = run_oos_tournament(_universe(8), cfg)
    assert report["research_status"] == "PURGED_OOS_ALPHA_TOURNAMENT_PILOT_NOT_LIVE_SIGNAL"
    assert report["fold_count"] == 2
    assert report["panel_rows"] > 1000
    variants = {x["variant"] for x in report["summary"]}
    assert {"logistic", "hgb", "random_forest", "momentum_baseline", "ichimoku_baseline", "equal_weight_market"}.issubset(variants)
    for row in report["summary"]:
        assert row["explicit_cost_sum"] >= 0
        assert row["net_total_return"] <= row["gross_total_return"] + 0.10  # compounding can make exact delta nonlinear
    assert report["promotion_status"] in {"NO_MODEL_PROMOTED", "PROVISIONAL_OOS_WINNER_NEEDS_BOOTSTRAP_AND_MULTI_SEED"}
    serialized = str(report).upper()
    assert "CONFIRMED_ENTRY" not in serialized
    assert "BUY_SIGNAL" not in serialized
