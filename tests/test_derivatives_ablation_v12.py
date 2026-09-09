from __future__ import annotations

import numpy as np
import pandas as pd

from research_bot.derivatives_ablation_v12 import V12Config, add_v12_features, run_v12_ablation


def _panel(n_assets: int = 7, n_times: int = 430, seed: int = 812) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2025-01-01", periods=n_times, freq="8h", tz="UTC")
    rows = []
    for j in range(n_assets):
        funding = rng.normal(0.00005, 0.00012, n_times)
        premium = 5.0 * funding + rng.normal(0.0, 0.00025, n_times)
        orderflow = np.clip(rng.normal(0.0, 0.10, n_times), -0.5, 0.5)
        oi_ret = rng.normal(0.0005, 0.015, n_times)
        oi = (1_000_000 + j * 100_000) * np.exp(np.cumsum(oi_ret))
        # Small predictable component is intentionally present only to ensure
        # the evaluation path has non-degenerate labels; unit tests do not
        # assert profitability or promotion.
        ret = (
            0.0001
            + 0.08 * np.r_[0.0, orderflow[:-1]] * 0.01
            - 0.04 * np.r_[0.0, funding[:-1]]
            + rng.normal(0.0, 0.006, n_times)
        )
        close = (100.0 + 12.0 * j) * np.exp(np.cumsum(ret))
        open_ = np.r_[close[0], close[:-1]]
        span = close * (0.002 + rng.uniform(0.0, 0.001, n_times))
        high = np.maximum(open_, close) + span
        low = np.minimum(open_, close) - span
        quote_volume = rng.lognormal(15.0, 0.35, n_times)
        buy_share = np.clip((1.0 + orderflow) / 2.0, 0.02, 0.98)
        taker_buy = quote_volume * buy_share
        for i, t in enumerate(ts):
            rows.append({
                "timestamp": t,
                "symbol": f"ASSET{j}USDT",
                "open": open_[i],
                "high": high[i],
                "low": low[i],
                "futures_close": close[i],
                "futures_quote_volume": quote_volume[i],
                "futures_taker_buy_quote": taker_buy[i],
                "funding_rate": funding[i],
                "funding_interval_hours": 8.0,
                "premium_index_close": premium[i],
                "binance_open_interest": oi[i],
                "binance_open_interest_value": oi[i] * close[i],
            })
    return pd.DataFrame(rows)


def test_v12_features_are_point_in_time_and_have_expected_families():
    feat, fam = add_v12_features(_panel(n_times=150))
    assert set(fam) == {"price", "ichimoku", "derivatives"}
    assert all(fam.values())
    assert not any("chikou" in c.lower() for cols in fam.values() for c in cols)
    assert "ichi_tenkan_kijun" in fam["ichimoku"]
    assert "deriv_funding" in fam["derivatives"]
    assert "deriv_orderflow_imbalance" in fam["derivatives"]
    assert "deriv_oi_change1" in fam["derivatives"]
    assert feat["future_return"].notna().sum() > 0


def test_v12_run_is_complete_and_fail_closed():
    cfg = V12Config(
        timeframe="8h",
        holdout_fraction=0.25,
        development_folds=2,
        top_quantile=0.30,
        one_way_cost_bps=8.0,
        min_assets_per_timestamp=6,
        min_development_timestamps=180,
        min_holdout_timestamps=70,
        seeds=(11, 42),
        bootstrap_samples=40,
        block_length=6,
        alpha=0.05,
        fdr_alpha=0.10,
        max_allowed_drawdown=-0.35,
        min_regime_periods=10,
        random_state=12012,
    )
    out = run_v12_ablation(_panel(), cfg)

    assert out["research_status"] == "V12_DERIVATIVES_EXTERNAL_HOLDOUT_NOT_TRADING_SIGNAL"
    assert out["panel_timestamps"] >= 400
    assert out["final_holdout_start"] > out["development_end"]
    assert out["multiple_testing"]["method"] == "BENJAMINI_HOCHBERG_FDR"
    assert out["multiple_testing"]["planned_tests"] == 10
    assert out["multiple_testing"]["realized_tests"] == 10
    assert len(out["holdout_summary"]) == 8  # 2 models x 4 feature-family variants
    assert set(out["feature_families"]) == {"price", "ichimoku", "derivatives"}
    assert out["provisional_best_full_model"] in {"logistic", "hgb"}
    assert out["promotion_status"] in {
        "NO_INCREMENTAL_DERIVATIVES_EVIDENCE",
        "PROVISIONAL_DERIVATIVES_EDGE_NEEDS_FORWARD_PAPER_REPLICATION",
    }
    assert "BUY_SIGNAL" not in str(out).upper()
    assert "CONFIRMED_ENTRY" not in str(out).upper()
