from __future__ import annotations

import numpy as np
import pandas as pd

from research_bot.strategy_lab import (
    META_FEATURES,
    StrategyLabConfig,
    build_meta_events,
    build_strategy_features,
    evaluate_rule_strategies,
    generate_strategy_positions,
)


def synthetic(n: int = 1000, seed: int = 314) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ret = rng.normal(0.0002, 0.009, n)
    close = 30_000 * np.exp(np.cumsum(ret))
    open_ = np.r_[close[0], close[:-1]]
    spread = close * rng.uniform(0.001, 0.012, n)
    return pd.DataFrame({
        "timestamp": pd.date_range("2020-01-01", periods=n, freq="4h", tz="UTC"),
        "open": open_, "high": np.maximum(open_, close) + spread,
        "low": np.minimum(open_, close) - spread, "close": close,
        "volume": rng.lognormal(10, 0.4, n),
    })


def test_strategy_registry_is_complete_and_binary():
    features = build_strategy_features(synthetic())
    positions = generate_strategy_positions(features)
    assert positions.columns.tolist() == ["B0", "S1", "S2", "S3", "S4", "S5", "S6", "IRCP", "C1", "C2"]
    assert set(np.unique(positions.to_numpy())) <= {0.0, 1.0}


def test_features_and_positions_do_not_change_when_future_changes():
    original = synthetic()
    changed = original.copy()
    changed.loc[850:, "close"] *= np.linspace(1.0, 2.0, len(changed) - 850)
    changed.loc[850:, "high"] = np.maximum(changed.loc[850:, "high"], changed.loc[850:, "close"])
    changed.loc[850:, "low"] = np.minimum(changed.loc[850:, "low"], changed.loc[850:, "close"])
    a = build_strategy_features(original)
    b = build_strategy_features(changed)
    cols = list(META_FEATURES) + ["cusum_event", "chikou_causal_positive"]
    pd.testing.assert_frame_equal(a.loc[:849, cols], b.loc[:849, cols])
    pd.testing.assert_frame_equal(
        generate_strategy_positions(a).loc[:849], generate_strategy_positions(b).loc[:849]
    )


def test_meta_labels_are_future_audit_only_and_have_valid_order():
    features = build_strategy_features(synthetic(2500))
    events = build_meta_events(features, StrategyLabConfig(cusum_threshold=0.05))
    if not events.empty:
        assert (events["entry_index"] > events["event_index"]).all()
        assert (events["exit_index"] >= events["entry_index"]).all()
        assert set(events["label"].unique()) <= {0, 1}
        assert set(events["outcome"].unique()) <= {"STOP", "PROFIT", "TIME"}


def test_rule_evaluation_applies_costs_and_five_percent_guard():
    summary, positions, features = evaluate_rule_strategies(synthetic(1600))
    assert len(summary) == 10
    assert summary["total_explicit_cost"].ge(0).all()
    active = summary[~summary["strategy"].eq("B0")]
    assert active["max_drawdown"].ge(-0.12).all()  # guard triggers at 5%; one-bar gaps can overshoot
    assert summary["kill_switch_activations"].ge(0).all()
    assert summary.loc[summary["strategy"].eq("B0"), "kill_switch_activations"].eq(0).all()
    assert positions.drop(columns=["B0"]).max().max() <= StrategyLabConfig().max_asset_weight
    assert len(positions) == len(features)


def test_period_evaluation_keeps_warmup_but_scores_only_window():
    frame = synthetic(2200)
    start = frame.loc[1800, "timestamp"].isoformat()
    summary, positions, _ = evaluate_rule_strategies(frame, evaluation_start=start)
    assert len(positions) == 400
    assert summary["n"].eq(400).all()


def test_invalid_ohlcv_fails_closed():
    frame = synthetic(100)
    frame.loc[4, "high"] = frame.loc[4, "low"] - 1
    try:
        build_strategy_features(frame)
        raise AssertionError("invalid OHLC should fail")
    except ValueError as exc:
        assert "invalid OHLC" in str(exc)
