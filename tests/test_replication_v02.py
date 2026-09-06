import numpy as np
import pandas as pd

from research_bot.replication import PurgedWalkForwardSplit, ewma_volatility_threshold, symmetric_cusum_events, triple_barrier_events


def _prices(n=800, seed=13):
    rng = np.random.default_rng(seed)
    ret = rng.normal(0.0001, 0.01, n)
    close = 20000 * np.exp(np.cumsum(ret))
    spread = close * rng.uniform(0.002, 0.015, n)
    return pd.DataFrame({"open": np.r_[close[0], close[:-1]], "high": close + spread, "low": close - spread, "close": close, "volume": rng.lognormal(9, 0.5, n)}, index=pd.date_range("2023-01-01", periods=n, freq="4h", tz="UTC"))


def test_cusum_and_triple_barrier_are_nonempty_and_ordered():
    df = _prices()
    threshold = ewma_volatility_threshold(df["close"], span=40, multiplier=0.8)
    events = symmetric_cusum_events(df["close"], threshold)
    labels = triple_barrier_events(df, events, volatility=threshold / 0.8, horizon_bars=12, allow_overlap=False)
    assert len(events) > 10
    assert len(labels) > 5
    assert set(labels["label"].unique()).issubset({-1, 0, 1})
    assert (labels["exit_position"] >= labels["entry_position"]).all()
    if len(labels) > 1:
        assert (labels["entry_position"].iloc[1:].to_numpy() > labels["exit_position"].iloc[:-1].to_numpy()).all()


def test_purged_walk_forward_has_gap_and_order():
    x = np.arange(1000)
    splitter = PurgedWalkForwardSplit(n_splits=4, min_train_size=300, purge_bars=12, embargo_bars=4)
    folds = list(splitter.split(x))
    assert len(folds) >= 3
    for train, test in folds:
        assert train.max() < test.min()
        assert test.min() - train.max() - 1 >= 16
