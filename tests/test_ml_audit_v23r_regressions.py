import numpy as np
import pandas as pd

from research_bot.ml_audit_v23r import audit_dataset
from research_bot.ml_framework_v23r import SplitContract, chronological_purged_split, select_feature_columns


def test_causal_realized_volatility_feature_is_allowed():
    df = pd.DataFrame({
        "f_realized_vol20": [0.01, 0.02, 0.015],
        "timeframe": ["4h"] * 3,
        "side": [1, 1, 1],
    })
    numeric, _ = select_feature_columns(df)
    assert "f_realized_vol20" in numeric


def test_panel_timestamp_repeats_are_not_duplicate_observations():
    times = pd.date_range("2024-01-01", periods=120, freq="4h", tz="UTC")
    rows = []
    for i, t in enumerate(times):
        for symbol in ("BTC/USDT", "ETH/USDT", "SOL/USDT"):
            rows.append({
                "signal_time": t,
                "label_end_time": t + pd.Timedelta(hours=8),
                "symbol": symbol,
                "timeframe": "4h",
                "side": 1,
                "f_ret1": np.sin(i / 10),
                "f_realized_vol20": 0.01 + i * 1e-6,
                "label_positive_net": int(i % 3 == 0),
            })
    df = pd.DataFrame(rows)
    splits = chronological_purged_split(df, label_end_time_col="label_end_time", contract=SplitContract(embargo_rows=2))
    parts = []
    for name, part in splits.items():
        z = part.copy(); z["segment"] = name; parts.append(z)
    audited = pd.concat(parts, ignore_index=True)
    _, summary = audit_dataset(audited)
    assert summary["decision"] == "AUDIT_PASS"
    assert summary["duplicate_observation_fraction"] == 0.0
    assert summary["panel_timestamp_repeat_fraction"] > 0.0
