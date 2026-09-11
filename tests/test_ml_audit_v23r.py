import numpy as np
import pandas as pd

from research_bot.ml_audit_v23r import (
    audit_dataset,
    audit_future_mutation_invariance,
    audit_prediction_selection,
    audit_prefix_invariance,
    write_audit_bundle,
)
from research_bot.ml_framework_v23r import SplitContract, chronological_purged_split


def market_frame(n=500):
    rng = np.random.default_rng(314)
    ret = rng.normal(0.0001, 0.01, n)
    close = 100 * np.exp(np.cumsum(ret))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * (1 + rng.uniform(0.0001, 0.004, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0.0001, 0.004, n))
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC"),
        "open": open_, "high": high, "low": low, "close": close,
        "volume": rng.lognormal(8, 0.4, n),
    })


def causal_builder(frame):
    x = frame.copy().sort_values("timestamp").reset_index(drop=True)
    out = pd.DataFrame({"signal_time": x.timestamp})
    out["f_ret1"] = x.close.pct_change()
    out["f_vol20"] = x.close.pct_change().shift(1).rolling(20, min_periods=5).std()
    out["f_volume_ratio"] = x.volume / x.volume.shift(1).rolling(20, min_periods=5).median()
    return out


def audited_dataset():
    base = market_frame(360)
    f = causal_builder(base)
    f["label_end_time"] = f.signal_time + pd.Timedelta(hours=8)
    f["label_positive_net"] = ((np.arange(len(f)) % 4) == 0).astype(int)
    f["timeframe"] = "4h"
    f["side"] = 1
    splits = chronological_purged_split(f, label_end_time_col="label_end_time", contract=SplitContract(embargo_rows=2))
    parts = []
    for name, part in splits.items():
        z = part.copy(); z["segment"] = name; parts.append(z)
    return pd.concat(parts, ignore_index=True).sort_values("signal_time").reset_index(drop=True)


def test_clean_dataset_audit_passes():
    report, summary = audit_dataset(audited_dataset())
    assert summary["decision"] == "AUDIT_PASS"
    assert not summary["fatal_codes"]
    assert (report["code"] == "TEMPORAL_SEPARATION").any()


def test_outcome_like_f_feature_fails_audit():
    df = audited_dataset()
    df["f_label_future_return"] = 1.0
    report, summary = audit_dataset(df)
    assert summary["decision"] == "AUDIT_FAIL"
    assert "FEATURE_WHITELIST" in summary["fatal_codes"]
    assert (report["message"].astype(str).str.contains("banned outcome-like")).any()


def test_prefix_invariance_passes_for_causal_builder():
    result = audit_prefix_invariance(market_frame(), causal_builder, cut_fraction=0.7)
    assert result["passed"] is True
    assert result["features_checked"] >= 3


def test_future_mutation_does_not_change_historical_features():
    result = audit_future_mutation_invariance(market_frame(), causal_builder, cut_fraction=0.7)
    assert result["passed"] is True
    assert not result["violating_features"]


def test_audit_detects_noncausal_builder():
    def noncausal(frame):
        x = frame.copy().reset_index(drop=True)
        out = pd.DataFrame({"signal_time": x.timestamp})
        # The centered rolling window uses future observations and must fail.
        out["f_bad"] = x.close.rolling(7, center=True, min_periods=1).mean()
        return out
    result = audit_future_mutation_invariance(market_frame(), noncausal, cut_fraction=0.7)
    assert result["passed"] is False
    assert "f_bad" in result["violating_features"]


def test_prediction_audit_rejects_test_selected_threshold():
    pred = pd.DataFrame({
        "segment": ["validation", "test"],
        "threshold_source": ["validation", "test"],
        "model_selection_source": ["validation", "validation"],
    })
    _, summary = audit_prediction_selection(pred)
    assert summary["decision"] == "AUDIT_FAIL"
    assert "TEST_THRESHOLD_FREEZE" in summary["fatal_codes"]


def test_prediction_audit_passes_when_test_is_read_only():
    pred = pd.DataFrame({
        "segment": ["validation", "test"],
        "threshold_source": ["validation", "validation"],
        "model_selection_source": ["validation", "validation"],
    })
    _, summary = audit_prediction_selection(pred)
    assert summary["decision"] == "AUDIT_PASS"


def test_audit_bundle_is_saved_with_hash(tmp_path):
    report, summary = audit_dataset(audited_dataset())
    csv_path, json_path = write_audit_bundle(tmp_path, report, summary)
    assert csv_path.exists() and json_path.exists()
    assert len(pd.read_json(json_path, typ="series")["report_sha256"]) == 64
