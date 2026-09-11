import json

import numpy as np
import pandas as pd
import pytest

from research_bot.ml_framework_v23r import (
    CostContract,
    MLResearchContract,
    RiskContract,
    SplitContract,
    assert_unique_columns,
    chronological_purged_split,
    cost_aware_next_open_labels,
    economic_metrics,
    research_candidate_decision,
    select_feature_columns,
    supervised_model_registry,
    unsupervised_model_registry,
    write_contract_manifest,
)


def panel(n_times=120, symbols=("BTC/USDT", "ETH/USDT", "SOL/USDT")):
    times = pd.date_range("2024-01-01", periods=n_times, freq="4h", tz="UTC")
    rows = []
    for i, t in enumerate(times):
        for j, s in enumerate(symbols):
            rows.append({
                "signal_time": t,
                "label_end_time": t + pd.Timedelta(hours=8),
                "symbol": s,
                "timeframe": "4h",
                "side": 1 if (i + j) % 2 else -1,
                "f_ret1": np.sin(i / 11.0) + j * 0.01,
                "f_atr_pct": 0.01 + 0.001 * j,
                "label_positive_net": int((i + j) % 3 == 0),
            })
    return pd.DataFrame(rows)


def test_panel_split_keeps_same_timestamp_in_one_segment_only():
    df = panel()
    splits = chronological_purged_split(
        df,
        label_end_time_col="label_end_time",
        contract=SplitContract(embargo_rows=2),
    )
    timestamp_sets = {k: set(v.signal_time.tolist()) for k, v in splits.items()}
    assert not (timestamp_sets["development"] & timestamp_sets["validation"])
    assert not (timestamp_sets["validation"] & timestamp_sets["test"])
    assert not (timestamp_sets["development"] & timestamp_sets["test"])
    # Every retained timestamp carries the entire cross-sectional panel.
    for part in splits.values():
        assert set(part.groupby("signal_time").size().unique()) == {3}


def test_label_window_is_purged_before_next_segment():
    df = panel()
    splits = chronological_purged_split(df, label_end_time_col="label_end_time", contract=SplitContract(embargo_rows=1))
    assert splits["development"].label_end_time.max() < splits["validation"].signal_time.min()
    assert splits["validation"].label_end_time.max() < splits["test"].signal_time.min()


def test_duplicate_columns_fail_closed():
    df = pd.DataFrame([[1.0, 2.0]], columns=["f_x", "f_x"])
    with pytest.raises(ValueError, match="duplicate columns"):
        assert_unique_columns(df)


def test_feature_whitelist_excludes_identity_by_default():
    df = panel()
    numeric, categorical = select_feature_columns(df)
    assert numeric == ["f_atr_pct", "f_ret1"]
    assert "timeframe" in categorical and "side" in categorical
    assert "symbol" not in categorical


def test_outcome_like_feature_name_is_rejected_even_with_f_prefix():
    df = panel()
    df["f_future_return"] = 0.5
    with pytest.raises(ValueError, match="banned outcome-like"):
        select_feature_columns(df)


def test_cost_aware_label_uses_next_open_to_following_open_and_24bps_default():
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=5, freq="4h", tz="UTC"),
        "open": [100.0, 100.0, 101.0, 100.0, 102.0],
    })
    out = cost_aware_next_open_labels(frame, CostContract(fee_bps_each_way=10, slippage_bps_each_way=2))
    assert len(out) == 3
    # For signal at row 0: entry open[1]=100, exit open[2]=101 => 1% gross - 0.24% cost.
    assert np.isclose(out.loc[0, "label_future_gross_return"], 0.01)
    assert np.isclose(out.loc[0, "label_future_net_return"], 0.0076)
    assert int(out.loc[0, "label_positive_net"]) == 1
    assert out.loc[0, "label_end_time"] == frame.loc[2, "timestamp"]


def test_supervised_and_unsupervised_registries_cover_distinct_roles():
    supervised = supervised_model_registry(314)
    unsupervised = unsupervised_model_registry(314)
    assert {"logistic", "random_forest", "extra_trees", "hist_gradient_boosting"}.issubset(supervised)
    assert {"isolation_forest", "gmm_3", "gmm_5"}.issubset(unsupervised)
    assert not (set(supervised) & set(unsupervised))


def test_economic_metrics_are_unit_exposure_and_do_not_fake_risk_sizing():
    r = np.array([0.01, -0.005, 0.02, -0.002])
    m = economic_metrics(r, np.array([True, True, True, True]))
    assert m["selected"] == 4
    assert m["profit_factor"] > 1
    assert "unit_exposure_total_return" in m
    assert "unit_exposure_max_drawdown" in m


def test_research_decision_never_authorizes_paper_or_live():
    decision = research_candidate_decision({
        "selected": 150,
        "mean_net_return": 0.001,
        "profit_factor": 1.2,
    }, risk=RiskContract())
    assert decision["decision"] == "ML_RESEARCH_CANDIDATE"
    assert decision["forward_paper_authorized"] is False
    assert decision["live_execution_authorized"] is False


def test_contract_manifest_is_explicit_and_hashed(tmp_path):
    path = write_contract_manifest(tmp_path, MLResearchContract(), extra={"track": "audit2"})
    payload = json.loads(path.read_text())
    assert payload["contract"]["live_execution_authorized"] is False
    assert np.isclose(payload["contract"]["roundtrip_cost_fraction"], 0.0024)
    assert len(payload["manifest_sha256"]) == 64
