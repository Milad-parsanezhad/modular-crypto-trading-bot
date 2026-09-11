import numpy as np
import pandas as pd

from research_bot.strategy_family_meta_v24b import (
    V24BContract,
    choose_family_threshold,
    family_sample_table,
    merge_family_predictions,
    portfolio_overlap_backtest,
)


def event_rows(n=120, seed=24):
    rng = np.random.default_rng(seed)
    t = pd.date_range("2025-01-01", periods=n, freq="8h", tz="UTC")
    good = rng.random(n) < 0.40
    r = np.where(good, rng.uniform(0.5, 3.0, n), -rng.uniform(0.5, 1.2, n))
    return pd.DataFrame({
        "strategy": "S",
        "symbol": np.where(np.arange(n) % 2, "BTC/USDT", "ETH/USDT"),
        "signal_time": t - pd.Timedelta(hours=4),
        "entry_time": t,
        "exit_time": t + pd.Timedelta(hours=4),
        "side": np.where(np.arange(n) % 3, 1, -1),
        "r_multiple": r,
        "entry": 100.0,
        "stop": 99.0,
        "target": 103.0,
        "exit_reason": np.where(r > 0, "target", "stop"),
    })


def test_family_eligibility_depends_only_on_sample_counts():
    parts = []
    for strategy, counts in {
        "ENOUGH": (300, 80, 100),
        "SHORT_DEV": (299, 80, 100),
        "SHORT_VAL": (300, 79, 100),
        "SHORT_SHADOW": (300, 80, 99),
    }.items():
        for segment, n in zip(("development", "validation", "test"), counts):
            parts.append(pd.DataFrame({"strategy": strategy, "segment": segment, "label_meta_execute": np.arange(n) % 2}))
    df = pd.concat(parts, ignore_index=True)
    a = family_sample_table(df).set_index("strategy")
    df["label_meta_execute"] = 1 - df["label_meta_execute"]
    b = family_sample_table(df).set_index("strategy")
    assert bool(a.loc["ENOUGH", "eligible_by_counts_only"])
    assert not bool(a.loc["SHORT_DEV", "eligible_by_counts_only"])
    assert not bool(a.loc["SHORT_VAL", "eligible_by_counts_only"])
    assert not bool(a.loc["SHORT_SHADOW", "eligible_by_counts_only"])
    assert a["eligible_by_counts_only"].equals(b["eligible_by_counts_only"])


def test_family_threshold_can_abstain_from_known_bad_events():
    rows = event_rows(140)
    score = np.where(rows["r_multiple"].to_numpy() > 0, 0.9, 0.1)
    contract = V24BContract(min_validation_selected=20)
    threshold, result = choose_family_threshold(rows, score, contract)
    assert 0 <= threshold <= 1
    assert result["filtered"]["selected"] >= 20
    assert result["filtered"]["mean_r"] > result["base"]["mean_r"]
    assert result["filtered"]["profit_factor"] > result["base"]["profit_factor"]


def test_portfolio_rejects_same_symbol_overlap_and_respects_risk_cap():
    t = pd.Timestamp("2026-01-01T00:00:00Z")
    rows = pd.DataFrame([
        {"strategy": "A", "symbol": "BTC/USDT", "signal_time": t - pd.Timedelta(hours=4), "entry_time": t, "exit_time": t + pd.Timedelta(hours=12), "side": 1, "r_multiple": 1.0},
        {"strategy": "B", "symbol": "BTC/USDT", "signal_time": t - pd.Timedelta(hours=4), "entry_time": t, "exit_time": t + pd.Timedelta(hours=8), "side": 1, "r_multiple": 1.0},
        {"strategy": "C", "symbol": "ETH/USDT", "signal_time": t - pd.Timedelta(hours=4), "entry_time": t, "exit_time": t + pd.Timedelta(hours=8), "side": 1, "r_multiple": 1.0},
        {"strategy": "D", "symbol": "SOL/USDT", "signal_time": t - pd.Timedelta(hours=4), "entry_time": t, "exit_time": t + pd.Timedelta(hours=8), "side": 1, "r_multiple": 1.0},
        {"strategy": "E", "symbol": "XRP/USDT", "signal_time": t - pd.Timedelta(hours=4), "entry_time": t, "exit_time": t + pd.Timedelta(hours=8), "side": -1, "r_multiple": 1.0},
    ])
    summary, ledger = portfolio_overlap_backtest(rows)
    assert (ledger["reject_reason"] == "SYMBOL_OVERLAP_CAP").sum() == 1
    assert summary["max_open_risk_fraction_seen"] <= V24BContract().max_open_risk_fraction + 1e-12
    assert summary["max_concurrent"] <= V24BContract().max_concurrent_positions


def test_hard_drawdown_kill_is_fail_closed_for_later_entries():
    t = pd.Timestamp("2026-01-01T00:00:00Z")
    rows = pd.DataFrame([
        {"strategy": "A", "symbol": "BTC/USDT", "signal_time": t - pd.Timedelta(hours=4), "entry_time": t, "exit_time": t + pd.Timedelta(hours=4), "side": 1, "r_multiple": -25.0},
        {"strategy": "B", "symbol": "ETH/USDT", "signal_time": t + pd.Timedelta(hours=4), "entry_time": t + pd.Timedelta(hours=8), "exit_time": t + pd.Timedelta(hours=12), "side": 1, "r_multiple": 2.0},
    ])
    summary, ledger = portfolio_overlap_backtest(rows)
    assert summary["hard_kill_triggered"] is True
    assert ledger.iloc[1]["reject_reason"] == "HARD_REALIZED_DRAWDOWN_KILL"


def test_unsupported_family_defaults_to_reject_when_predictions_are_merged():
    rows = event_rows(3)
    rows.loc[0, "strategy"] = "SUPPORTED"
    rows.loc[1:, "strategy"] = "UNSUPPORTED"
    pred = rows.iloc[[0]][["strategy", "symbol", "signal_time"]].copy()
    pred["family_meta_selected"] = True
    pred["family_meta_score"] = 0.9
    pred["family_champion_model"] = "ridge_classifier"
    pred["family_frozen_threshold"] = 0.5
    merged = merge_family_predictions(rows, pred)
    assert bool(merged.iloc[0]["family_meta_selected"])
    assert not bool(merged.iloc[1]["family_meta_selected"])
    assert not bool(merged.iloc[2]["family_meta_selected"])


def test_contract_permanently_disables_paper_and_live_for_reused_shadow():
    c = V24BContract()
    assert c.reused_shadow_test is True
    assert c.forward_paper_authorized is False
    assert c.paper_replacement_authorized is False
    assert c.live_execution_authorized is False
