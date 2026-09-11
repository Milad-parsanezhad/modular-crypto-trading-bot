from __future__ import annotations

import numpy as np
import pandas as pd

from research_bot.portfolio_mtm_v24c import MTMRiskContract
from research_bot.portfolio_ranked_v25 import simulate_ranked_mtm_portfolio
from research_bot.risk_ranker_v25 import (
    V25RankingContract,
    build_rank_features,
    fit_ranking_tournament,
    score_ranker,
)


def _ranking_rows(segment: str, groups: int, start: str) -> pd.DataFrame:
    base = pd.Timestamp(start, tz="UTC")
    rows = []
    for g in range(groups):
        t = base + pd.Timedelta(hours=4 * g)
        for j, strategy in enumerate(("H4_S6_BREAKOUT", "H4_D1_OB_BOS_RISK")):
            # Feature f_alpha is available at decision time. Its relation to the
            # realized outcome is synthetic and used only to test the ranking code.
            f_alpha = float((g % 7) / 6.0 + 0.45 * j)
            r = 1.2 if f_alpha > 0.75 else -0.8
            rows.append({
                "strategy": strategy,
                "family": strategy,
                "timeframe": "4h",
                "symbol": f"S{j}/USDT",
                "signal_time": t - pd.Timedelta(hours=4),
                "entry_time": t,
                "exit_time": t + pd.Timedelta(hours=4),
                "side": "long",
                "entry": 100.0,
                "stop": 98.0,
                "target": 104.0,
                "r_multiple": r,
                "segment": segment,
                "f_alpha": f_alpha,
                "f_beta": float(g % 3),
                "f_strategy_max_hold_bars": 4.0,
                "f_strategy_stop_atr": 1.0,
                "frozen_score": 0.7 + 0.02 * j,
                "frozen_threshold": 0.5,
                "frozen_selected": True,
                "frozen_score_margin": 0.2 + 0.02 * j,
            })
    return pd.DataFrame(rows)


def test_rank_features_do_not_use_outcome_columns():
    rows = _ranking_rows("development", 5, "2025-01-01")
    X, names = build_rank_features(rows)
    assert len(X) == len(rows)
    assert "r_multiple" not in names
    assert "exit_time" not in names
    assert all(name.startswith("f_") for name in names)


def test_tournament_uses_development_fit_and_validation_selection_only():
    dev = _ranking_rows("development", 60, "2025-01-01")
    val = _ranking_rows("validation", 35, "2025-03-01")
    board, snapshot = fit_ranking_tournament(dev, val, V25RankingContract())
    assert not board.empty
    assert snapshot["validation_only_selection"] is True
    assert snapshot["spent_v24d_external_used_for_selection"] is False
    assert snapshot["future_start_utc"] == "2026-09-11T12:00:00Z"
    assert snapshot["live_execution_authorized"] is False
    score = score_ranker(val, snapshot)
    assert len(score) == len(val)
    assert np.isfinite(score).all()


def _bars(symbol: str) -> pd.DataFrame:
    ts = pd.date_range("2025-01-01", periods=4, freq="4h", tz="UTC")
    return pd.DataFrame({
        "timestamp": ts,
        "open": [100.0] * 4,
        "high": [101.0] * 4,
        "low": [99.0] * 4,
        "close": [100.0] * 4,
        "volume": [1000.0] * 4,
        "source": [symbol] * 4,
    })


def test_priority_changes_same_timestamp_admission_without_reading_outcome():
    t = pd.Timestamp("2025-01-01T04:00:00Z")
    events = pd.DataFrame([
        {
            "strategy": "H4_S6_BREAKOUT", "symbol": "A/USDT", "side": "long",
            "signal_time": t - pd.Timedelta(hours=4), "entry_time": t, "exit_time": t + pd.Timedelta(hours=4),
            "entry": 100.0, "stop": 98.0, "r_multiple": -1.0,
        },
        {
            "strategy": "H4_D1_OB_BOS_RISK", "symbol": "B/USDT", "side": "long",
            "signal_time": t - pd.Timedelta(hours=4), "entry_time": t, "exit_time": t + pd.Timedelta(hours=4),
            "entry": 100.0, "stop": 98.0, "r_multiple": 2.0,
        },
    ])
    frames = {"A/USDT": _bars("A"), "B/USDT": _bars("B")}
    contract = MTMRiskContract(max_concurrent_positions=1, max_open_portfolio_risk=0.01)
    summary, ledger, _ = simulate_ranked_mtm_portfolio(
        events, frames, priority=[0.1, 0.9], selected=[True, True], contract=contract, mode="test"
    )
    accepted = ledger.loc[ledger["accepted"], "symbol"].tolist()
    assert accepted == ["B/USDT"]
    assert summary["priority_aware"] is True
    assert summary["live_execution_authorized"] if "live_execution_authorized" in summary else True
