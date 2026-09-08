from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from research_bot.backtest import backtest_positions
from research_bot.contracts import Decision, EvidenceStamp, ExecutionMode, ResearchStatus
from research_bot.decision import DecisionEngine, DecisionThresholds
from research_bot.execution import (
    ExecutionPolicy,
    ExecutionRequest,
    OrderSide,
    PaperExecutionEngine,
)
from research_bot.ichimoku_advanced import add_ichimoku_state, detect_kumo_triangle_breakout
from research_bot.point_in_time import assert_no_future_availability, point_in_time_asof_join
from research_bot.risk import RiskEngine, RiskLimits, RiskSnapshot
from research_bot.universe import EligibilityPolicy, MarketListing, coverage_report, evaluate_listing


UTC = timezone.utc


def _ohlcv(n: int = 160) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    ret = rng.normal(0.0005, 0.008, n)
    close = 100 * np.cumprod(1 + ret)
    high = close * (1 + rng.uniform(0.001, 0.01, n))
    low = close * (1 - rng.uniform(0.001, 0.01, n))
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=n, freq="4h", tz="UTC"),
            "open": np.r_[close[0], close[:-1]],
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.uniform(100, 1000, n),
        }
    )


def test_evidence_before_opinion_rejects_future_availability():
    now = datetime(2026, 1, 1, 12, tzinfo=UTC)
    future = EvidenceStamp(
        source="onchain-provider",
        observed_at=now,
        available_at=now + timedelta(hours=1),
        status=ResearchStatus.UNVERIFIED,
    )
    result = DecisionEngine().evaluate(
        asset="BTC",
        timestamp=now,
        expected_return=0.02,
        expected_cost=0.001,
        risk_penalty=0.001,
        uncertainty_penalty=0.001,
        confidence=0.9,
        evidence=(future,),
    )
    assert result.decision is Decision.REJECT
    assert "POINT_IN_TIME_EVIDENCE_NOT_AVAILABLE" in result.reasons


def test_cost_aware_abstention_requires_positive_net_alpha():
    engine = DecisionEngine(DecisionThresholds(min_confidence=0.6, min_net_alpha=0.001))
    now = datetime(2026, 1, 1, tzinfo=UTC)
    no_trade = engine.evaluate(
        asset="ETH",
        timestamp=now,
        expected_return=0.002,
        expected_cost=0.0015,
        risk_penalty=0.0004,
        uncertainty_penalty=0.0004,
        confidence=0.8,
    )
    assert no_trade.net_alpha < 0.001
    assert no_trade.decision is Decision.NO_TRADE

    candidate = engine.evaluate(
        asset="ETH",
        timestamp=now,
        expected_return=0.01,
        expected_cost=0.0015,
        risk_penalty=0.001,
        uncertainty_penalty=0.001,
        confidence=0.8,
    )
    assert candidate.decision is Decision.BUY_CANDIDATE


def test_risk_engine_kill_switch_on_drawdown():
    engine = RiskEngine(RiskLimits(max_drawdown=0.10))
    decision = engine.evaluate(
        RiskSnapshot(
            equity=89.0,
            peak_equity=100.0,
            gross_exposure=0.2,
            asset_weight=0.2,
            turnover=0.1,
            spread_bps=5,
            slippage_bps=2,
            recent_returns=tuple([0.001] * 30),
        )
    )
    assert not decision.approved
    assert decision.kill_switch
    assert "MAX_DRAWDOWN_BREACH" in decision.reasons


def test_paper_execution_costs_partial_fill_and_idempotency():
    engine = PaperExecutionEngine(
        ExecutionPolicy(
            mode=ExecutionMode.PAPER,
            fee_bps=10,
            slippage_bps=5,
            max_order_notional=10_000,
        )
    )
    request = ExecutionRequest(
        client_order_id="abc-1",
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        quantity=0.05,
        reference_price=100_000,
        created_at=datetime.now(UTC),
    )
    fill = engine.execute(request, fill_fraction=0.5)
    assert fill.status == "PARTIALLY_FILLED"
    assert fill.filled_quantity == pytest.approx(0.025)
    assert fill.fill_price > request.reference_price
    assert fill.fee_paid > 0
    assert fill.slippage_paid > 0
    with pytest.raises(RuntimeError, match="duplicate client_order_id"):
        engine.execute(request)


def test_live_execution_is_disabled_by_default():
    with pytest.raises(RuntimeError, match="LIVE execution is disabled"):
        PaperExecutionEngine(ExecutionPolicy(mode=ExecutionMode.LIVE))


def test_dynamic_universe_eligibility_and_unavailable_coverage():
    policy = EligibilityPolicy(min_volume_24h_quote=1_000_000, min_history_bars=100)
    btc = MarketListing(
        exchange="venue-a",
        symbol="BTC/USDT",
        base="BTC",
        quote="USDT",
        market_type="spot",
        active=True,
        volume_24h_quote=50_000_000,
        spread_bps=4,
        history_bars=1000,
        missing_fraction=0,
        abnormal_fraction=0,
    )
    usdc = MarketListing(
        exchange="venue-a",
        symbol="USDC/USDT",
        base="USDC",
        quote="USDT",
        market_type="spot",
        active=True,
        volume_24h_quote=50_000_000,
        spread_bps=2,
        history_bars=1000,
        missing_fraction=0,
        abnormal_fraction=0,
    )
    assert evaluate_listing(btc, policy).eligible
    stable = evaluate_listing(usdc, policy)
    assert not stable.eligible
    assert "STABLECOIN_BASE" in stable.reasons
    _, report = coverage_report([btc, usdc], policy=policy)
    assert report.eligible_assets == 1
    assert report.eligible_market_cap_coverage is None
    assert report.market_cap_status == "DATA_UNAVAILABLE"


def test_point_in_time_join_never_uses_future_release():
    decisions = pd.DataFrame(
        {
            "asset": ["BTC", "BTC"],
            "timestamp": ["2026-01-01T12:00:00Z", "2026-01-01T14:00:00Z"],
        }
    )
    features = pd.DataFrame(
        {
            "asset": ["BTC", "BTC"],
            "available_at": ["2026-01-01T11:00:00Z", "2026-01-01T13:00:00Z"],
            "mvrv": [1.1, 2.2],
        }
    )
    merged = point_in_time_asof_join(decisions, features)
    assert merged.loc[0, "mvrv"] == pytest.approx(1.1)
    assert merged.loc[1, "mvrv"] == pytest.approx(2.2)
    assert_no_future_availability(merged)


def test_ichimoku_feature_path_has_no_chikou_and_is_point_in_time():
    data = _ohlcv()
    out = add_ichimoku_state(data)
    assert "ichi_tk_distance_pct" in out.columns
    assert "ichi_cloud_width_pct" in out.columns
    assert not any("chikou" in c.lower() for c in out.columns)
    detector = detect_kumo_triangle_breakout(data)
    assert "triangle_candidate" in detector.columns
    assert set(detector["triangle_candidate"].dropna().unique()).issubset({0.0, 1.0})


def test_backtest_reports_tail_risk_and_explicit_costs():
    future_returns = pd.Series([0.01, -0.005, 0.002, -0.02] * 20)
    positions = pd.Series([1.0, 1.0, 0.0, 1.0] * 20)
    _, metrics = backtest_positions(
        future_returns,
        positions,
        timeframe="4h",
        fee_bps=10,
        slippage_bps=2,
    )
    assert "cvar_95_loss" in metrics
    assert "profit_factor_periods" in metrics
    assert metrics["total_explicit_cost"] > 0
    assert metrics["trade_events"] > 0
