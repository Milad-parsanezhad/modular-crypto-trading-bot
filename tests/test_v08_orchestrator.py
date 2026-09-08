from __future__ import annotations

from datetime import datetime, timezone

from research_bot.contracts import ExecutionMode
from research_bot.execution import ExecutionPolicy, PaperExecutionEngine
from research_bot.orchestrator import ResearchTradingOrchestrator
from research_bot.risk import RiskSnapshot


def test_forecast_to_paper_fill_is_gated_and_cost_aware():
    orchestrator = ResearchTradingOrchestrator(
        execution_engine=PaperExecutionEngine(
            ExecutionPolicy(
                mode=ExecutionMode.PAPER,
                fee_bps=10,
                slippage_bps=2,
                max_order_notional=20_000,
            )
        )
    )
    outcome = orchestrator.process_forecast(
        asset="BTC",
        symbol="BTC/USDT",
        timestamp=datetime(2026, 9, 8, tzinfo=timezone.utc),
        expected_return=0.012,
        expected_cost=0.0012,
        risk_penalty=0.001,
        uncertainty_penalty=0.001,
        confidence=0.80,
        currently_long=False,
        risk_snapshot=RiskSnapshot(
            equity=10_000,
            peak_equity=10_000,
            gross_exposure=0.0,
            asset_weight=0.0,
            turnover=0.0,
            spread_bps=4,
            slippage_bps=2,
            recent_returns=tuple([0.001] * 30),
        ),
        reference_price=100_000,
        quantity=0.05,
        client_order_id="v08-btc-001",
    )
    assert outcome.status == "EXECUTED_PAPER"
    assert outcome.fill is not None
    assert outcome.fill.mode is ExecutionMode.PAPER
    assert outcome.fill.fee_paid > 0


def test_low_net_alpha_abstains_before_risk_or_execution():
    orchestrator = ResearchTradingOrchestrator()
    outcome = orchestrator.process_forecast(
        asset="ETH",
        symbol="ETH/USDT",
        timestamp=datetime(2026, 9, 8, tzinfo=timezone.utc),
        expected_return=0.001,
        expected_cost=0.0012,
        risk_penalty=0.0003,
        uncertainty_penalty=0.0003,
        confidence=0.90,
        currently_long=False,
        risk_snapshot=RiskSnapshot(
            equity=10_000,
            peak_equity=10_000,
            gross_exposure=0.0,
            asset_weight=0.0,
            turnover=0.0,
            spread_bps=4,
            slippage_bps=2,
            recent_returns=tuple([0.001] * 30),
        ),
        reference_price=4_000,
        quantity=0.5,
        client_order_id="v08-eth-001",
    )
    assert outcome.status == "ABSTAINED"
    assert outcome.fill is None
    assert outcome.risk is None
