from __future__ import annotations

import os
from dataclasses import asdict
from datetime import datetime, timezone

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .contracts import ExecutionMode
from .execution import ExecutionPolicy, PaperExecutionEngine
from .orchestrator import ResearchTradingOrchestrator
from .risk import RiskSnapshot


app = FastAPI(
    title="Modular Crypto Trading Bot — Research API",
    version="0.8.0",
    description="Evidence-driven research service. LIVE execution is disabled.",
)

_execution = PaperExecutionEngine(
    ExecutionPolicy(
        mode=ExecutionMode.PAPER,
        fee_bps=float(os.getenv("BOT_FEE_BPS", "10")),
        slippage_bps=float(os.getenv("BOT_SLIPPAGE_BPS", "2")),
        max_order_notional=float(os.getenv("BOT_MAX_ORDER_NOTIONAL", "5000")),
        live_execution_enabled=False,
    )
)
_orchestrator = ResearchTradingOrchestrator(execution_engine=_execution)


class ForecastRequest(BaseModel):
    asset: str = Field(min_length=2, max_length=32)
    symbol: str = Field(min_length=3, max_length=64)
    expected_return: float
    expected_cost: float = Field(ge=0)
    risk_penalty: float = Field(ge=0)
    uncertainty_penalty: float = Field(ge=0)
    confidence: float = Field(ge=0, le=1)
    currently_long: bool = False
    equity: float = Field(gt=0)
    peak_equity: float = Field(gt=0)
    gross_exposure: float = Field(ge=0)
    asset_weight: float = Field(ge=0)
    turnover: float = Field(ge=0)
    spread_bps: float = Field(ge=0)
    slippage_bps: float = Field(ge=0)
    reference_price: float = Field(gt=0)
    quantity: float = Field(gt=0)
    client_order_id: str = Field(min_length=4, max_length=128)
    recent_returns: list[float] = Field(default_factory=list)


def _safe(value):
    if isinstance(value, dict):
        return {k: _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if hasattr(value, "value"):
        return value.value
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "modular-crypto-research-bot",
        "version": "0.8.0",
        "execution_mode": "PAPER",
        "live_execution": False,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/research/status")
def research_status() -> dict:
    return {
        "principle": "Evidence Before Opinion",
        "live_execution": False,
        "execution_ladder": ["BACKTEST", "PAPER", "TESTNET", "LIVE_READINESS_AUDIT"],
        "active_modules": [
            "dynamic_universe_eligibility",
            "point_in_time_join",
            "ichimoku_candidate_detector",
            "cost_uncertainty_abstention",
            "independent_risk_engine",
            "paper_execution",
            "reproducibility_manifest",
        ],
        "git_commit": os.getenv("RAILWAY_GIT_COMMIT_SHA", os.getenv("GITHUB_SHA", "UNKNOWN")),
        "environment": os.getenv("RAILWAY_ENVIRONMENT_NAME", "UNKNOWN"),
        "warning": "No profitability claim is implied by service availability.",
    }


@app.post("/decision/evaluate")
def evaluate_forecast(req: ForecastRequest) -> dict:
    outcome = _orchestrator.process_forecast(
        asset=req.asset,
        symbol=req.symbol,
        timestamp=datetime.now(timezone.utc),
        expected_return=req.expected_return,
        expected_cost=req.expected_cost,
        risk_penalty=req.risk_penalty,
        uncertainty_penalty=req.uncertainty_penalty,
        confidence=req.confidence,
        currently_long=req.currently_long,
        risk_snapshot=RiskSnapshot(
            equity=req.equity,
            peak_equity=req.peak_equity,
            gross_exposure=req.gross_exposure,
            asset_weight=req.asset_weight,
            turnover=req.turnover,
            spread_bps=req.spread_bps,
            slippage_bps=req.slippage_bps,
            recent_returns=tuple(req.recent_returns),
        ),
        reference_price=req.reference_price,
        quantity=req.quantity,
        client_order_id=req.client_order_id,
        metadata={"source": "research_api_v08"},
    )
    return _safe(
        {
            "status": outcome.status,
            "signal": asdict(outcome.signal),
            "risk": asdict(outcome.risk) if outcome.risk is not None else None,
            "fill": asdict(outcome.fill) if outcome.fill is not None else None,
            "live_execution": False,
        }
    )
