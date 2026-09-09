from __future__ import annotations

import asyncio
import os
from dataclasses import asdict
from datetime import datetime, timezone

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .contracts import ExecutionMode
from .execution import ExecutionPolicy, PaperExecutionEngine
from .forward_paper_v14 import ForwardPaperConfig, ForwardPaperRunner, STRATEGY_VERSION
from .orchestrator import ResearchTradingOrchestrator
from .persistence import store_from_environment
from .risk import RiskSnapshot


SERVICE_VERSION = "1.0.0-rc1"

app = FastAPI(title="Modular Crypto Trading Bot — Research & Forward Paper API", version=SERVICE_VERSION, description="Evidence-driven thesis system. Paper execution only; LIVE remains disabled.")

_execution = PaperExecutionEngine(ExecutionPolicy(mode=ExecutionMode.PAPER, fee_bps=float(os.getenv("BOT_FEE_BPS", "10")), slippage_bps=float(os.getenv("BOT_SLIPPAGE_BPS", "2")), max_order_notional=float(os.getenv("BOT_MAX_ORDER_NOTIONAL", "5000")), live_execution_enabled=False))
_orchestrator = ResearchTradingOrchestrator(execution_engine=_execution)
_paper_store = None
_paper_runner = None
_background_task: asyncio.Task | None = None
_last_cycle: dict | None = None
_last_cycle_error: str | None = None


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "true" if default else "false").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _paper_config() -> ForwardPaperConfig:
    symbols = tuple(x.strip() for x in os.getenv("BOT_SYMBOLS", "BTC/USDT,ETH/USDT").split(",") if x.strip())
    return ForwardPaperConfig(symbols=symbols or ("BTC/USDT", "ETH/USDT"), period=os.getenv("BOT_TIMEFRAME", "4hour"), bars=int(os.getenv("BOT_HISTORY_BARS", "280")), initial_cash=float(os.getenv("BOT_INITIAL_CASH", "10000")), risk_fraction=float(os.getenv("BOT_RISK_FRACTION", "0.005")), max_order_notional=float(os.getenv("BOT_MAX_ORDER_NOTIONAL", "2000")), entry_rule_score=float(os.getenv("BOT_ENTRY_RULE_SCORE", "0.80")), fee_bps=float(os.getenv("BOT_FEE_BPS", "10")), slippage_bps=float(os.getenv("BOT_SLIPPAGE_BPS", "2")), max_spread_bps=float(os.getenv("BOT_MAX_SPREAD_BPS", "35")), max_drawdown=float(os.getenv("BOT_MAX_DRAWDOWN", "0.12")), max_asset_weight=float(os.getenv("BOT_MAX_ASSET_WEIGHT", "0.25")))


def _runtime():
    global _paper_store, _paper_runner
    if _paper_store is None:
        cfg = _paper_config()
        _paper_store = store_from_environment(cfg.initial_cash)
        _paper_runner = ForwardPaperRunner(_paper_store, config=cfg, paper_execution_enabled=_env_bool("BOT_PAPER_EXECUTION_ENABLED", False))
    return _paper_store, _paper_runner


async def _forward_loop() -> None:
    global _last_cycle, _last_cycle_error
    interval = max(60, int(os.getenv("BOT_FORWARD_PAPER_INTERVAL_SECONDS", "900")))
    while True:
        try:
            _, runner = _runtime()
            _last_cycle = await asyncio.to_thread(runner.run_cycle)
            _last_cycle_error = None
        except Exception as exc:
            _last_cycle_error = f"{type(exc).__name__}: {exc}"
        await asyncio.sleep(interval)


@app.on_event("startup")
async def startup_forward_paper() -> None:
    global _background_task
    if _env_bool("BOT_FORWARD_PAPER_ENABLED", False) and _background_task is None:
        _runtime()
        _background_task = asyncio.create_task(_forward_loop())


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
    if isinstance(value, dict): return {k: _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_safe(v) for v in value]
    if hasattr(value, "value"): return value.value
    if hasattr(value, "isoformat"): return value.isoformat()
    return value


@app.get("/")
def root() -> dict:
    return {"service": "modular-crypto-research-bot", "version": SERVICE_VERSION, "dashboard": "/dashboard", "docs": "/docs", "live_execution": False}


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "modular-crypto-research-bot", "version": SERVICE_VERSION, "execution_mode": "PAPER", "live_execution": False, "forward_paper_enabled": _env_bool("BOT_FORWARD_PAPER_ENABLED", False), "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/research/status")
def research_status() -> dict:
    return {"principle": "Evidence Before Opinion", "version": SERVICE_VERSION, "live_execution": False, "latest_completed_research_gate": "v0.12 external derivatives holdout", "latest_research_decision": "NO_INCREMENTAL_DERIVATIVES_EVIDENCE", "forward_research": "v0.13 post-holdout microstructure collection", "forward_paper_strategy": STRATEGY_VERSION, "forward_paper_label": "HYPOTHESIS_SHADOW_NOT_VALIDATED_ALPHA", "execution_ladder": ["BACKTEST", "PAPER", "TESTNET", "LIVE_READINESS_AUDIT"], "active_modules": ["dynamic_universe_eligibility", "point_in_time_join", "ichimoku_candidate_detector", "v13_forward_microstructure_collection", "cost_uncertainty_abstention", "independent_risk_engine", "paper_execution", "postgres_persistence", "restart_deduplication", "reproducibility_manifest"], "git_commit": os.getenv("RAILWAY_GIT_COMMIT_SHA", os.getenv("GITHUB_SHA", "UNKNOWN")), "environment": os.getenv("RAILWAY_ENVIRONMENT_NAME", "UNKNOWN"), "warning": "Service availability and paper fills are not profitability evidence."}


@app.post("/decision/evaluate")
def evaluate_forecast(req: ForecastRequest) -> dict:
    outcome = _orchestrator.process_forecast(asset=req.asset, symbol=req.symbol, timestamp=datetime.now(timezone.utc), expected_return=req.expected_return, expected_cost=req.expected_cost, risk_penalty=req.risk_penalty, uncertainty_penalty=req.uncertainty_penalty, confidence=req.confidence, currently_long=req.currently_long, risk_snapshot=RiskSnapshot(equity=req.equity, peak_equity=req.peak_equity, gross_exposure=req.gross_exposure, asset_weight=req.asset_weight, turnover=req.turnover, spread_bps=req.spread_bps, slippage_bps=req.slippage_bps, recent_returns=tuple(req.recent_returns)), reference_price=req.reference_price, quantity=req.quantity, client_order_id=req.client_order_id, metadata={"source": "research_api_v1_rc1"})
    return _safe({"status": outcome.status, "signal": asdict(outcome.signal), "risk": asdict(outcome.risk) if outcome.risk is not None else None, "fill": asdict(outcome.fill) if outcome.fill is not None else None, "live_execution": False})


@app.get("/paper/status")
def paper_status() -> dict:
    try:
        store, _ = _runtime(); summary = store.summary(); account = summary.get("account") or {}; initial = _paper_config().initial_cash; equity = float(account.get("equity", initial)); peak = float(account.get("peak_equity", max(initial, equity)))
        return {"status": "RUNNING" if _env_bool("BOT_FORWARD_PAPER_ENABLED", False) else "READY_DISABLED", "strategy_version": STRATEGY_VERSION, "paper_execution_enabled": _env_bool("BOT_PAPER_EXECUTION_ENABLED", False), "live_execution": False, "store": summary, "cumulative_return": equity / initial - 1.0 if initial > 0 else None, "current_drawdown": equity / peak - 1.0 if peak > 0 else None, "last_cycle": _last_cycle, "last_cycle_error": _last_cycle_error}
    except Exception as exc:
        return {"status": "STORE_ERROR", "error": f"{type(exc).__name__}: {exc}", "live_execution": False}


@app.post("/paper/run-once")
def paper_run_once() -> dict:
    global _last_cycle, _last_cycle_error
    if not _env_bool("BOT_FORWARD_PAPER_ENABLED", False): return {"status": "DISABLED", "live_execution": False}
    try:
        _, runner = _runtime(); _last_cycle = runner.run_cycle(); _last_cycle_error = None; return _safe(_last_cycle)
    except Exception as exc:
        _last_cycle_error = f"{type(exc).__name__}: {exc}"; return {"status": "ERROR", "error": _last_cycle_error, "live_execution": False}


@app.get("/paper/observations")
def paper_observations(limit: int = Query(50, ge=1, le=500)) -> dict:
    store, _ = _runtime(); return {"items": _safe(store.recent_observations(limit)), "live_execution": False}


@app.get("/paper/fills")
def paper_fills(limit: int = Query(50, ge=1, le=500)) -> dict:
    store, _ = _runtime(); return {"items": _safe(store.recent_fills(limit)), "live_execution": False}


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard() -> str:
    return r'''<!doctype html><html lang="en"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/><title>Thesis Trading Bot — Research Dashboard</title><style>body{font-family:system-ui,-apple-system,Segoe UI,sans-serif;margin:0;background:#f5f7fa;color:#17202a}header{background:#111827;color:white;padding:22px 5vw}main{max-width:1200px;margin:24px auto;padding:0 20px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px}.card{background:white;border:1px solid #dfe5ec;border-radius:10px;padding:16px;box-shadow:0 1px 3px #0001}h1{font-size:22px;margin:0 0 6px}h2{font-size:16px;margin:0 0 12px}.metric{font-size:24px;font-weight:700}.muted{color:#667085;font-size:13px}table{width:100%;border-collapse:collapse;font-size:12px}th,td{padding:8px;border-bottom:1px solid #eee;text-align:left}.badge{display:inline-block;padding:4px 8px;border-radius:999px;background:#eef2ff;font-size:12px}.warn{background:#fff7ed;padding:10px;border-left:4px solid #f59e0b;margin:16px 0}</style></head><body><header><h1>Modular Crypto Trading Bot — Thesis Research Dashboard</h1><div class="muted" style="color:#cbd5e1">Evidence Before Opinion · PAPER only · LIVE disabled</div></header><main><div class="warn">Paper fills are forward research evidence, not a profitability claim or live-trading authorization.</div><div class="grid"><div class="card"><h2>Service</h2><div id="health" class="metric">…</div><div class="muted" id="version"></div></div><div class="card"><h2>Paper status</h2><div id="paper" class="metric">…</div><div class="muted" id="strategy"></div></div><div class="card"><h2>Paper return</h2><div id="ret" class="metric">…</div><div class="muted">mark-to-market since paper start</div></div><div class="card"><h2>Drawdown</h2><div id="dd" class="metric">…</div><div class="muted">current vs paper peak</div></div></div><div class="card" style="margin-top:16px"><h2>Research gate</h2><div id="research">…</div></div><div class="card" style="margin-top:16px"><h2>Recent observations</h2><div style="overflow:auto"><table><thead><tr><th>bar</th><th>symbol</th><th>action</th><th>score</th><th>price</th><th>spread bps</th></tr></thead><tbody id="obs"></tbody></table></div></div><div class="card" style="margin-top:16px"><h2>Recent paper fills</h2><div style="overflow:auto"><table><thead><tr><th>time</th><th>symbol</th><th>side</th><th>qty</th><th>price</th><th>fee</th><th>status</th></tr></thead><tbody id="fills"></tbody></table></div></div></main><script>const pct=x=>x==null?'—':(100*x).toFixed(2)+'%';async function refresh(){const [h,r,p,o,f]=await Promise.all(['/health','/research/status','/paper/status','/paper/observations?limit=30','/paper/fills?limit=30'].map(u=>fetch(u).then(x=>x.json())));health.textContent=h.status;version.textContent=h.version+' · '+h.execution_mode;paper.textContent=p.status;strategy.textContent=p.strategy_version||'';ret.textContent=pct(p.cumulative_return);dd.textContent=pct(p.current_drawdown);research.innerHTML='<span class="badge">'+r.latest_research_decision+'</span><p>'+r.warning+'</p>';obs.innerHTML=(o.items||[]).map(x=>`<tr><td>${x.bar_timestamp||''}</td><td>${x.symbol||''}</td><td>${x.action||''}</td><td>${Number(x.rule_score||0).toFixed(2)}</td><td>${x.reference_price||''}</td><td>${x.spread_bps==null?'':Number(x.spread_bps).toFixed(2)}</td></tr>`).join('');fills.innerHTML=(f.items||[]).map(x=>`<tr><td>${x.timestamp||''}</td><td>${x.symbol||''}</td><td>${x.side||''}</td><td>${x.filled_quantity||''}</td><td>${x.fill_price||''}</td><td>${x.fee_paid||''}</td><td>${x.status||''}</td></tr>`).join('');}refresh();setInterval(refresh,30000);</script></body></html>'''
