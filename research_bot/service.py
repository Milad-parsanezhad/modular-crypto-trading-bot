from __future__ import annotations

import os
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from .event_competing_risk_v41 import preregistration_manifest_v41
from .mother_strategy_v39 import mother_strategy_manifest_v39
from .two_stage_hurdle_v40 import preregistration_manifest_v40


SERVICE_VERSION = "1.0.0-rc2"
RESEARCH_MODE = "RESEARCH_ONLY"
LATEST_EXPERIMENT = "v0.41"
MOTHER_STRATEGY = "v0.39"
KRAKEN_STATE = "SEALED"
PAPER_EXECUTION = False
LIVE_EXECUTION = False

app = FastAPI(
    title="Modular Crypto Trading Bot — Thesis Research API",
    version=SERVICE_VERSION,
    description=(
        "Fail-closed academic crypto research service. "
        "PAPER and LIVE execution are disabled until an explicit scientific promotion gate authorizes them."
    ),
)


def _git_commit() -> str:
    return os.getenv("RAILWAY_GIT_COMMIT_SHA", os.getenv("GITHUB_SHA", "UNKNOWN"))


def _environment() -> str:
    return os.getenv("RAILWAY_ENVIRONMENT_NAME", "UNKNOWN")


@app.get("/")
def root() -> dict:
    return {
        "service": "modular-crypto-research-bot",
        "version": SERVICE_VERSION,
        "mode": RESEARCH_MODE,
        "mother_strategy": MOTHER_STRATEGY,
        "latest_experiment": LATEST_EXPERIMENT,
        "dashboard": "/dashboard",
        "docs": "/docs",
        "paper_execution": PAPER_EXECUTION,
        "live_execution": LIVE_EXECUTION,
        "kraken_holdout": KRAKEN_STATE,
    }


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "modular-crypto-research-bot",
        "version": SERVICE_VERSION,
        "execution_mode": RESEARCH_MODE,
        "paper_execution": PAPER_EXECUTION,
        "live_execution": LIVE_EXECUTION,
        "kraken_holdout": KRAKEN_STATE,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/research/status")
def research_status() -> dict:
    return {
        "principle": "Evidence Before Opinion",
        "version": SERVICE_VERSION,
        "mode": RESEARCH_MODE,
        "mother_strategy": MOTHER_STRATEGY,
        "latest_completed_experiment": "v0.40",
        "latest_completed_decision": "V40_DEVELOPMENT_REJECT_OR_INSUFFICIENT_EVIDENCE",
        "current_experiment": LATEST_EXPERIMENT,
        "current_hypothesis": "event-family-specific discrete-time competing risks",
        "development_venues": ["coinex", "okx", "kucoin"],
        "reserved_holdout": "kraken",
        "kraken_holdout": KRAKEN_STATE,
        "paper_execution": PAPER_EXECUTION,
        "live_execution": LIVE_EXECUTION,
        "git_commit": _git_commit(),
        "environment": _environment(),
        "warning": "Service availability and model predictions are not profitability evidence or execution authorization.",
    }


@app.get("/research/manifest/{version}")
def research_manifest(version: str) -> dict:
    manifests = {
        "v39": mother_strategy_manifest_v39,
        "v40": preregistration_manifest_v40,
        "v41": preregistration_manifest_v41,
    }
    fn = manifests.get(version.lower())
    if fn is None:
        raise HTTPException(status_code=404, detail="Unknown research version. Use v39, v40 or v41.")
    return fn()


@app.post("/decision/evaluate")
def evaluate_forecast(payload: dict) -> dict:
    del payload
    raise HTTPException(
        status_code=423,
        detail=(
            "Decision/execution endpoint is locked by the current thesis governance. "
            "Run frozen characterization through the CLI; PAPER/LIVE remain disabled."
        ),
    )


@app.get("/paper/status")
def paper_status() -> dict:
    return {
        "status": "DISABLED_BY_SCIENTIFIC_GATE",
        "paper_execution_enabled": False,
        "live_execution": False,
        "kraken_holdout": KRAKEN_STATE,
        "required_next_step": "development qualification before any external holdout or paper promotion",
    }


@app.post("/paper/run-once")
def paper_run_once() -> dict:
    raise HTTPException(
        status_code=423,
        detail="Forward PAPER execution is disabled until an explicit promotion gate is passed.",
    )


@app.get("/paper/observations")
def paper_observations() -> dict:
    return {"items": [], "status": "DISABLED_BY_SCIENTIFIC_GATE", "live_execution": False}


@app.get("/paper/fills")
def paper_fills() -> dict:
    return {"items": [], "status": "DISABLED_BY_SCIENTIFIC_GATE", "live_execution": False}


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard() -> str:
    return r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Thesis Crypto Research Bot</title>
<style>
body{font-family:system-ui,-apple-system,Segoe UI,sans-serif;margin:0;background:#f5f7fa;color:#17202a}
header{background:#111827;color:white;padding:22px 5vw}main{max-width:1100px;margin:24px auto;padding:0 20px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px}
.card{background:white;border:1px solid #dfe5ec;border-radius:10px;padding:16px;box-shadow:0 1px 3px #0001}
h1{font-size:22px;margin:0 0 6px}h2{font-size:16px;margin:0 0 12px}.metric{font-size:22px;font-weight:700}
.muted{color:#667085;font-size:13px}.ok{color:#067647}.locked{color:#b54708}.warn{background:#fff7ed;padding:12px;border-left:4px solid #f59e0b;margin:16px 0}
code{background:#f2f4f7;padding:2px 6px;border-radius:5px}
</style>
</head>
<body>
<header><h1>Modular Crypto Trading Bot — Thesis Research Dashboard</h1><div style="color:#cbd5e1">Evidence Before Opinion · fail-closed execution</div></header>
<main>
<div class="warn">This service is research-only. PAPER and LIVE execution are locked until a preregistered scientific promotion gate passes.</div>
<div class="grid">
<div class="card"><h2>Service</h2><div id="health" class="metric ok">…</div><div id="version" class="muted"></div></div>
<div class="card"><h2>Mode</h2><div id="mode" class="metric">…</div><div class="muted">No exchange order endpoint is enabled.</div></div>
<div class="card"><h2>Current experiment</h2><div id="experiment" class="metric">…</div><div class="muted">Event-specific competing risks</div></div>
<div class="card"><h2>Kraken holdout</h2><div id="kraken" class="metric locked">…</div><div class="muted">Untouched until development qualification.</div></div>
</div>
<div class="card" style="margin-top:16px"><h2>Research status</h2><pre id="research" style="white-space:pre-wrap">…</pre></div>
</main>
<script>
async function refresh(){
  const [h,r]=await Promise.all(['/health','/research/status'].map(u=>fetch(u).then(x=>x.json())));
  health.textContent=h.status; version.textContent=h.version; mode.textContent=h.execution_mode;
  experiment.textContent=r.current_experiment; kraken.textContent=r.kraken_holdout;
  research.textContent=JSON.stringify(r,null,2);
}
refresh(); setInterval(refresh,30000);
</script>
</body>
</html>'''
