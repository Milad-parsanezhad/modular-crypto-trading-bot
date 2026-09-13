# v0.50 — Final Research Deployment Record

Date: 2026-09-12
Status: **DEPLOYED / HEALTHY / FAIL-CLOSED RESEARCH MODE**

## Scientific source

- Private canonical repository: `parsa314/modular-crypto-trading-bot`
- Latest completed experiment: `v0.50`
- Canonical scientific head: `1dd0b1fe506fc51ceec4ff8934b77090f86b6cc2`
- Canonical workflow run: `34707823108`
- Canonical artifact: `10302830689`
- Artifact digest: `sha256:da5813a8c031f9da6cc940fda942efc846ca536dbd952f4930584c30732373e9`
- Scientific decision: `V50_NONOVERLAP_FAILURE_SUPPORTED`
- Next prospective research question: v0.51 overlap-conflict arbitration.

## Deployment readiness

A dedicated private deployment branch `deploy/research-v50` was created. The FastAPI service was refreshed to v0.50 fail-closed semantics and a startup execution firewall was added.

Deployment-readiness GitHub Actions run `34708929137` completed successfully, including:

- dependency installation;
- service smoke tests;
- v0.50 provenance/status assertions;
- fail-closed startup test when an execution flag is forced true;
- Railway configuration checks.

## Infrastructure constraints encountered

The intended new Railway service could not be provisioned because the account/project was at the plan resource limit. No existing service, volume, database or domain was deleted to bypass this limit.

The existing Railway thesis service was then selected for an in-place, non-destructive upgrade. Its historical private-repository binding could no longer fetch the current private repository state from Railway. Repeated snapshot redeploys therefore remained pinned to the old scientific snapshot. Legacy `railway.toml` configuration also overrode interim service-level start-command experiments.

These failures were treated as infrastructure defects, not scientific results. No trading model, gate, feature, risk rule or v0.50 evidence was changed.

## Deployment-only public shell

The user's existing empty public repository `parsa314/miladchicomobot` was repurposed strictly as a deployment shell. It contains no trading model, research data, exchange secret, API key, order-routing implementation or private thesis source.

The shell exposes only a fail-closed monitoring API/dashboard with:

- service version `1.0.0-rc3`;
- mode `RESEARCH_ONLY`;
- latest experiment `v0.50`;
- latest decision `V50_NONOVERLAP_FAILURE_SUPPORTED`;
- canonical v0.50 provenance identifiers;
- `LIVE_EXECUTION=false`;
- `PAPER_EXECUTION=false`;
- Kraken holdout `SEALED`;
- locked decision/paper execution endpoints.

Public-shell GitHub Actions smoke run `34709565125` passed, including the forced execution-firewall failure test.

## Railway production deployment

Existing Railway service (preserved):

- project: `crypto-intelligence-studio-v6`
- environment: `production`
- service: `thesis-trading-bot-v08`
- service ID: `17678ca8-f984-43c6-aa9b-68257ea056b9`
- domain: `thesis-trading-bot-v08-production.up.railway.app`

Safety variables were explicitly set to:

- `LIVE_EXECUTION=false`
- `PAPER_EXECUTION=false`
- `BOT_FORWARD_PAPER_ENABLED=false`
- `BOT_PAPER_EXECUTION_ENABLED=false`
- `BOT_RELEASE_CHANNEL=research-v50-fail-closed`

The service source was rebound in place to the deployment-only public repository, preserving the existing domain and resources.

Final fresh-source deployment:

- deployment ID: `b3ae45d1-c53a-4d8d-ab96-840e8be47d63`
- deployment source commit: `2c30a019ceab4d600dd6d17d6aa3069e78b4aeba`
- branch: `main`
- final status: **SUCCESS**
- start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
- healthcheck: `/health`
- runtime: Uvicorn listening on `0.0.0.0:8080`
- Railway health request: `GET /health HTTP/1.1` -> `200 OK`

## Engineering repair history

During installation, several pre-success deployment attempts failed for infrastructure-only reasons:

1. Free-plan service provisioning limit blocked a new resource.
2. Railway could not refresh the current private-repository source, leaving old snapshots pinned.
3. Interim wrapper deployments exposed a virtual-environment PATH mismatch.
4. Service watch patterns were inherited from the old repository and did not match the deployment-shell files.

Each problem was isolated and repaired without changing scientific results. The final solution was a fresh source deployment from the minimal public shell after its CI tests passed.

## Safety / interpretation

This is an **operational research and monitoring deployment**, not a live trading deployment and not evidence of profitability.

The scientific implementation remains private and canonical. The deployed shell cannot submit exchange orders and intentionally refuses paper/decision execution while the v0.50 scientific gate remains unresolved.

Final state:

`V50_RESEARCH_SERVICE_DEPLOYED / HEALTH_200 / RESEARCH_ONLY / LIVE_OFF / PAPER_OFF / KRAKEN_SEALED`
