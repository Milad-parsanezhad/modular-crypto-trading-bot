from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def main():
    p = argparse.ArgumentParser(); p.add_argument("--output", default="artifacts/v1/release_manifest.json"); a = p.parse_args(); out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"release": "1.0.0-rc1", "generated_at": datetime.now(timezone.utc).isoformat(), "git_commit": os.getenv("GITHUB_SHA", os.getenv("RAILWAY_GIT_COMMIT_SHA", "UNKNOWN")), "principle": "Evidence Before Opinion", "research_milestones": {"v0.10": "purged live-universe OOS tournament — no learned model promoted", "v0.11": "multi-seed robustness/bootstrap/FDR — no learned model promoted", "v0.12": "external derivatives holdout — NO_INCREMENTAL_DERIVATIVES_EVIDENCE", "v0.13": "post-v0.12 forward microstructure collection", "v0.14": "persistent CoinEx BTC/ETH forward paper observation/execution"}, "execution": {"backtest": "implemented", "paper": "implemented", "testnet": "adapter/readiness gate only", "live": "DISABLED_FAIL_CLOSED"}, "forward_paper": {"strategy": "ICHIMOKU_SHADOW_V14", "label": "HYPOTHESIS_SHADOW_NOT_VALIDATED_ALPHA", "symbols": ["BTC/USDT", "ETH/USDT"], "timeframe": "4hour", "persistence": "PostgreSQL when DATABASE_URL is configured", "restart_deduplication": True, "independent_risk_gate": True}, "scientific_claim": "Engineering release candidate only. No profitability or live-trading claim. Forward paper evidence requires elapsed unseen market time."}
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8"); print(json.dumps(payload, indent=2))


if __name__ == "__main__": main()
