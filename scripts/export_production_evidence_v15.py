from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from research_bot.forward_evidence_v15 import build_forward_evidence, evidence_markdown


DEFAULT_BASE_URL = "https://thesis-trading-bot-v08-production.up.railway.app"


def fetch_json(url: str, retries: int = 5, timeout: int = 30) -> dict:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            req = Request(url, headers={"User-Agent": "modular-crypto-trading-bot-v15-evidence/1.0"})
            with urlopen(req, timeout=timeout) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"unexpected HTTP status {resp.status} for {url}")
                return json.loads(resp.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(2 + attempt * 2)
    raise RuntimeError(f"failed to fetch {url}: {last_error}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export v0.15 forward production-paper evidence")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--output-dir", default="artifacts/v15_forward_evidence")
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    endpoints = {
        "health": f"{base}/health",
        "research": f"{base}/research/status",
        "paper": f"{base}/paper/status",
        "observations": f"{base}/paper/observations?limit={args.limit}",
        "fills": f"{base}/paper/fills?limit={args.limit}",
    }
    raw = {name: fetch_json(url) for name, url in endpoints.items()}

    evidence = build_forward_evidence(
        health=raw["health"],
        research=raw["research"],
        paper=raw["paper"],
        observations_payload=raw["observations"],
        fills_payload=raw["fills"],
    )

    (output_dir / "v15_forward_evidence.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "v15_forward_evidence.md").write_text(evidence_markdown(evidence), encoding="utf-8")
    (output_dir / "raw_production_endpoints.json").write_text(
        json.dumps(raw, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )

    print(json.dumps({
        "evidence_state": evidence["scientific_state"]["evidence_state"],
        "live_promotion": evidence["scientific_state"]["live_promotion"],
        "safety_contract_ok": evidence["safety_contract"]["ok"],
        "observations": evidence["forward_counts"]["observations_total"],
        "fills": evidence["forward_counts"]["fills_total"],
        "equity_points": evidence["forward_counts"]["equity_points_total"],
        "cumulative_return": evidence["paper_account"]["cumulative_return"],
        "current_drawdown": evidence["paper_account"]["current_drawdown"],
    }, sort_keys=True))

    if not evidence["safety_contract"]["ok"]:
        return 2
    if evidence["scientific_state"]["live_promotion"] != "PROHIBITED":
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
