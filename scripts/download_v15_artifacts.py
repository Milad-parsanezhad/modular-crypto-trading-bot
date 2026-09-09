from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import urllib.request
import zipfile


API = "https://api.github.com"
PREFIX = "v15-forward-evidence-"


def _request_json(url: str, token: str) -> dict:
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "modular-crypto-trading-bot-v16",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _request_bytes(url: str, token: str) -> bytes:
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "modular-crypto-trading-bot-v16",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def list_v15_artifacts(repo: str, token: str, max_artifacts: int = 120) -> list[dict]:
    found: list[dict] = []
    page = 1
    while len(found) < max_artifacts:
        payload = _request_json(f"{API}/repos/{repo}/actions/artifacts?per_page=100&page={page}", token)
        rows = list(payload.get("artifacts") or [])
        if not rows:
            break
        for row in rows:
            if row.get("expired"):
                continue
            if str(row.get("name") or "").startswith(PREFIX):
                found.append(row)
                if len(found) >= max_artifacts:
                    break
        if len(rows) < 100:
            break
        page += 1
    found.sort(key=lambda x: str(x.get("created_at") or ""))
    return found


def download_snapshots(repo: str, token: str, output_dir: Path, max_artifacts: int = 120) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    seen_captured: set[str] = set()
    for artifact in list_v15_artifacts(repo, token, max_artifacts=max_artifacts):
        artifact_id = int(artifact["id"])
        raw = _request_bytes(f"{API}/repos/{repo}/actions/artifacts/{artifact_id}/zip", token)
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            candidates = [x for x in zf.namelist() if x.endswith("v15_forward_evidence.json")]
            if not candidates:
                continue
            data = json.loads(zf.read(candidates[0]).decode("utf-8"))
        captured = str(data.get("captured_at") or "")
        if not captured or captured in seen_captured:
            continue
        seen_captured.add(captured)
        path = output_dir / f"{artifact_id}_v15_forward_evidence.json"
        path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        out.append(path)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=os.getenv("GITHUB_REPOSITORY", "parsa314/modular-crypto-trading-bot"))
    parser.add_argument("--token", default=os.getenv("GITHUB_TOKEN", ""))
    parser.add_argument("--output-dir", default="artifacts/v16_input")
    parser.add_argument("--max-artifacts", type=int, default=120)
    args = parser.parse_args()
    if not args.token:
        raise SystemExit("GITHUB_TOKEN is required")
    paths = download_snapshots(args.repo, args.token, Path(args.output_dir), max_artifacts=args.max_artifacts)
    print(json.dumps({"snapshots_downloaded": len(paths), "output_dir": args.output_dir}, sort_keys=True))
    if not paths:
        raise SystemExit("No v0.15 forward-evidence artifacts were found")


if __name__ == "__main__":
    main()
