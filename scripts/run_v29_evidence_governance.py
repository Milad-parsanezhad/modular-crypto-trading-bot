from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from research_bot.evidence_governance_v29 import (
    evidence_record,
    governance_bundle,
    sha256_file,
)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="v0.29 evidence governance and thesis reproducibility bundle")
    ap.add_argument("--v27-artifact-dir", required=True)
    ap.add_argument("--v28-artifact-dir")
    ap.add_argument("--v27-run-id", type=int)
    ap.add_argument("--v27-commit-sha")
    ap.add_argument("--v27-artifact-id", type=int)
    ap.add_argument("--v28-run-id", type=int)
    ap.add_argument("--v28-commit-sha")
    ap.add_argument("--v28-artifact-id", type=int)
    ap.add_argument("--output-dir", default="artifacts/v29-evidence-governance")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    v27_dir = Path(args.v27_artifact_dir)
    v27_decision_path = v27_dir / "decision_v27.json"
    v27 = load_json(v27_decision_path)

    v28 = None
    v28_decision_path: Path | None = None
    if args.v28_artifact_dir:
        v28_dir = Path(args.v28_artifact_dir)
        candidate = v28_dir / "decision_v28.json"
        if candidate.exists():
            v28_decision_path = candidate
            v28 = load_json(candidate)

    records = [
        evidence_record(
            stage="V27_CROSS_VENUE_HOLDOUT",
            decision_payload=v27,
            workflow_run_id=args.v27_run_id,
            commit_sha=args.v27_commit_sha,
            artifact_id=args.v27_artifact_id,
            artifact_sha256=sha256_file(v27_decision_path),
        )
    ]
    if v28 is not None and v28_decision_path is not None:
        records.append(
            evidence_record(
                stage="V28_FRESH_TEMPORAL_OOS",
                decision_payload=v28,
                workflow_run_id=args.v28_run_id,
                commit_sha=args.v28_commit_sha,
                artifact_id=args.v28_artifact_id,
                artifact_sha256=sha256_file(v28_decision_path),
            )
        )

    bundle = governance_bundle(v27, v28, records=records)
    bundle["source_files"] = {
        "v27_decision": {
            "path": str(v27_decision_path),
            "sha256": sha256_file(v27_decision_path),
        },
        "v28_decision": None if v28_decision_path is None else {
            "path": str(v28_decision_path),
            "sha256": sha256_file(v28_decision_path),
        },
    }
    bundle["reproducibility_contract"] = {
        "canonical_json": True,
        "sha256_decision_hashing": True,
        "append_only_evidence_chain": True,
        "negative_results_preserved": True,
        "threshold_relaxation": False,
        "winner_reselection": False,
        "live_execution_authorized": False,
    }

    write_json(out / "evidence_governance_v29.json", bundle)
    write_json(out / "claim_policy_v29.json", bundle["claim_policy"])
    write_json(out / "promotion_state_v29.json", bundle["decision_chain"])
    write_json(out / "evidence_chain_v29.json", bundle["evidence_chain"])
    print(json.dumps(bundle["decision_chain"], indent=2, sort_keys=True))
    print(json.dumps(bundle["claim_policy"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
