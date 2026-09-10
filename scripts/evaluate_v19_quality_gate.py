from __future__ import annotations

import argparse
import json
from pathlib import Path

from research_bot.microstructure_quality_v19 import evaluate_quality_gate


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True, help="Directory containing v0.19 microstructure JSON snapshots")
    p.add_argument("--pilot-start-at", required=True, help="Frozen UTC start timestamp for the corrected pilot")
    p.add_argument("--as-of", default=None, help="Optional UTC evaluation timestamp; defaults to latest valid snapshot")
    p.add_argument("--output", default="artifacts/v19/quality_gate.json")
    args = p.parse_args()

    paths = sorted(Path(args.input_dir).glob("*.json"))
    snapshots = []
    errors = []
    for path in paths:
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
            if row.get("version") == "v0.19" and row.get("research_status") == "PROSPECTIVE_MULTI_VENUE_MICROSTRUCTURE_COLLECTION_ONLY":
                snapshots.append(row)
        except Exception as exc:
            errors.append({"path": str(path), "error": f"{type(exc).__name__}:{exc}"})

    result = evaluate_quality_gate(
        snapshots,
        pilot_start_at=args.pilot_start_at,
        as_of=args.as_of,
    )
    result["input_files_scanned"] = len(paths)
    result["valid_v19_snapshots_loaded"] = len(snapshots)
    result["input_parse_errors"] = errors

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps({
        "decision": result["decision"],
        "expected_measurement_opportunities": result["expected_measurement_opportunities"],
        "unique_measurement_slots": result["unique_measurement_slots"],
        "feature_predictive_modeling_authorized": result["feature_predictive_modeling_authorized"],
        "output": str(out),
    }, indent=2))


if __name__ == "__main__":
    main()
