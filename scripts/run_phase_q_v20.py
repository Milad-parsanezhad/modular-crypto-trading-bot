from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

from research_bot.phase_q_v20 import evaluate_phase_q


def _load_snapshots(input_dir: Path) -> tuple[list[dict], list[dict]]:
    snapshots: list[dict] = []
    errors: list[dict] = []
    for path in sorted(input_dir.rglob("*.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append({"path": str(path), "error": f"{type(exc).__name__}:{exc}"})
            continue
        if row.get("version") == "v0.19" and row.get("research_status") == "PROSPECTIVE_MULTI_VENUE_MICROSTRUCTURE_COLLECTION_ONLY":
            snapshots.append(row)
    return snapshots, errors


def _render_markdown(result: dict, parse_errors: list[dict]) -> str:
    core = result["core_gate"]
    diag = result["diagnostics"]
    lines = [
        "# v0.20 Phase-Q Prospective Microstructure Quality Report",
        "",
        f"Generated as-of: `{result['as_of']}`",
        f"Decision: **{result['decision']}**",
        f"Evidence ledger SHA-256: `{result['evidence_ledger_sha256']}`",
        "",
        "## Frozen scientific contract",
        "",
        f"- Pilot start: `{result['protocol']['pilot_start_at']}`",
        f"- Earliest maturity: `{result['protocol']['maturity_not_before']}`",
        "- Countable evidence: scheduled GitHub run on `refs/heads/main` only.",
        "- Manual dispatch, PR smoke, backfill and tampered snapshots never increase maturity.",
        "- Forecast/trading horizon remains 4h; microstructure measurement target remains 30min.",
        "- Passing this gate permits only freezing the next 4h feature specification, not fitting or promoting a model.",
        "",
        "## Maturity",
        "",
        f"- Elapsed hours: **{core['elapsed_hours']:.2f} / {core['config']['min_elapsed_hours']}**",
        f"- Expected opportunities: **{core['expected_measurement_opportunities']} / {core['config']['min_expected_opportunities']}**",
        f"- Unique measurement slots: **{core['unique_measurement_slots']}**",
        f"- Countable snapshots: **{result['countable_snapshot_count']}**",
        f"- Excluded snapshots: **{result['excluded_snapshot_count']}**",
        "",
        "## Symbol quality",
        "",
        "| Symbol | Authorized slots | Coverage | p95 clock skew (s) | p95 trade staleness (s) | Provider failures |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for symbol, stats in core["symbols"].items():
        cov = stats["authorized_coverage_ratio"]
        skew = stats["p95_clock_skew_seconds"]
        stale = stats["p95_trade_staleness_seconds"]
        lines.append(
            f"| {symbol} | {stats['authorized_slots']} | {cov:.3f} | "
            f"{('NA' if skew is None else f'{skew:.2f}')} | "
            f"{('NA' if stale is None else f'{stale:.2f}')} | {stats['provider_failure_events']} |"
        )
    lines.extend([
        "",
        "## Operational diagnostics",
        "",
        f"- Duplicate-slot snapshots: **{diag['duplicate_slot_snapshots']}** ({diag['duplicate_slot_ratio']:.3%})",
        f"- Maximum consecutive missing slots between observations: **{diag['max_consecutive_missing_slots_between_observations']}**",
        f"- Snapshots with provider failure: **{diag['snapshots_with_provider_failure']}** ({diag['provider_failure_snapshot_ratio']:.3%})",
        f"- Provider failure events: **{diag['provider_failure_events']}**",
        f"- Warnings: **{', '.join(diag['warnings']) if diag['warnings'] else 'NONE'}**",
        f"- Input parse errors: **{len(parse_errors)}**",
        "",
        "## Authorization boundary",
        "",
        f"- Feature freeze authorized: **{result['feature_freeze_authorized']}**",
        f"- Predictive modeling authorized: **{result['predictive_modeling_authorized']}**",
        f"- PAPER strategy replacement authorized: **{result['paper_strategy_replacement_authorized']}**",
        f"- Testnet promotion authorized: **{result['testnet_promotion_authorized']}**",
        f"- LIVE execution authorized: **{result['live_execution_authorized']}**",
        f"- Next allowed action: **{result['next_allowed_action']}**",
        "",
        "Negative, immature and provider-limited outcomes are retained as first-class thesis evidence.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True)
    p.add_argument("--as-of", default=None, help="UTC timestamp; defaults to current UTC wall clock")
    p.add_argument("--output", default="artifacts/v20/phase_q_quality_gate.json")
    p.add_argument("--markdown", default="artifacts/v20/PHASE_Q_QUALITY_REPORT.md")
    args = p.parse_args()

    input_dir = Path(args.input_dir)
    snapshots, parse_errors = _load_snapshots(input_dir)
    as_of = args.as_of or datetime.now(timezone.utc).isoformat()
    result = evaluate_phase_q(snapshots, as_of=as_of)
    result["input_files_scanned"] = len(list(input_dir.rglob("*.json")))
    result["valid_v19_snapshot_files_loaded"] = len(snapshots)
    result["input_parse_errors"] = parse_errors
    result["report_payload_sha256"] = sha256(
        json.dumps(result, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()

    out = Path(args.output)
    md = Path(args.markdown)
    out.parent.mkdir(parents=True, exist_ok=True)
    md.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    md.write_text(_render_markdown(result, parse_errors), encoding="utf-8")
    print(json.dumps({
        "decision": result["decision"],
        "countable_snapshot_count": result["countable_snapshot_count"],
        "expected_measurement_opportunities": result["core_gate"]["expected_measurement_opportunities"],
        "unique_measurement_slots": result["core_gate"]["unique_measurement_slots"],
        "feature_freeze_authorized": result["feature_freeze_authorized"],
        "predictive_modeling_authorized": result["predictive_modeling_authorized"],
        "output": str(out),
        "markdown": str(md),
    }, indent=2))


if __name__ == "__main__":
    main()
