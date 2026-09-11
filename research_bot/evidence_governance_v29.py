from __future__ import annotations

"""v0.29 evidence governance, provenance hashing and fail-closed promotion state.

This module does not create alpha and does not change any trading threshold.
Its purpose is to make the v0.27 -> v0.28 research path auditable and to ensure
that no downstream claim or authorization can outrun the evidence actually
produced by the frozen experiments.
"""

from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


V27_PASS = "ROBUST_CANDIDATE_PASSED_HOLDOUT_REQUIRES_FRESH_TEMPORAL_OOS"
V28_FORWARD = "FORWARD_PAPER_CANDIDATE"

V27_TERMINAL_NONPASS = {
    "NO_ROBUST_DEVELOPMENT_CANDIDATE",
    "ROBUST_CANDIDATE_LOCKED_HOLDOUT_UNAVAILABLE",
    "ROBUST_CANDIDATE_HOLDOUT_EVIDENCE_INSUFFICIENT",
    "ROBUST_CANDIDATE_REJECTED_HOLDOUT",
}
V28_NONPASS = {
    "BLOCKED_BY_V27",
    "FRESH_TEMPORAL_OOS_ACCUMULATING",
    "REJECTED_FRESH_TEMPORAL_OOS",
}


@dataclass(frozen=True)
class EvidenceRecordV29:
    stage: str
    version: str
    decision: str
    winner: str | None
    timeframe: str | None
    workflow_run_id: int | None
    commit_sha: str | None
    artifact_id: int | None
    artifact_sha256: str | None
    decision_sha256: str


def canonical_json_bytes(payload: Any) -> bytes:
    """Canonical JSON representation used for reproducible decision hashing."""
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_payload(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _assert_fail_closed_flags(payload: dict[str, Any], *, stage: str) -> None:
    if bool(payload.get("live_execution_authorized", False)):
        raise ValueError(f"{stage} illegally authorizes live execution")
    if bool(payload.get("paper_replacement_authorized", False)):
        raise ValueError(f"{stage} illegally authorizes paper replacement")
    if bool(payload.get("threshold_relaxation", False)):
        raise ValueError(f"{stage} contains threshold relaxation")
    if bool(payload.get("strategy_parameter_retuning", False)):
        raise ValueError(f"{stage} contains strategy retuning")
    if bool(payload.get("winner_reselection", False)) or bool(payload.get("holdout_winner_reselection", False)):
        raise ValueError(f"{stage} contains winner reselection")
    if bool(payload.get("historical_test_recycling", False)):
        raise ValueError(f"{stage} recycles inspected historical test evidence")


def validate_v27_v28_chain(
    v27: dict[str, Any],
    v28: dict[str, Any] | None,
) -> dict[str, Any]:
    """Validate the only scientifically permitted v0.27 -> v0.28 transitions."""
    if str(v27.get("version")) != "v0.27":
        raise ValueError("expected v0.27 decision payload")
    _assert_fail_closed_flags(v27, stage="v0.27")

    v27_state = str(v27.get("decision"))
    winner = v27.get("winner")
    timeframe = v27.get("timeframe")

    if v27_state == V27_PASS:
        if not winner or not timeframe:
            raise ValueError("v0.27 pass requires locked winner and timeframe")
        if not bool(v27.get("holdout_used", False)):
            raise ValueError("v0.27 pass requires the frozen holdout to have been used")
    elif v27_state in V27_TERMINAL_NONPASS:
        pass
    else:
        raise ValueError(f"unrecognized v0.27 decision: {v27_state}")

    if v28 is None:
        return {
            "chain_valid": True,
            "highest_stage": "v0.27",
            "promotion_state": "AWAITING_V28" if v27_state == V27_PASS else "STOPPED_AT_V27",
            "winner": winner,
            "timeframe": timeframe,
            "forward_paper_candidate_authorized": False,
            "paper_replacement_authorized": False,
            "live_execution_authorized": False,
        }

    if str(v28.get("version")) != "v0.28":
        raise ValueError("expected v0.28 decision payload")
    _assert_fail_closed_flags(v28, stage="v0.28")
    v28_state = str(v28.get("decision"))

    if v28.get("winner") != winner or v28.get("timeframe") != timeframe:
        raise ValueError("v0.28 winner/timeframe drifted from v0.27 lock")

    if v27_state != V27_PASS:
        if v28_state != "BLOCKED_BY_V27":
            raise ValueError("non-passing v0.27 may only transition to BLOCKED_BY_V27")
    elif v28_state not in V28_NONPASS | {V28_FORWARD}:
        raise ValueError(f"unrecognized v0.28 decision: {v28_state}")

    forward = v27_state == V27_PASS and v28_state == V28_FORWARD
    return {
        "chain_valid": True,
        "highest_stage": "v0.28",
        "promotion_state": "FORWARD_PAPER_CANDIDATE" if forward else (
            "TEMPORAL_EVIDENCE_ACCUMULATING" if v28_state == "FRESH_TEMPORAL_OOS_ACCUMULATING" else
            "STOPPED_AT_V28"
        ),
        "winner": winner,
        "timeframe": timeframe,
        "forward_paper_candidate_authorized": forward,
        "paper_replacement_authorized": False,
        "live_execution_authorized": False,
    }


def claim_policy(chain: dict[str, Any]) -> dict[str, bool]:
    """Return thesis/engineering claims permitted by the current evidence chain."""
    forward = bool(chain.get("forward_paper_candidate_authorized", False))
    return {
        "research_pipeline_operational": True,
        "causal_portfolio_allocator_implemented": True,
        "cross_venue_robustness_evaluated": True,
        "fresh_temporal_oos_passed": forward,
        "forward_paper_candidate": forward,
        "stable_alpha_proven": False,
        "future_profitability_guaranteed": False,
        "paper_replacement_authorized": False,
        "live_ready": False,
        "real_money_execution_authorized": False,
    }


def evidence_record(
    *,
    stage: str,
    decision_payload: dict[str, Any],
    workflow_run_id: int | None = None,
    commit_sha: str | None = None,
    artifact_id: int | None = None,
    artifact_sha256: str | None = None,
) -> EvidenceRecordV29:
    return EvidenceRecordV29(
        stage=stage,
        version=str(decision_payload.get("version")),
        decision=str(decision_payload.get("decision")),
        winner=None if decision_payload.get("winner") is None else str(decision_payload.get("winner")),
        timeframe=None if decision_payload.get("timeframe") is None else str(decision_payload.get("timeframe")),
        workflow_run_id=workflow_run_id,
        commit_sha=commit_sha,
        artifact_id=artifact_id,
        artifact_sha256=artifact_sha256,
        decision_sha256=sha256_payload(decision_payload),
    )


def hash_chain(records: Iterable[EvidenceRecordV29]) -> list[dict[str, Any]]:
    """Create an append-only cryptographic chain across evidence milestones."""
    previous = "0" * 64
    output: list[dict[str, Any]] = []
    for record in records:
        row = asdict(record)
        row["previous_chain_sha256"] = previous
        row["chain_sha256"] = sha256_payload({"previous": previous, "record": asdict(record)})
        previous = row["chain_sha256"]
        output.append(row)
    return output


def governance_bundle(
    v27: dict[str, Any],
    v28: dict[str, Any] | None,
    *,
    records: Iterable[EvidenceRecordV29] = (),
) -> dict[str, Any]:
    chain = validate_v27_v28_chain(v27, v28)
    cryptographic_chain = hash_chain(records)
    return {
        "version": "v0.29",
        "experiment": "EVIDENCE_GOVERNANCE_AND_REPRODUCIBILITY_GATE",
        "decision_chain": chain,
        "claim_policy": claim_policy(chain),
        "evidence_chain": cryptographic_chain,
        "evidence_chain_tip_sha256": cryptographic_chain[-1]["chain_sha256"] if cryptographic_chain else None,
        "threshold_relaxation": False,
        "strategy_parameter_retuning": False,
        "winner_reselection": False,
        "historical_test_recycling": False,
        "paper_replacement_authorized": False,
        "live_execution_authorized": False,
    }
