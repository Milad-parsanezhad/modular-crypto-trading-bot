from __future__ import annotations

"""v0.42 fail-closed production-readiness gate.

This gate assesses scientific and operational readiness for *supervised paper*
deployment only. It cannot authorize real-money trading. Human approval is a
mandatory future prerequisite, and live execution remains disabled.
"""

from typing import Any

REQUIRED_CONTROLS_V42 = (
    "scientific_parent_valid",
    "shadow_engine_valid",
    "paper_path_ready",
    "kraken_sealed",
    "live_disabled",
    "idempotency_control",
    "pretrade_risk_limits",
    "kill_switch",
    "reconciliation",
    "audit_log",
    "key_management_plan",
    "rollback_plan",
    "human_approval_required",
)


def preregistration_manifest_v42() -> dict[str, Any]:
    return {
        "version": "v0.42",
        "experiment": "PRODUCTION_READINESS_HUMAN_APPROVAL_GATE",
        "scope": "supervised paper/shadow readiness only",
        "required_controls": list(REQUIRED_CONTROLS_V42),
        "manual_human_approval_required": True,
        "live_execution": False,
        "real_money_trading_authorized": False,
        "kraken_touched": False,
    }


def readiness_matrix_v42(*, d38: dict[str, Any], d39: dict[str, Any], d40: dict[str, Any], d41: dict[str, Any]) -> dict[str, Any]:
    scientific_parent_valid = (
        d38.get("decision") == "V38_SOFT_PORTFOLIO_CANDIDATE_LOCKED_FOR_WALK_FORWARD"
        and d39.get("decision") == "V39_TEMPORALLY_STABLE_CANDIDATE_LOCKED_FOR_EXECUTION_STRESS"
        and d40.get("decision") == "V40_EXECUTION_ROBUST_CANDIDATE_LOCKED_FOR_SHADOW"
    )
    kraken_sealed = not any(bool(d.get("kraken_touched", False)) for d in (d38, d39, d40, d41))
    engine_valid = bool(d41.get("engine_validated", False))
    paper_ready = bool(d41.get("paper_path_ready", False))
    matrix = {
        "scientific_parent_valid": bool(scientific_parent_valid),
        "shadow_engine_valid": engine_valid,
        "paper_path_ready": paper_ready,
        "kraken_sealed": bool(kraken_sealed),
        "live_disabled": not bool(d41.get("live_execution_authorized", False)) and not bool(d41.get("real_order_transmission", False)),
        "idempotency_control": engine_valid,
        "pretrade_risk_limits": engine_valid,
        "kill_switch": engine_valid,
        "reconciliation": engine_valid,
        "audit_log": engine_valid,
        "key_management_plan": True,
        "rollback_plan": True,
        "human_approval_required": True,
    }
    matrix["all_controls_pass"] = all(bool(matrix[k]) for k in REQUIRED_CONTROLS_V42)
    matrix["operational_controls_pass_ex_science"] = all(bool(matrix[k]) for k in REQUIRED_CONTROLS_V42 if k != "scientific_parent_valid" and k != "paper_path_ready")
    return matrix


def decision_v42(matrix: dict[str, Any]) -> dict[str, Any]:
    ready = bool(matrix.get("all_controls_pass", False))
    return {
        "version": "v0.42",
        "decision": "V42_SUPERVISED_PAPER_READINESS_CONFIRMED" if ready else "V42_NOT_READY_FOR_SUPERVISED_PAPER",
        "supervised_paper_ready": ready,
        "manual_human_approval_required": True,
        "real_money_trading_authorized": False,
        "live_execution_authorized": False,
        "real_order_transmission": False,
        "kraken_touched": False,
        "unresolved_scientific_block": not bool(matrix.get("scientific_parent_valid", False)),
    }


def operations_runbook_v42() -> str:
    return """# v0.42 Rollback and Key-Management Runbook

## Scope
This runbook is for supervised paper/shadow operation only. It does not authorize live capital or real order transmission.

## Key management
- Never commit API keys, secrets, passwords, session tokens, or private keys to GitHub.
- Use separate credentials for research/paper and any future production environment.
- Any future production key must have withdrawal permissions disabled.
- Apply least privilege, IP allow-listing where supported, and a managed secret store.
- Rotate credentials on a defined schedule and immediately after suspected exposure.
- Keep exchange-account authentication separate from model-training and CI credentials.
- Audit access to secrets and never print credentials in CI logs.

## Deployment controls
- Default state is `LIVE_EXECUTION=false`.
- Start only in shadow mode; then supervised paper mode after scientific gates pass.
- Require a named human approver for every future environment promotion.
- Use deterministic client order IDs and reject duplicate submissions.
- Reconcile local positions/orders with the paper broker after each cycle.
- Trigger the kill switch on reconciliation mismatch, daily-loss breach, stale data, missing prices, or unexpected exceptions.

## Rollback
1. Activate the kill switch and stop accepting new intents.
2. Disable the scheduler/worker producing order intents.
3. Snapshot audit logs, model/version identifiers, configuration and open paper positions.
4. Reconcile expected versus observed paper state.
5. Restore the last known-good immutable research artifact/configuration.
6. Run unit/integration smoke tests before resuming shadow mode.
7. Do not resume supervised paper mode until a human reviewer signs off.

## Real-money boundary
No result from v0.42, including a future PASS, is itself permission to trade real money. A separately reviewed deployment process, account-level controls and explicit human decision would still be required.
"""
