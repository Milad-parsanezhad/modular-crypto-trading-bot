from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable


@dataclass(frozen=True)
class EvidenceMilestone:
    version: str
    question: str
    evidence_class: str
    decision: str
    execution_authorized: bool
    thesis_use: str


MILESTONES: tuple[EvidenceMilestone, ...] = (
    EvidenceMilestone(
        version="v0.10",
        question="Do learned challengers beat simple baselines in purged live-universe OOS evaluation after costs?",
        evidence_class="PURGED_OOS",
        decision="NO_MODEL_PROMOTED",
        execution_authorized=False,
        thesis_use="Shows that aggregate ML performance can be unstable across folds and regimes and that cost-aware baselines remain essential.",
    ),
    EvidenceMilestone(
        version="v0.11",
        question="Does the provisional learned candidate survive multi-seed, paired moving-block bootstrap, FDR and regime diagnostics?",
        evidence_class="ROBUSTNESS_INFERENCE",
        decision="NO_MODEL_PROMOTED",
        execution_authorized=False,
        thesis_use="Shows that stronger inference can reverse a superficially encouraging model-selection result and reveals strong regime dependence.",
    ),
    EvidenceMilestone(
        version="v0.12",
        question="Do point-in-time funding, premium/basis proxy, order-flow and OI features add incremental alpha on an untouched holdout?",
        evidence_class="EXTERNAL_UNTOUCHED_HOLDOUT",
        decision="NO_INCREMENTAL_DERIVATIVES_EVIDENCE",
        execution_authorized=False,
        thesis_use="Negative external-holdout result: adding a theoretically strong feature family did not produce sufficient incremental tradable alpha.",
    ),
    EvidenceMilestone(
        version="v0.13",
        question="Can finer forward microstructure features be collected prospectively without reusing the spent v0.12 holdout?",
        evidence_class="PROSPECTIVE_COLLECTION",
        decision="FORWARD_COLLECTION_ACTIVE",
        execution_authorized=False,
        thesis_use="Preserves scientific independence by moving to a new source/window rather than tuning the consumed holdout.",
    ),
    EvidenceMilestone(
        version="v0.14",
        question="Can the system run end-to-end with live market observations, independent risk, paper execution and persistent state?",
        evidence_class="PRODUCTION_ENGINEERING_PAPER",
        decision="PAPER_RUNTIME_OPERATIONAL",
        execution_authorized=False,
        thesis_use="Engineering evidence only: demonstrates real runtime integration, not profitability or alpha.",
    ),
    EvidenceMilestone(
        version="v0.15",
        question="Can prospective paper evidence be captured automatically with immutable artifacts and pre-registered minimums?",
        evidence_class="PROSPECTIVE_FORWARD_EVIDENCE",
        decision="INSUFFICIENT_FORWARD_SAMPLE",
        execution_authorized=False,
        thesis_use="Creates a forward evidence stream with minimum 168 h, 100 observations and 10 simulated fills before formal review.",
    ),
    EvidenceMilestone(
        version="v0.16",
        question="Can forward evidence be aggregated, deduplicated and reported for thesis/defense without inflating sample size?",
        evidence_class="FORMAL_FORWARD_REPORTING",
        decision="INSUFFICIENT_FORWARD_SAMPLE",
        execution_authorized=False,
        thesis_use="Formal reporting layer; duplicate/manual snapshots are collapsed and Sharpe/Sortino remain withheld until numerical depth is adequate.",
    ),
)


CLAIM_POLICY = {
    "engineering_operational": True,
    "forward_paper_operational": True,
    "profitable_strategy": False,
    "statistically_significant_alpha": False,
    "stable_sharpe": False,
    "live_ready": False,
    "real_money_execution_authorized": False,
}


def ledger() -> list[dict]:
    return [asdict(x) for x in MILESTONES]


def validate_ledger(rows: Iterable[EvidenceMilestone] = MILESTONES) -> None:
    rows = tuple(rows)
    versions = [x.version for x in rows]
    if versions != [f"v0.{n}" for n in range(10, 17)]:
        raise AssertionError(f"unexpected milestone sequence: {versions}")
    if any(x.execution_authorized for x in rows):
        raise AssertionError("No v0.10-v0.16 milestone authorizes real execution")
    if CLAIM_POLICY["profitable_strategy"] or CLAIM_POLICY["statistically_significant_alpha"]:
        raise AssertionError("Current evidence does not authorize profitability/alpha claims")
    if CLAIM_POLICY["live_ready"] or CLAIM_POLICY["real_money_execution_authorized"]:
        raise AssertionError("LIVE must remain fail-closed")


def thesis_conclusion() -> str:
    validate_ledger()
    return (
        "The project demonstrates an operational, auditable research-and-paper-trading system, "
        "but the accumulated empirical evidence through v0.16 does not justify a claim of stable alpha, "
        "profitability, or real-money live readiness. Negative and inconclusive results are retained as "
        "first-class evidence and determine the next experiment rather than being tuned away."
    )
