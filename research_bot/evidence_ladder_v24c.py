from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Iterable


class EvidenceState(str, Enum):
    NOT_RUN = "NOT_RUN"
    PASS = "PASS"
    FAIL = "FAIL"
    DATA_INSUFFICIENT = "DATA_INSUFFICIENT"
    BLOCKED = "BLOCKED"


class PlanName(str, Enum):
    EXTERNAL_UNTOUCHED = "PLAN_A_EXTERNAL_UNTOUCHED"
    FUTURE_TIME_FORWARD = "PLAN_B_FUTURE_TIME_FORWARD"
    PAPER_SHADOW_ACCUMULATION = "PLAN_C_PAPER_SHADOW_ACCUMULATION"
    RESEARCH_REDESIGN = "PLAN_D_RESEARCH_REDESIGN"


class MLTrack(str, Enum):
    TABULAR_META = "TABULAR_META"
    TEMPORAL_SEQUENCE = "TEMPORAL_SEQUENCE"
    VISION_MULTIMODAL = "VISION_MULTIMODAL"
    UNSUPERVISED_REGIME = "UNSUPERVISED_REGIME"
    PORTFOLIO_ALLOCATOR = "PORTFOLIO_ALLOCATOR"
    ENSEMBLE_STACK = "ENSEMBLE_STACK"
    REINFORCEMENT_LEARNING = "REINFORCEMENT_LEARNING"


@dataclass(frozen=True)
class ValidationGate:
    min_trades: int = 200
    min_profit_factor: float = 1.05
    min_mean_r: float = 0.0
    max_mtm_drawdown: float = 0.05
    min_positive_symbol_fraction: float = 0.60
    min_bootstrap_uplift_lower_bound: float = 0.0
    max_pbo: float = 0.50
    min_dsr_probability: float = 0.95
    require_36bps_nonnegative: bool = True
    require_external_or_forward: bool = True


@dataclass(frozen=True)
class RiskGate:
    risk_per_trade: float = 0.0025
    max_open_portfolio_risk: float = 0.01
    max_strategy_open_risk: float = 0.005
    max_directional_open_risk: float = 0.0075
    max_concurrent_positions: int = 5
    hard_mtm_drawdown_kill: float = 0.05
    cvar_alpha: float = 0.95
    max_cvar_loss_fraction: float = 0.02
    max_pairwise_correlation_for_full_size: float = 0.80


@dataclass(frozen=True)
class TrackContract:
    track: MLTrack
    requires: tuple[MLTrack, ...] = ()
    minimum_status: EvidenceState = EvidenceState.PASS
    promotable: bool = True
    notes: str = ""


TRACK_CONTRACTS: tuple[TrackContract, ...] = (
    TrackContract(
        MLTrack.TABULAR_META,
        notes="Strategy-aware supervised baselines; development fit, validation-only selection, untouched test once.",
    ),
    TrackContract(
        MLTrack.TEMPORAL_SEQUENCE,
        notes="LSTM/GRU/TCN/Transformer/Patch-style sequence models on causal tensors; multi-seed and frozen horizon.",
    ),
    TrackContract(
        MLTrack.VISION_MULTIMODAL,
        notes="Raw candle localization + numeric/temporal fusion; independent ablation required.",
    ),
    TrackContract(
        MLTrack.UNSUPERVISED_REGIME,
        promotable=False,
        notes="GMM/HMM/anomaly/regime discovery is context only and cannot create alpha by itself.",
    ),
    TrackContract(
        MLTrack.PORTFOLIO_ALLOCATOR,
        requires=(MLTrack.TABULAR_META,),
        notes="Allocator must optimize portfolio-level economics under overlap, correlation, CVaR and MTM risk.",
    ),
    TrackContract(
        MLTrack.ENSEMBLE_STACK,
        requires=(MLTrack.TABULAR_META, MLTrack.TEMPORAL_SEQUENCE, MLTrack.VISION_MULTIMODAL),
        notes="Stacking is allowed only after each input track shows independent validation value; no weak-model aggregation.",
    ),
    TrackContract(
        MLTrack.REINFORCEMENT_LEARNING,
        requires=(MLTrack.PORTFOLIO_ALLOCATOR,),
        notes="RL is restricted to frozen state representations and paper/testnet allocation/execution until external/forward evidence passes.",
    ),
)


@dataclass(frozen=True)
class PlanRule:
    plan: PlanName
    purpose: str
    admissible_after: tuple[EvidenceState, ...]
    uses_fresh_evidence: bool
    may_retune_previous_test: bool = False


PLAN_RULES: tuple[PlanRule, ...] = (
    PlanRule(
        PlanName.EXTERNAL_UNTOUCHED,
        "Primary scientific route: freeze hypothesis/model/threshold and evaluate on an untouched external venue/data source.",
        (EvidenceState.NOT_RUN,),
        True,
    ),
    PlanRule(
        PlanName.FUTURE_TIME_FORWARD,
        "If external venue coverage is unavailable or structurally incompatible, accumulate genuinely future bars after the freeze timestamp.",
        (EvidenceState.DATA_INSUFFICIENT, EvidenceState.BLOCKED),
        True,
    ),
    PlanRule(
        PlanName.PAPER_SHADOW_ACCUMULATION,
        "If sample size remains insufficient, collect sequential PAPER/shadow decisions without changing the frozen model.",
        (EvidenceState.DATA_INSUFFICIENT, EvidenceState.BLOCKED),
        True,
    ),
    PlanRule(
        PlanName.RESEARCH_REDESIGN,
        "If the hypothesis fails economically, redesign on development data only and require a new fresh validation/test cycle.",
        (EvidenceState.FAIL,),
        True,
    ),
)


@dataclass(frozen=True)
class EvidenceSnapshot:
    plan: PlanName
    state: EvidenceState
    test_seen: bool
    model_or_threshold_changed_after_test: bool
    source_is_fresh: bool
    forward_paper_authorized: bool = False
    live_execution_authorized: bool = False


def assert_evidence_integrity(snapshot: EvidenceSnapshot) -> None:
    if snapshot.model_or_threshold_changed_after_test:
        raise ValueError("POST_TEST_RETUNING_FORBIDDEN")
    if snapshot.state == EvidenceState.PASS and not snapshot.source_is_fresh:
        raise ValueError("PASS_REQUIRES_FRESH_EVIDENCE")
    if snapshot.live_execution_authorized:
        raise ValueError("V24C_LIVE_MUST_REMAIN_FAIL_CLOSED")
    if snapshot.forward_paper_authorized and snapshot.state != EvidenceState.PASS:
        raise ValueError("FORWARD_PAPER_REQUIRES_PASS")


def next_plan(current: EvidenceSnapshot) -> PlanName | None:
    """Return the next pre-registered route without cherry-picking the same test.

    Important: a scientific FAIL does not route to another model on the same terminal
    test. It routes to RESEARCH_REDESIGN, which must start a new development/validation
    cycle and later consume fresh evidence.
    """
    assert_evidence_integrity(current)
    if current.state == EvidenceState.PASS:
        return None
    if current.plan == PlanName.EXTERNAL_UNTOUCHED:
        if current.state in (EvidenceState.DATA_INSUFFICIENT, EvidenceState.BLOCKED):
            return PlanName.FUTURE_TIME_FORWARD
        if current.state == EvidenceState.FAIL:
            return PlanName.RESEARCH_REDESIGN
    if current.plan == PlanName.FUTURE_TIME_FORWARD:
        if current.state in (EvidenceState.DATA_INSUFFICIENT, EvidenceState.BLOCKED):
            return PlanName.PAPER_SHADOW_ACCUMULATION
        if current.state == EvidenceState.FAIL:
            return PlanName.RESEARCH_REDESIGN
    if current.plan == PlanName.PAPER_SHADOW_ACCUMULATION:
        if current.state == EvidenceState.FAIL:
            return PlanName.RESEARCH_REDESIGN
        if current.state in (EvidenceState.DATA_INSUFFICIENT, EvidenceState.BLOCKED):
            return None
    if current.plan == PlanName.RESEARCH_REDESIGN:
        return None
    return None


def eligible_tracks(status: dict[MLTrack, EvidenceState]) -> list[MLTrack]:
    out: list[MLTrack] = []
    for contract in TRACK_CONTRACTS:
        if contract.track == MLTrack.REINFORCEMENT_LEARNING:
            # RL is additionally held back until portfolio allocator evidence is PASS.
            pass
        if all(status.get(req) == contract.minimum_status for req in contract.requires):
            out.append(contract.track)
    return out


def promotion_ready(
    *,
    validation: dict[str, float | int | bool],
    gate: ValidationGate | None = None,
) -> tuple[bool, list[str]]:
    g = gate or ValidationGate()
    failures: list[str] = []
    if int(validation.get("trades", 0)) < g.min_trades:
        failures.append("MIN_TRADES")
    if float(validation.get("profit_factor", float("-inf"))) < g.min_profit_factor:
        failures.append("PROFIT_FACTOR")
    if float(validation.get("mean_r", float("-inf"))) <= g.min_mean_r:
        failures.append("MEAN_R")
    if abs(float(validation.get("mtm_drawdown", float("inf")))) > g.max_mtm_drawdown:
        failures.append("MTM_DRAWDOWN")
    if float(validation.get("positive_symbol_fraction", 0.0)) < g.min_positive_symbol_fraction:
        failures.append("SYMBOL_BREADTH")
    if float(validation.get("bootstrap_uplift_low", float("-inf"))) <= g.min_bootstrap_uplift_lower_bound:
        failures.append("BOOTSTRAP_UPLIFT")
    if float(validation.get("pbo", float("inf"))) > g.max_pbo:
        failures.append("PBO")
    if float(validation.get("dsr_probability", 0.0)) < g.min_dsr_probability:
        failures.append("DSR")
    if g.require_36bps_nonnegative and not bool(validation.get("stress_36bps_pass", False)):
        failures.append("COST_STRESS_36BPS")
    if g.require_external_or_forward and not bool(validation.get("fresh_external_or_forward", False)):
        failures.append("FRESH_EVIDENCE")
    return (len(failures) == 0, failures)


def protocol_manifest() -> dict:
    return {
        "version": "v0.24c",
        "principle": "multiple pre-registered plans, no same-test rescue tuning",
        "validation_gate": asdict(ValidationGate()),
        "risk_gate": asdict(RiskGate()),
        "plans": [asdict(x) for x in PLAN_RULES],
        "tracks": [
            {
                "track": x.track.value,
                "requires": [r.value for r in x.requires],
                "promotable": x.promotable,
                "notes": x.notes,
            }
            for x in TRACK_CONTRACTS
        ],
        "forward_paper_authorized": False,
        "live_execution_authorized": False,
    }
