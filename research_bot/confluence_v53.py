from __future__ import annotations

from dataclasses import dataclass, asdict
import math

import pandas as pd


@dataclass(frozen=True)
class ConfluenceConfigV53:
    minimum_family_quorum: int = 3
    minimum_score: float = 0.67

    def __post_init__(self) -> None:
        if not 1 <= self.minimum_family_quorum <= 4:
            raise ValueError("minimum_family_quorum must be in [1,4]")
        if not 0.5 <= self.minimum_score <= 1.0:
            raise ValueError("minimum_score must be in [0.5,1]")


@dataclass(frozen=True)
class ConfluenceDecisionV53:
    action: str
    long_score: float
    short_score: float
    family_votes_long: int
    family_votes_short: int
    reasons: tuple[str, ...]
    execution_authorized: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def _truth(value) -> bool:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return False
    try:
        return float(value) >= 0.5
    except (TypeError, ValueError):
        return bool(value)


def _positive(value) -> bool:
    try:
        return math.isfinite(float(value)) and float(value) > 0
    except (TypeError, ValueError):
        return False


def _negative(value) -> bool:
    try:
        return math.isfinite(float(value)) and float(value) < 0
    except (TypeError, ValueError):
        return False


def decide_confluence_v53(row: pd.Series | dict, config: ConfluenceConfigV53 | None = None) -> ConfluenceDecisionV53:
    """Transparent research-only confluence decision.

    Four independent families vote: local ICT/SMC, local Brooks, local Ichimoku,
    and higher-timeframe agreement. The score is the fraction of positive
    checks inside each family, then averaged equally across families. There are
    no fitted weights and no execution authorization.
    """

    cfg = config or ConfluenceConfigV53()
    r = dict(row)

    long_families = {
        "SMC": [
            _positive(r.get("smc_structure_state")),
            _truth(r.get("smc_bos_bull")) or _truth(r.get("smc_choch_bull")),
            _truth(r.get("ict_sweep_bull")) or _truth(r.get("ict_bull_breaker_retest")),
        ],
        "BROOKS": [
            _positive(r.get("brooks_always_in")),
            _positive(r.get("brooks_market_trend")),
            float(r.get("brooks_bull_signal_quality", 0.0) or 0.0) >= 0.5,
        ],
        "ICHIMOKU": [
            _truth(r.get("ichi_tk_bullish")),
            _truth(r.get("ichi_price_above_visible_cloud")),
            _truth(r.get("ichi_projected_cloud_bullish")),
        ],
        "HTF": [
            _positive(r.get("4h_smc_structure_state")),
            _positive(r.get("4h_brooks_always_in")),
            _truth(r.get("4h_ichi_projected_cloud_bullish")),
        ],
    }
    short_families = {
        "SMC": [
            _negative(r.get("smc_structure_state")),
            _truth(r.get("smc_bos_bear")) or _truth(r.get("smc_choch_bear")),
            _truth(r.get("ict_sweep_bear")) or _truth(r.get("ict_bear_breaker_retest")),
        ],
        "BROOKS": [
            _negative(r.get("brooks_always_in")),
            _negative(r.get("brooks_market_trend")),
            float(r.get("brooks_bear_signal_quality", 0.0) or 0.0) >= 0.5,
        ],
        "ICHIMOKU": [
            not _truth(r.get("ichi_tk_bullish")),
            _truth(r.get("ichi_price_below_visible_cloud")),
            not _truth(r.get("ichi_projected_cloud_bullish")),
        ],
        "HTF": [
            _negative(r.get("4h_smc_structure_state")),
            _negative(r.get("4h_brooks_always_in")),
            r.get("4h_ichi_projected_cloud_bullish") is not None and not _truth(r.get("4h_ichi_projected_cloud_bullish")),
        ],
    }

    def score_family(checks: list[bool]) -> float:
        return sum(bool(v) for v in checks) / len(checks)

    long_scores = {name: score_family(checks) for name, checks in long_families.items()}
    short_scores = {name: score_family(checks) for name, checks in short_families.items()}
    long_score = sum(long_scores.values()) / len(long_scores)
    short_score = sum(short_scores.values()) / len(short_scores)
    long_votes = sum(v >= cfg.minimum_score for v in long_scores.values())
    short_votes = sum(v >= cfg.minimum_score for v in short_scores.values())

    reasons: list[str] = []
    for name, value in long_scores.items():
        if value >= cfg.minimum_score:
            reasons.append(f"LONG_{name}_{value:.2f}")
    for name, value in short_scores.items():
        if value >= cfg.minimum_score:
            reasons.append(f"SHORT_{name}_{value:.2f}")

    if long_votes >= cfg.minimum_family_quorum and long_score > short_score:
        action = "BUY_CANDIDATE"
    elif short_votes >= cfg.minimum_family_quorum and short_score > long_score:
        action = "SELL_CANDIDATE"
    else:
        action = "NO_TRADE"

    return ConfluenceDecisionV53(
        action=action,
        long_score=float(long_score),
        short_score=float(short_score),
        family_votes_long=int(long_votes),
        family_votes_short=int(short_votes),
        reasons=tuple(reasons),
        execution_authorized=False,
    )
