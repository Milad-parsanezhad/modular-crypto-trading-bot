from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from math import isfinite
from statistics import mean
from typing import Any


V15_EVIDENCE_VERSION = "v0.15"
MIN_FORWARD_HOURS = 168.0
MIN_OBSERVATIONS = 100
MIN_FILLS = 10


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if isfinite(out) else default


def _parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _window_hours(observations: list[dict]) -> float:
    times = [_parse_dt(x.get("observed_at")) for x in observations]
    times = [x for x in times if x is not None]
    if len(times) < 2:
        return 0.0
    return max(0.0, (max(times) - min(times)).total_seconds() / 3600.0)


def build_forward_evidence(
    *,
    health: dict,
    research: dict,
    paper: dict,
    observations_payload: dict,
    fills_payload: dict,
    captured_at: datetime | None = None,
) -> dict:
    """Build a conservative production-paper evidence snapshot.

    This function deliberately does not infer profitability from a small forward
    sample. It summarizes observable runtime evidence and keeps LIVE promotion
    prohibited until a later, pre-registered statistical review.
    """
    captured_at = captured_at or datetime.now(timezone.utc)
    observations = list(observations_payload.get("items") or [])
    fills = list(fills_payload.get("items") or [])
    store = dict(paper.get("store") or {})
    account = dict(store.get("account") or {})

    actions = Counter(str(x.get("action") or "UNKNOWN") for x in observations)
    symbols = Counter(str(x.get("symbol") or "UNKNOWN") for x in observations)
    fill_sides = Counter(str(x.get("side") or "UNKNOWN") for x in fills)

    spreads = [_as_float(x.get("spread_bps"), float("nan")) for x in observations]
    spreads = [x for x in spreads if isfinite(x)]
    scores = [_as_float(x.get("rule_score"), float("nan")) for x in observations]
    scores = [x for x in scores if isfinite(x)]

    total_fill_notional = 0.0
    total_fees = 0.0
    for fill in fills:
        qty = abs(_as_float(fill.get("filled_quantity")))
        price = abs(_as_float(fill.get("fill_price")))
        total_fill_notional += qty * price
        total_fees += max(0.0, _as_float(fill.get("fee_paid")))

    total_observations = int(store.get("observations") or len(observations))
    total_fills = int(store.get("fills") or len(fills))
    equity_points = int(store.get("equity_points") or 0)
    window_hours = _window_hours(observations)

    safety_checks = {
        "health_ok": health.get("status") == "ok",
        "service_live_false": health.get("live_execution") is False,
        "paper_live_false": paper.get("live_execution") is False,
        "research_live_false": research.get("live_execution") is False,
        "execution_mode_paper": health.get("execution_mode") == "PAPER",
        "postgres_persistence": store.get("backend") == "postgres",
        "paper_running": paper.get("status") == "RUNNING",
        "paper_execution_enabled": paper.get("paper_execution_enabled") is True,
        "strategy_unvalidated_label_preserved": research.get("forward_paper_label") == "HYPOTHESIS_SHADOW_NOT_VALIDATED_ALPHA",
    }
    contract_ok = all(safety_checks.values())

    sample_gate = {
        "minimum_forward_hours": MIN_FORWARD_HOURS,
        "minimum_observations": MIN_OBSERVATIONS,
        "minimum_fills": MIN_FILLS,
        "observed_forward_hours_recent_window": window_hours,
        "observations_total": total_observations,
        "fills_total": total_fills,
        "hours_pass": window_hours >= MIN_FORWARD_HOURS,
        "observations_pass": total_observations >= MIN_OBSERVATIONS,
        "fills_pass": total_fills >= MIN_FILLS,
    }
    sample_gate["passed"] = bool(
        sample_gate["hours_pass"]
        and sample_gate["observations_pass"]
        and sample_gate["fills_pass"]
        and contract_ok
    )

    if not contract_ok:
        evidence_state = "SAFETY_CONTRACT_FAILURE"
    elif sample_gate["passed"]:
        evidence_state = "READY_FOR_FORMAL_STATISTICAL_EVALUATION"
    else:
        evidence_state = "INSUFFICIENT_FORWARD_SAMPLE"

    fee_bps_on_recent_notional = (
        total_fees / total_fill_notional * 10_000.0 if total_fill_notional > 0 else 0.0
    )

    return {
        "evidence_version": V15_EVIDENCE_VERSION,
        "captured_at": captured_at.astimezone(timezone.utc).isoformat(),
        "service": {
            "version": health.get("version"),
            "status": health.get("status"),
            "execution_mode": health.get("execution_mode"),
            "forward_paper_enabled": health.get("forward_paper_enabled"),
            "strategy_version": paper.get("strategy_version"),
            "backend": store.get("backend"),
        },
        "scientific_state": {
            "latest_completed_gate": research.get("latest_completed_research_gate"),
            "latest_research_decision": research.get("latest_research_decision"),
            "forward_paper_label": research.get("forward_paper_label"),
            "evidence_state": evidence_state,
            "live_promotion": "PROHIBITED",
            "reason": "Forward paper evidence is descriptive until the pre-registered minimum sample and a separate formal statistical review are completed.",
        },
        "paper_account": {
            "cash": _as_float(account.get("cash")),
            "equity": _as_float(account.get("equity")),
            "peak_equity": _as_float(account.get("peak_equity")),
            "cumulative_return": paper.get("cumulative_return"),
            "current_drawdown": paper.get("current_drawdown"),
            "position_count": len(store.get("positions") or []),
            "positions": store.get("positions") or [],
        },
        "forward_counts": {
            "observations_total": total_observations,
            "fills_total": total_fills,
            "equity_points_total": equity_points,
            "recent_observation_rows": len(observations),
            "recent_fill_rows": len(fills),
        },
        "recent_activity": {
            "observation_window_hours": window_hours,
            "symbols": dict(symbols),
            "actions": dict(actions),
            "fill_sides": dict(fill_sides),
            "mean_rule_score": mean(scores) if scores else None,
            "mean_spread_bps": mean(spreads) if spreads else None,
            "paper_fill_notional": total_fill_notional,
            "paper_fees_paid": total_fees,
            "fee_bps_on_recent_notional": fee_bps_on_recent_notional,
        },
        "safety_contract": {
            "ok": contract_ok,
            "checks": safety_checks,
        },
        "sample_gate": sample_gate,
        "warnings": [
            "Paper fills are simulated and are not real-money orders.",
            "A small forward sample cannot establish alpha, profitability, Sharpe stability, or live readiness.",
            "The v0.12 negative derivatives result remains part of the thesis evidence and must not be tuned away.",
        ],
    }


def evidence_markdown(evidence: dict) -> str:
    state = evidence["scientific_state"]
    svc = evidence["service"]
    acct = evidence["paper_account"]
    counts = evidence["forward_counts"]
    activity = evidence["recent_activity"]
    gate = evidence["sample_gate"]
    safety = evidence["safety_contract"]

    def fmt_pct(value: Any) -> str:
        if value is None:
            return "n/a"
        return f"{100.0 * _as_float(value):.4f}%"

    lines = [
        "# v0.15 Forward Evidence Snapshot",
        "",
        f"Captured: `{evidence['captured_at']}`",
        "",
        "## Production contract",
        "",
        f"- Service version: `{svc.get('version')}`",
        f"- Execution mode: `{svc.get('execution_mode')}`",
        f"- Strategy: `{svc.get('strategy_version')}`",
        f"- Persistence: `{svc.get('backend')}`",
        f"- Safety contract: `{'PASS' if safety.get('ok') else 'FAIL'}`",
        f"- LIVE promotion: `{state.get('live_promotion')}`",
        "",
        "## Paper account",
        "",
        f"- Equity: `{acct.get('equity'):.6f}`",
        f"- Cash: `{acct.get('cash'):.6f}`",
        f"- Cumulative return: `{fmt_pct(acct.get('cumulative_return'))}`",
        f"- Current drawdown: `{fmt_pct(acct.get('current_drawdown'))}`",
        f"- Open paper positions: `{acct.get('position_count')}`",
        "",
        "## Forward sample",
        "",
        f"- Observations: `{counts.get('observations_total')}`",
        f"- Paper fills: `{counts.get('fills_total')}`",
        f"- Equity points: `{counts.get('equity_points_total')}`",
        f"- Recent observation window: `{gate.get('observed_forward_hours_recent_window'):.2f} h`",
        f"- Recent paper fill notional: `{activity.get('paper_fill_notional'):.6f}`",
        f"- Recent paper fees: `{activity.get('paper_fees_paid'):.6f}`",
        "",
        "## Scientific gate",
        "",
        f"- Evidence state: `{state.get('evidence_state')}`",
        f"- Minimum window: `{gate.get('minimum_forward_hours')} h`",
        f"- Minimum observations: `{gate.get('minimum_observations')}`",
        f"- Minimum fills: `{gate.get('minimum_fills')}`",
        f"- Sample gate passed: `{gate.get('passed')}`",
        "",
        "No alpha, profitability, Sharpe stability, or live-readiness claim is authorized by this descriptive snapshot.",
        "",
    ]
    return "\n".join(lines)
