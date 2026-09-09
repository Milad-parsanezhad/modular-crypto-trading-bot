from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from math import isfinite, sqrt
from statistics import mean, median, pstdev
from typing import Any, Iterable


V16_VERSION = "v0.16"
MIN_RATIO_POINTS = 20
MIN_INDEPENDENT_SPACING_HOURS = 3.5


def _f(value: Any, default: float | None = None) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if isfinite(out) else default


def _dt(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        out = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if out.tzinfo is None:
        out = out.replace(tzinfo=timezone.utc)
    return out.astimezone(timezone.utc)


def _dedupe_sort(snapshots: Iterable[dict]) -> list[dict]:
    by_time: dict[str, dict] = {}
    for raw in snapshots:
        if not isinstance(raw, dict):
            continue
        captured = _dt(raw.get("captured_at"))
        if captured is None:
            continue
        by_time[captured.isoformat()] = raw
    return [by_time[k] for k in sorted(by_time)]


def _snapshot_signature(snapshot: dict) -> tuple:
    counts = snapshot.get("forward_counts") or {}
    account = snapshot.get("paper_account") or {}
    gate = snapshot.get("sample_gate") or {}
    return (
        int(counts.get("observations_total") or 0),
        int(counts.get("fills_total") or 0),
        int(counts.get("equity_points_total") or 0),
        round(float(_f(account.get("equity"), 0.0) or 0.0), 8),
        round(float(_f(account.get("current_drawdown"), 0.0) or 0.0), 10),
        len(account.get("positions") or []),
        bool(gate.get("passed")),
        round(float(_f(gate.get("observed_forward_hours_recent_window"), 0.0) or 0.0), 6),
        bool((snapshot.get("safety_contract") or {}).get("ok")),
    )


def _collapse_nonindependent(rows: list[dict]) -> list[dict]:
    """Collapse rapid duplicate/manual snapshots that add no new market evidence.

    A snapshot is retained if the observable paper state changes, or if at least
    3.5 hours have elapsed since the last retained snapshot. This prevents rapid
    CI/manual reruns from inflating the time-series sample while preserving the
    scheduled four-hour evidence cadence.
    """
    if not rows:
        return []
    kept = [rows[0]]
    previous_signature = _snapshot_signature(rows[0])
    previous_time = _dt(rows[0].get("captured_at"))
    for row in rows[1:]:
        current_time = _dt(row.get("captured_at"))
        signature = _snapshot_signature(row)
        elapsed = (
            (current_time - previous_time).total_seconds() / 3600.0
            if current_time is not None and previous_time is not None and current_time > previous_time
            else 0.0
        )
        if signature != previous_signature or elapsed >= MIN_INDEPENDENT_SPACING_HOURS:
            kept.append(row)
            previous_signature = signature
            previous_time = current_time
    return kept


def _equity_series(snapshots: list[dict]) -> list[tuple[datetime, float]]:
    rows: list[tuple[datetime, float]] = []
    for s in snapshots:
        ts = _dt(s.get("captured_at"))
        eq = _f((s.get("paper_account") or {}).get("equity"))
        if ts is not None and eq is not None and eq > 0:
            rows.append((ts, eq))
    return rows


def _returns(values: list[float]) -> list[float]:
    return [b / a - 1.0 for a, b in zip(values[:-1], values[1:]) if a > 0]


def _annualization(times: list[datetime]) -> float | None:
    if len(times) < 3:
        return None
    hours = [(b - a).total_seconds() / 3600.0 for a, b in zip(times[:-1], times[1:]) if b > a]
    if not hours:
        return None
    step = median(hours)
    return 8760.0 / step if step > 0 else None


def _risk_ratios(values: list[float], times: list[datetime]) -> dict[str, float | None]:
    rets = _returns(values)
    annual = _annualization(times)
    if len(rets) < MIN_RATIO_POINTS or annual is None:
        return {"period_returns": len(rets), "annualization_factor": None, "sharpe": None, "sortino": None}
    mu = mean(rets)
    sigma = pstdev(rets)
    downside = [min(0.0, r) for r in rets]
    downside_dev = sqrt(mean([x * x for x in downside])) if downside else 0.0
    return {
        "period_returns": len(rets),
        "annualization_factor": annual,
        "sharpe": (mu / sigma * sqrt(annual)) if sigma > 0 else None,
        "sortino": (mu / downside_dev * sqrt(annual)) if downside_dev > 0 else None,
    }


def _latest_total(snapshot: dict, key: str) -> int:
    return int((snapshot.get("forward_counts") or {}).get(key) or 0)


def build_formal_forward_evaluation(snapshots: Iterable[dict], *, generated_at: datetime | None = None) -> dict:
    """Aggregate prospective v0.15 snapshots without weakening their scientific gate."""
    generated_at = generated_at or datetime.now(timezone.utc)
    raw_rows = _dedupe_sort(snapshots)
    if not raw_rows:
        raise ValueError("at least one valid v0.15 evidence snapshot is required")
    rows = _collapse_nonindependent(raw_rows)

    latest = raw_rows[-1]
    first_independent = rows[0]
    last_independent = rows[-1]
    series = _equity_series(rows)
    times = [x[0] for x in series]
    equities = [x[1] for x in series]

    safety_passes = [bool((s.get("safety_contract") or {}).get("ok")) for s in raw_rows]
    modes = [str((s.get("service") or {}).get("execution_mode") or "UNKNOWN") for s in raw_rows]
    promotions = [str((s.get("scientific_state") or {}).get("live_promotion") or "UNKNOWN") for s in raw_rows]
    safety_all = bool(safety_passes) and all(safety_passes) and all(x == "PAPER" for x in modes) and all(x == "PROHIBITED" for x in promotions)

    latest_gate = dict(latest.get("sample_gate") or {})
    preregistered_gate_passed = bool(latest_gate.get("passed")) and safety_all
    ratios = _risk_ratios(equities, times)
    ratio_authorized = preregistered_gate_passed and int(ratios["period_returns"] or 0) >= MIN_RATIO_POINTS

    latest_return = _f((latest.get("paper_account") or {}).get("cumulative_return"))
    latest_dd = _f((latest.get("paper_account") or {}).get("current_drawdown"))
    observed_dds = [x for x in (_f((s.get("paper_account") or {}).get("current_drawdown")) for s in raw_rows) if x is not None]

    first_ts = _dt(first_independent.get("captured_at"))
    last_ts = _dt(last_independent.get("captured_at"))
    snapshot_window_hours = max(0.0, (last_ts - first_ts).total_seconds() / 3600.0) if first_ts and last_ts else 0.0

    latest_positions = list((latest.get("paper_account") or {}).get("positions") or [])
    positions_by_symbol = {
        str(x.get("symbol")): {"quantity": _f(x.get("quantity"), 0.0), "avg_price": _f(x.get("avg_price"), 0.0)}
        for x in latest_positions if isinstance(x, dict) and x.get("symbol")
    }
    evidence_states = Counter(str((s.get("scientific_state") or {}).get("evidence_state") or "UNKNOWN") for s in raw_rows)

    if not safety_all:
        formal_state = "SAFETY_CONTRACT_FAILURE"
    elif not preregistered_gate_passed:
        formal_state = "INSUFFICIENT_FORWARD_SAMPLE"
    elif not ratio_authorized:
        formal_state = "SAMPLE_GATE_PASSED_AWAITING_METRIC_DEPTH"
    else:
        formal_state = "READY_FOR_FORMAL_FORWARD_REVIEW"

    metric_policy = {
        "descriptive_metrics_authorized": safety_all,
        "risk_ratios_authorized": ratio_authorized,
        "alpha_claim_authorized": False,
        "profitability_claim_authorized": False,
        "live_promotion_authorized": False,
        "note": "Passing a sample-size gate permits formal review; it does not by itself establish alpha or authorize live trading.",
    }

    timeline_rows = [x for x in rows if (_f((x.get("paper_account") or {}).get("equity")) or 0.0) > 0]
    timeline = []
    for s in timeline_rows:
        ts = _dt(s.get("captured_at"))
        eq = _f((s.get("paper_account") or {}).get("equity"))
        if ts is None or eq is None:
            continue
        timeline.append({
            "captured_at": ts.isoformat(),
            "equity": eq,
            "observations_total": _latest_total(s, "observations_total"),
            "fills_total": _latest_total(s, "fills_total"),
            "equity_points_total": _latest_total(s, "equity_points_total"),
            "cumulative_return": _f((s.get("paper_account") or {}).get("cumulative_return")),
            "current_drawdown": _f((s.get("paper_account") or {}).get("current_drawdown")),
        })

    return {
        "evaluation_version": V16_VERSION,
        "generated_at": generated_at.astimezone(timezone.utc).isoformat(),
        "source_contract": {
            "source_evidence_version": latest.get("evidence_version"),
            "raw_snapshot_count": len(raw_rows),
            "snapshot_count": len(rows),
            "independence_spacing_hours": MIN_INDEPENDENT_SPACING_HOURS,
            "first_snapshot": first_ts.isoformat() if first_ts else None,
            "last_snapshot": last_ts.isoformat() if last_ts else None,
            "snapshot_window_hours": snapshot_window_hours,
            "safety_all_snapshots": safety_all,
            "execution_modes": dict(Counter(modes)),
            "evidence_states": dict(evidence_states),
        },
        "sample_progress": {
            "observations_total": _latest_total(latest, "observations_total"),
            "fills_total": _latest_total(latest, "fills_total"),
            "equity_points_total": _latest_total(latest, "equity_points_total"),
            "observed_forward_hours": _f(latest_gate.get("observed_forward_hours_recent_window"), 0.0),
            "minimum_forward_hours": _f(latest_gate.get("minimum_forward_hours"), 0.0),
            "minimum_observations": int(latest_gate.get("minimum_observations") or 0),
            "minimum_fills": int(latest_gate.get("minimum_fills") or 0),
            "preregistered_gate_passed": preregistered_gate_passed,
        },
        "paper_performance": {
            "latest_equity": _f((latest.get("paper_account") or {}).get("equity")),
            "latest_cumulative_return": latest_return,
            "latest_current_drawdown": latest_dd,
            "worst_observed_current_drawdown": min(observed_dds) if observed_dds else None,
            "snapshot_equity_points": len(equities),
            "period_returns": ratios["period_returns"],
            "annualization_factor": ratios["annualization_factor"] if ratio_authorized else None,
            "sharpe": ratios["sharpe"] if ratio_authorized else None,
            "sortino": ratios["sortino"] if ratio_authorized else None,
        },
        "portfolio_snapshot": {"position_count": len(latest_positions), "positions_by_symbol": positions_by_symbol},
        "formal_state": formal_state,
        "metric_policy": metric_policy,
        "live_promotion": "PROHIBITED",
        "timeline": timeline,
        "warnings": [
            "All fills are simulated paper fills; no real-money execution is represented here.",
            "The current strategy remains a shadow hypothesis, not validated alpha.",
            "Rapid unchanged CI/manual snapshots are collapsed so they cannot inflate the formal time-series sample.",
            "v0.12 negative external-holdout evidence remains part of the thesis and is not overwritten by forward paper observations.",
        ],
    }


def chapter4_markdown(evaluation: dict) -> str:
    p = evaluation["paper_performance"]
    s = evaluation["sample_progress"]
    c = evaluation["source_contract"]
    policy = evaluation["metric_policy"]

    def pct(x: Any) -> str:
        v = _f(x)
        return "n/a" if v is None else f"{100.0 * v:.4f}%"

    def num(x: Any, digits: int = 4) -> str:
        v = _f(x)
        return "n/a" if v is None else f"{v:.{digits}f}"

    return "\n".join([
        "# v0.16 Forward Paper Evaluation — Chapter 4 Evidence",
        "",
        "## Scientific status",
        "",
        f"- Formal state: `{evaluation['formal_state']}`",
        f"- LIVE promotion: `{evaluation['live_promotion']}`",
        f"- Safety across all collected snapshots: `{c['safety_all_snapshots']}`",
        f"- Alpha claim authorized: `{policy['alpha_claim_authorized']}`",
        "",
        "## Forward sample progress",
        "",
        f"- Raw evidence snapshots: `{c['raw_snapshot_count']}`",
        f"- Independent evidence snapshots: `{c['snapshot_count']}`",
        f"- Independent snapshot window: `{c['snapshot_window_hours']:.2f} h`",
        f"- Production observations: `{s['observations_total']}` / `{s['minimum_observations']}` minimum",
        f"- Simulated fills: `{s['fills_total']}` / `{s['minimum_fills']}` minimum",
        f"- Forward observation window: `{s['observed_forward_hours']:.2f} h` / `{s['minimum_forward_hours']:.2f} h` minimum",
        f"- Pre-registered sample gate passed: `{s['preregistered_gate_passed']}`",
        "",
        "## Paper performance",
        "",
        f"- Latest equity: `{num(p['latest_equity'], 6)}`",
        f"- Cumulative return: `{pct(p['latest_cumulative_return'])}`",
        f"- Current drawdown: `{pct(p['latest_current_drawdown'])}`",
        f"- Worst observed drawdown in evidence snapshots: `{pct(p['worst_observed_current_drawdown'])}`",
        f"- Sharpe: `{num(p['sharpe']) if policy['risk_ratios_authorized'] else 'WITHHELD_UNTIL_GATE'}`",
        f"- Sortino: `{num(p['sortino']) if policy['risk_ratios_authorized'] else 'WITHHELD_UNTIL_GATE'}`",
        "",
        "## Interpretation",
        "",
        "The forward-paper system is an operational evidence generator. Descriptive engineering results may be reported immediately, but profitability, alpha, stable Sharpe, or live-readiness conclusions remain prohibited until the pre-registered forward sample is complete and separately reviewed.",
        "",
    ])
