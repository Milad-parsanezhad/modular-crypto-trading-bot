from __future__ import annotations

from collections import Counter
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from research_bot.portfolio_mtm_v24c import (
    MTMRiskContract,
    _historical_cvar,
    _normalize_frame,
    _position_mark_pnl,
    _rolling_correlation,
)


def simulate_ranked_mtm_portfolio(
    events: pd.DataFrame,
    frames: Mapping[str, pd.DataFrame],
    priority: Iterable[float],
    *,
    selected: Iterable[bool] | None = None,
    contract: MTMRiskContract | None = None,
    mode: str = "ranked",
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Priority-aware extension of the v0.24c MTM portfolio simulator.

    The v0.24d post-hoc failure analysis identified deterministic alphabetical
    admission as a potential allocator defect when multiple already-eligible
    events compete for limited risk. This simulator changes only the tie-breaking
    order: events at the same entry timestamp are considered by descending frozen
    priority, then deterministic strategy/symbol keys. Risk, correlation, CVaR,
    MTM and hard-kill semantics remain inherited from v0.24c.
    """
    c = contract or MTMRiskContract()
    if events.empty:
        return {
            "mode": mode, "events": 0, "accepted": 0, "total_realized_return": 0.0,
            "max_mtm_drawdown": 0.0, "max_intrabar_stress_drawdown": 0.0,
            "profit_factor": np.nan, "hard_mtm_kill_triggered": False,
        }, pd.DataFrame(), pd.DataFrame()

    norm_frames = {s: _normalize_frame(f) for s, f in frames.items()}
    x = events.copy().reset_index(drop=True)
    for col in ["signal_time", "entry_time", "exit_time"]:
        x[col] = pd.to_datetime(x[col], utc=True)
    for col in ["entry", "stop", "r_multiple"]:
        x[col] = pd.to_numeric(x[col], errors="coerce")
    pr = np.asarray(list(priority), dtype=float)
    if len(pr) != len(x):
        raise ValueError("priority length mismatch")
    mask = np.ones(len(x), dtype=bool) if selected is None else np.asarray(list(selected), dtype=bool)
    if len(mask) != len(x):
        raise ValueError("selection mask length mismatch")
    x["_priority"] = pr
    x["_selected"] = mask
    x = x.sort_values(
        ["entry_time", "_priority", "strategy", "symbol"],
        ascending=[True, False, True, True], kind="mergesort",
    ).reset_index(drop=True)

    min_t = x["entry_time"].min(); max_t = x["exit_time"].max()
    timeline = sorted({t for f in norm_frames.values() for t in f.loc[(f["timestamp"] >= min_t) & (f["timestamp"] <= max_t), "timestamp"]})
    if not timeline:
        raise ValueError("no MTM timeline overlaps candidate events")
    bar_lookup = {s: f.set_index("timestamp") for s, f in norm_frames.items()}
    entries: dict[pd.Timestamp, list[pd.Series]] = {}
    for _, row in x.iterrows():
        entries.setdefault(row["entry_time"], []).append(row)

    realized_equity = 1.0; close_peak = 1.0; stress_peak = 1.0
    min_close_dd = 0.0; min_stress_dd = 0.0
    active: list[dict] = []; ledger: list[dict] = []; curve: list[dict] = []
    realized_pnl: list[float] = []; accepted_r: list[float] = []; close_returns: list[float] = []
    previous_close_equity: float | None = None
    reject_counts: Counter[str] = Counter(); hard_kill = False
    corr_scaled_entries = 0; max_concurrent = 0; max_open_risk_fraction = 0.0; max_cvar = np.nan

    def settle_due(now: pd.Timestamp) -> None:
        nonlocal realized_equity, active
        due = [p for p in active if p["exit_time"] < now]
        for p in sorted(due, key=lambda q: (q["exit_time"], q["strategy"], q["symbol"])):
            pnl = float(p["risk_amount"] * p["r_multiple"])
            realized_equity += pnl; realized_pnl.append(pnl)
        active = [p for p in active if p not in due]

    for now in timeline:
        settle_due(now)
        rolling_cvar = _historical_cvar(close_returns[-c.cvar_lookback_bars :], c.cvar_alpha)
        if np.isfinite(rolling_cvar):
            max_cvar = float(rolling_cvar) if not np.isfinite(max_cvar) else max(max_cvar, float(rolling_cvar))

        for row in entries.get(now, []):
            reason = ""; risk_scale = 1.0; max_abs_corr = np.nan
            if not bool(row["_selected"]):
                reason = "MODEL_FILTER_REJECT"
            elif hard_kill:
                reason = "HARD_MTM_DRAWDOWN_KILL"
            elif np.isfinite(rolling_cvar) and rolling_cvar > c.max_cvar_loss_fraction:
                reason = "CVaR_BUDGET"
            elif any(p["symbol"] == str(row["symbol"]) for p in active):
                reason = "SYMBOL_OVERLAP_CAP"
            elif len(active) >= c.max_concurrent_positions:
                reason = "MAX_CONCURRENT_POSITIONS"
            else:
                correlations = [abs(_rolling_correlation(norm_frames, str(row["symbol"]), p["symbol"], now, c.correlation_lookback_bars)) for p in active]
                correlations = [v for v in correlations if np.isfinite(v)]
                max_abs_corr = max(correlations) if correlations else np.nan
                if np.isfinite(max_abs_corr) and max_abs_corr > c.max_pairwise_correlation_for_full_size:
                    risk_scale = c.correlated_risk_multiplier; corr_scaled_entries += 1
                planned_risk = realized_equity * c.risk_per_trade * risk_scale
                current_risk = sum(float(p["risk_amount"]) for p in active)
                strategy_risk = sum(float(p["risk_amount"]) for p in active if p["strategy"] == str(row["strategy"]))
                side = str(row["side"]).lower()
                directional_risk = sum(float(p["risk_amount"]) for p in active if p["side"] == side)
                if (current_risk + planned_risk) / max(realized_equity, 1e-12) > c.max_open_portfolio_risk + 1e-12:
                    reason = "PORTFOLIO_OPEN_RISK_CAP"
                elif (strategy_risk + planned_risk) / max(realized_equity, 1e-12) > c.max_strategy_open_risk + 1e-12:
                    reason = "STRATEGY_OPEN_RISK_CAP"
                elif (directional_risk + planned_risk) / max(realized_equity, 1e-12) > c.max_directional_open_risk + 1e-12:
                    reason = "DIRECTIONAL_OPEN_RISK_CAP"
                else:
                    stop_fraction = abs(float(row["entry"] - row["stop"])) / max(abs(float(row["entry"])), 1e-12)
                    if not np.isfinite(stop_fraction) or stop_fraction <= 0:
                        reason = "INVALID_STOP_DISTANCE"
                    else:
                        notional = planned_risk / stop_fraction
                        active.append({
                            "strategy": str(row["strategy"]), "symbol": str(row["symbol"]), "side": side,
                            "entry_time": row["entry_time"], "exit_time": row["exit_time"],
                            "entry": float(row["entry"]), "stop": float(row["stop"]),
                            "risk_amount": float(planned_risk), "risk_scale": float(risk_scale),
                            "notional": float(notional), "r_multiple": float(row["r_multiple"]),
                        })
                        accepted_r.append(float(row["r_multiple"])); max_concurrent = max(max_concurrent, len(active))
                        max_open_risk_fraction = max(max_open_risk_fraction, sum(float(p["risk_amount"]) for p in active) / max(realized_equity, 1e-12))
            if reason:
                reject_counts[reason] += 1
            ledger.append({
                "entry_time": row["entry_time"], "exit_time": row["exit_time"], "strategy": row["strategy"],
                "symbol": row["symbol"], "side": row["side"], "priority": float(row["_priority"]),
                "selected_input": bool(row["_selected"]), "accepted": reason == "", "rejection_reason": reason,
                "risk_scale": float(risk_scale), "max_abs_corr_at_entry": float(max_abs_corr) if np.isfinite(max_abs_corr) else np.nan,
            })

        close_unrealized = 0.0; stress_unrealized = 0.0
        for p in active:
            table = bar_lookup.get(p["symbol"])
            if table is None or now not in table.index:
                continue
            bar = table.loc[now]
            if isinstance(bar, pd.DataFrame): bar = bar.iloc[-1]
            close_unrealized += _position_mark_pnl(p, bar, "close", c)
            stress_unrealized += _position_mark_pnl(p, bar, "low" if p["side"] == "long" else "high", c)
        close_equity = realized_equity + close_unrealized; stress_equity = realized_equity + stress_unrealized
        close_peak = max(close_peak, close_equity); stress_peak = max(stress_peak, close_equity)
        close_dd = close_equity / max(close_peak, 1e-12) - 1.0; stress_dd = stress_equity / max(stress_peak, 1e-12) - 1.0
        min_close_dd = min(min_close_dd, close_dd); min_stress_dd = min(min_stress_dd, stress_dd)
        if stress_dd <= -c.hard_mtm_drawdown_kill: hard_kill = True
        if previous_close_equity is not None and previous_close_equity > 0:
            close_returns.append(close_equity / previous_close_equity - 1.0)
        previous_close_equity = close_equity
        curve.append({
            "timestamp": now, "realized_equity": realized_equity, "close_mtm_equity": close_equity,
            "intrabar_stress_equity": stress_equity, "close_mtm_drawdown": close_dd,
            "intrabar_stress_drawdown": stress_dd, "rolling_cvar_95": rolling_cvar,
            "active_positions": len(active), "open_risk_fraction": sum(float(p["risk_amount"]) for p in active) / max(realized_equity, 1e-12),
        })
        due_now = [p for p in active if p["exit_time"] == now]
        for p in sorted(due_now, key=lambda q: (q["strategy"], q["symbol"])):
            pnl = float(p["risk_amount"] * p["r_multiple"]); realized_equity += pnl; realized_pnl.append(pnl)
        active = [p for p in active if p not in due_now]

    for p in active:
        pnl = float(p["risk_amount"] * p["r_multiple"]); realized_equity += pnl; realized_pnl.append(pnl)
    wins = float(sum(v for v in realized_pnl if v > 0)); losses = float(-sum(v for v in realized_pnl if v < 0))
    pf = wins / losses if losses > 0 else (np.inf if wins > 0 else np.nan)
    summary = {
        "mode": mode, "events": int(len(x)), "selected_input": int(x["_selected"].sum()),
        "accepted": int(sum(1 for r in ledger if r["accepted"])),
        "total_realized_return": float(realized_equity - 1.0), "ending_realized_equity": float(realized_equity),
        "max_mtm_drawdown": float(min_close_dd), "max_intrabar_stress_drawdown": float(min_stress_dd),
        "max_rolling_cvar": float(max_cvar) if np.isfinite(max_cvar) else np.nan,
        "profit_factor": float(pf), "mean_r_accepted": float(np.mean(accepted_r)) if accepted_r else np.nan,
        "max_concurrent": int(max_concurrent), "max_open_risk_fraction_seen": float(max_open_risk_fraction),
        "correlation_scaled_entries": int(corr_scaled_entries), "hard_mtm_kill_triggered": bool(hard_kill),
        "reject_counts": dict(reject_counts), "priority_aware": True,
    }
    return summary, pd.DataFrame(ledger), pd.DataFrame(curve)
