from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MTMRiskContract:
    risk_per_trade: float = 0.0025
    max_open_portfolio_risk: float = 0.0100
    max_strategy_open_risk: float = 0.0050
    max_directional_open_risk: float = 0.0075
    max_concurrent_positions: int = 5
    hard_mtm_drawdown_kill: float = 0.05
    cvar_alpha: float = 0.95
    max_cvar_loss_fraction: float = 0.02
    cvar_lookback_bars: int = 90
    correlation_lookback_bars: int = 90
    max_pairwise_correlation_for_full_size: float = 0.80
    correlated_risk_multiplier: float = 0.50
    liquidation_roundtrip_bps: float = 24.0

    def to_dict(self) -> dict:
        return asdict(self)


def _normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", "open", "high", "low", "close"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing MTM frame columns: {sorted(missing)}")
    x = frame.copy()
    x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True)
    for c in ["open", "high", "low", "close"]:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.dropna(subset=["timestamp", "open", "high", "low", "close"])
    return x.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)


def _rolling_correlation(
    frames: Mapping[str, pd.DataFrame],
    left: str,
    right: str,
    before: pd.Timestamp,
    lookback: int,
) -> float:
    if left == right:
        return 1.0
    if left not in frames or right not in frames:
        return np.nan
    a = frames[left][["timestamp", "close"]].copy()
    b = frames[right][["timestamp", "close"]].copy()
    a = a[a["timestamp"] < before].tail(lookback + 1)
    b = b[b["timestamp"] < before].tail(lookback + 1)
    m = a.merge(b, on="timestamp", suffixes=("_a", "_b"))
    if len(m) < max(20, lookback // 3):
        return np.nan
    ra = np.log(m["close_a"]).diff().dropna()
    rb = np.log(m["close_b"]).diff().dropna()
    if len(ra) < 10 or ra.std(ddof=1) <= 0 or rb.std(ddof=1) <= 0:
        return np.nan
    return float(ra.corr(rb))


def _historical_cvar(returns: Iterable[float], alpha: float) -> float:
    x = np.asarray(list(returns), dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 20:
        return np.nan
    losses = -x
    var = float(np.quantile(losses, alpha))
    tail = losses[losses >= var]
    return float(tail.mean()) if len(tail) else var


def _position_mark_pnl(position: dict, bar: pd.Series, price_field: str, contract: MTMRiskContract) -> float:
    px = float(bar[price_field])
    entry = float(position["entry"])
    side = 1.0 if str(position["side"]).lower() == "long" else -1.0
    raw_return = side * (px / entry - 1.0)
    # Conservative liquidation mark: reserve the full round-trip friction while
    # the position is open instead of assuming a free exit.
    net_return = raw_return - contract.liquidation_roundtrip_bps / 10_000.0
    return float(position["notional"] * net_return)


def simulate_mtm_portfolio(
    events: pd.DataFrame,
    frames: Mapping[str, pd.DataFrame],
    selected: Iterable[bool] | None = None,
    *,
    contract: MTMRiskContract | None = None,
    mode: str = "base",
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Overlap-aware point-in-time MTM portfolio replay.

    Positions are sized from stop distance so the frozen risk budget corresponds
    to the loss at the recorded stop. New entries are constrained by symbol,
    strategy, direction, total-open-risk, rolling cross-asset correlation and a
    historical 95% CVaR budget. Open positions are revalued at each observed 4h
    close and at a conservative intrabar adverse mark (low for long, high for
    short). Outcome fields are used only for closing already-accepted positions.
    """
    c = contract or MTMRiskContract()
    if events.empty:
        empty = {
            "mode": mode, "events": 0, "accepted": 0, "total_return": 0.0,
            "max_mtm_drawdown": 0.0, "max_intrabar_stress_drawdown": 0.0,
            "max_rolling_cvar": np.nan, "profit_factor": np.nan,
            "hard_mtm_kill_triggered": False,
        }
        return empty, pd.DataFrame(), pd.DataFrame()

    norm_frames = {s: _normalize_frame(f) for s, f in frames.items()}
    x = events.copy().reset_index(drop=True)
    for col in ["signal_time", "entry_time", "exit_time"]:
        x[col] = pd.to_datetime(x[col], utc=True)
    for col in ["entry", "stop", "r_multiple"]:
        x[col] = pd.to_numeric(x[col], errors="coerce")
    mask = np.ones(len(x), dtype=bool) if selected is None else np.asarray(list(selected), dtype=bool)
    if len(mask) != len(x):
        raise ValueError("selection mask length mismatch")
    x["_selected"] = mask
    x = x.sort_values(["entry_time", "strategy", "symbol"], kind="mergesort").reset_index(drop=True)

    min_t = x["entry_time"].min()
    max_t = x["exit_time"].max()
    timeline = sorted({t for f in norm_frames.values() for t in f.loc[(f["timestamp"] >= min_t) & (f["timestamp"] <= max_t), "timestamp"]})
    if not timeline:
        raise ValueError("no MTM timeline overlaps candidate events")
    bar_lookup = {s: f.set_index("timestamp") for s, f in norm_frames.items()}
    entries: dict[pd.Timestamp, list[pd.Series]] = {}
    for _, row in x.iterrows():
        entries.setdefault(row["entry_time"], []).append(row)

    realized_equity = 1.0
    close_peak = 1.0
    stress_peak = 1.0
    min_close_dd = 0.0
    min_stress_dd = 0.0
    active: list[dict] = []
    ledger: list[dict] = []
    curve: list[dict] = []
    realized_pnl: list[float] = []
    accepted_r: list[float] = []
    close_returns: list[float] = []
    previous_close_equity: float | None = None
    reject_counts: Counter[str] = Counter()
    hard_kill = False
    corr_scaled_entries = 0
    max_concurrent = 0
    max_open_risk_fraction = 0.0
    max_cvar = np.nan

    def settle_due(now: pd.Timestamp) -> None:
        nonlocal realized_equity, active
        due = [p for p in active if p["exit_time"] < now]
        if not due:
            return
        for p in sorted(due, key=lambda q: (q["exit_time"], q["strategy"], q["symbol"])):
            pnl = float(p["risk_amount"] * p["r_multiple"])
            realized_equity += pnl
            realized_pnl.append(pnl)
            p["realized_pnl"] = pnl
            p["equity_after_exit"] = realized_equity
        active = [p for p in active if p not in due]

    for now in timeline:
        settle_due(now)
        rolling_cvar = _historical_cvar(close_returns[-c.cvar_lookback_bars :], c.cvar_alpha)
        if np.isfinite(rolling_cvar):
            max_cvar = float(rolling_cvar) if not np.isfinite(max_cvar) else max(max_cvar, float(rolling_cvar))

        for row in entries.get(now, []):
            reason = ""
            risk_scale = 1.0
            max_abs_corr = np.nan
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
                correlations = [
                    abs(_rolling_correlation(norm_frames, str(row["symbol"]), p["symbol"], now, c.correlation_lookback_bars))
                    for p in active
                ]
                correlations = [v for v in correlations if np.isfinite(v)]
                max_abs_corr = max(correlations) if correlations else np.nan
                if np.isfinite(max_abs_corr) and max_abs_corr > c.max_pairwise_correlation_for_full_size:
                    risk_scale = c.correlated_risk_multiplier
                    corr_scaled_entries += 1

                planned_risk = realized_equity * c.risk_per_trade * risk_scale
                current_risk = sum(float(p["risk_amount"]) for p in active)
                strategy_risk = sum(float(p["risk_amount"]) for p in active if p["strategy"] == str(row["strategy"]))
                direction = str(row["side"]).lower()
                directional_risk = sum(float(p["risk_amount"]) for p in active if p["side"] == direction)
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
                        pos = {
                            "strategy": str(row["strategy"]),
                            "symbol": str(row["symbol"]),
                            "side": str(row["side"]).lower(),
                            "entry_time": row["entry_time"],
                            "exit_time": row["exit_time"],
                            "entry": float(row["entry"]),
                            "stop": float(row["stop"]),
                            "risk_amount": float(planned_risk),
                            "risk_scale": float(risk_scale),
                            "notional": float(notional),
                            "r_multiple": float(row["r_multiple"]),
                            "max_abs_corr_at_entry": float(max_abs_corr) if np.isfinite(max_abs_corr) else np.nan,
                        }
                        active.append(pos)
                        accepted_r.append(float(row["r_multiple"]))
                        max_concurrent = max(max_concurrent, len(active))
                        max_open_risk_fraction = max(
                            max_open_risk_fraction,
                            sum(float(p["risk_amount"]) for p in active) / max(realized_equity, 1e-12),
                        )
            if reason:
                reject_counts[reason] += 1
            ledger.append({
                "entry_time": row["entry_time"],
                "exit_time": row["exit_time"],
                "strategy": row["strategy"],
                "symbol": row["symbol"],
                "side": row["side"],
                "selected_input": bool(row["_selected"]),
                "accepted": reason == "",
                "rejection_reason": reason,
                "risk_scale": float(risk_scale),
                "max_abs_corr_at_entry": float(max_abs_corr) if np.isfinite(max_abs_corr) else np.nan,
            })

        close_unrealized = 0.0
        stress_unrealized = 0.0
        for p in active:
            table = bar_lookup.get(p["symbol"])
            if table is None or now not in table.index:
                continue
            bar = table.loc[now]
            if isinstance(bar, pd.DataFrame):
                bar = bar.iloc[-1]
            close_unrealized += _position_mark_pnl(p, bar, "close", c)
            adverse_field = "low" if p["side"] == "long" else "high"
            stress_unrealized += _position_mark_pnl(p, bar, adverse_field, c)

        close_equity = realized_equity + close_unrealized
        stress_equity = realized_equity + stress_unrealized
        close_peak = max(close_peak, close_equity)
        stress_peak = max(stress_peak, close_equity)
        close_dd = close_equity / max(close_peak, 1e-12) - 1.0
        stress_dd = stress_equity / max(stress_peak, 1e-12) - 1.0
        min_close_dd = min(min_close_dd, close_dd)
        min_stress_dd = min(min_stress_dd, stress_dd)
        if stress_dd <= -c.hard_mtm_drawdown_kill:
            hard_kill = True
        if previous_close_equity is not None and previous_close_equity > 0:
            close_returns.append(close_equity / previous_close_equity - 1.0)
        previous_close_equity = close_equity
        curve.append({
            "timestamp": now,
            "realized_equity": realized_equity,
            "close_mtm_equity": close_equity,
            "intrabar_stress_equity": stress_equity,
            "close_mtm_drawdown": close_dd,
            "intrabar_stress_drawdown": stress_dd,
            "rolling_cvar_95": rolling_cvar,
            "active_positions": len(active),
            "open_risk_fraction": sum(float(p["risk_amount"]) for p in active) / max(realized_equity, 1e-12),
        })

        # Settle positions whose causal bracket exit occurred in the current bar
        # after recording that bar's adverse MTM excursion.
        due_now = [p for p in active if p["exit_time"] == now]
        for p in sorted(due_now, key=lambda q: (q["strategy"], q["symbol"])):
            pnl = float(p["risk_amount"] * p["r_multiple"])
            realized_equity += pnl
            realized_pnl.append(pnl)
            p["realized_pnl"] = pnl
            p["equity_after_exit"] = realized_equity
        active = [p for p in active if p not in due_now]

    # Any positions whose exit timestamp was not present in the union timeline are
    # settled using their recorded causal outcomes; this should normally be empty.
    for p in active:
        pnl = float(p["risk_amount"] * p["r_multiple"])
        realized_equity += pnl
        realized_pnl.append(pnl)

    wins = float(sum(v for v in realized_pnl if v > 0))
    losses = float(-sum(v for v in realized_pnl if v < 0))
    pf = wins / losses if losses > 0 else (np.inf if wins > 0 else np.nan)
    summary = {
        "mode": mode,
        "events": int(len(x)),
        "selected_input": int(x["_selected"].sum()),
        "accepted": int(sum(1 for r in ledger if r["accepted"])),
        "acceptance_fraction": float(sum(1 for r in ledger if r["accepted"]) / len(x)),
        "total_realized_return": float(realized_equity - 1.0),
        "ending_realized_equity": float(realized_equity),
        "max_mtm_drawdown": float(min_close_dd),
        "max_intrabar_stress_drawdown": float(min_stress_dd),
        "max_rolling_cvar": float(max_cvar) if np.isfinite(max_cvar) else np.nan,
        "profit_factor": float(pf),
        "mean_r_accepted": float(np.mean(accepted_r)) if accepted_r else np.nan,
        "max_concurrent": int(max_concurrent),
        "max_open_risk_fraction_seen": float(max_open_risk_fraction),
        "correlation_scaled_entries": int(corr_scaled_entries),
        "hard_mtm_kill_triggered": bool(hard_kill),
        "reject_counts": dict(reject_counts),
        "mark_to_market": True,
        "intrabar_stress_semantics": "long uses bar low; short uses bar high; full liquidation friction reserved",
    }
    return summary, pd.DataFrame(ledger), pd.DataFrame(curve)
