from __future__ import annotations

"""Causal event-driven risk governance for the v0.23 integrity audit.

This module deliberately leaves the frozen v0.20 implementation untouched.
The v0.20 overlay processes rows in entry-time order and immediately realizes
that row's eventual outcome. That is convenient for a sequential ledger but is
not causal when trades overlap across symbols: a later entry can observe an
earlier trade's outcome before the earlier trade has actually exited.

v0.23 separates *entry decisions* from *exit settlements*. At a candidate
entry timestamp, only positions with an exit timestamp strictly earlier than
the new entry are settled. Exits sharing the same timestamp are settled as a
batch, avoiding artificial within-bar ordering of portfolio equity.

The module is research/PAPER-only. It does not authorize live execution.
"""

from dataclasses import asdict
import json
from typing import Any

import numpy as np
import pandas as pd

from research_bot.multitimeframe_strategies_v20 import COOLDOWN, RiskPsychologyPolicy


INTRADAY_TIMEFRAMES = {"1m", "5m", "15m", "1h"}


def _timeframe_from_spec(spec: Any) -> str:
    timeframe = str(getattr(spec, "timeframe", spec))
    if timeframe not in COOLDOWN:
        raise ValueError(f"unsupported timeframe for causal risk overlay: {timeframe!r}")
    return timeframe


def _settle_batches_before(
    x: pd.DataFrame,
    pending: list[tuple[pd.Timestamp, int, int, float]],
    *,
    cutoff: pd.Timestamp | None,
    equity: float,
    peak: float,
    streak: int,
    cooldown: pd.Timestamp | None,
    hard_killed: bool,
    timeframe: str,
    policy: RiskPsychologyPolicy,
) -> tuple[list[tuple[pd.Timestamp, int, int, float]], float, float, int, pd.Timestamp | None, bool]:
    """Settle exits before ``cutoff``; ``None`` settles every remaining exit.

    Strict ``exit_time < entry_time`` ordering is intentional. In the bracket
    simulator, timestamps identify bars rather than exact intrabar event times,
    so an exit recorded on the same bar as a new entry is not assumed known at
    that bar's open.
    """

    if cutoff is None:
        due = pending
        remaining: list[tuple[pd.Timestamp, int, int, float]] = []
    else:
        due = [event for event in pending if event[0] < cutoff]
        remaining = [event for event in pending if event[0] >= cutoff]

    if not due:
        return remaining, equity, peak, streak, cooldown, hard_killed

    due = sorted(due, key=lambda event: (event[0], event[1]))
    cursor = 0
    while cursor < len(due):
        exit_time = due[cursor][0]
        batch: list[tuple[pd.Timestamp, int, int, float]] = []
        while cursor < len(due) and due[cursor][0] == exit_time:
            batch.append(due[cursor])
            cursor += 1

        returns = np.asarray([event[3] for event in batch], dtype=float)
        if np.any(~np.isfinite(returns)):
            raise ValueError("pending settlement contains a non-finite account return")
        if np.any(returns <= -1.0):
            raise ValueError("account return <= -100% is invalid for multiplicative equity accounting")

        # Same-timestamp exits form one portfolio settlement batch. Multiplying
        # factors is order-invariant and avoids inventing intrabar path/peaks.
        equity *= float(np.prod(1.0 + returns))
        peak = max(peak, equity)
        dd_after = equity / peak - 1.0

        for _, _, row_i, _ in batch:
            x.at[row_i, "settlement_equity_v23"] = equity
            x.at[row_i, "settlement_drawdown_v23"] = dd_after
            x.at[row_i, "settlement_batch_size_v23"] = len(batch)

        positive = int(np.sum(returns > 0.0))
        losses = int(np.sum(returns < 0.0))
        # Exact sequencing within one bar is unknown. A batch containing a win
        # breaks the loss streak; an all-loss batch accumulates its losses.
        if positive:
            streak = 0
        elif losses:
            streak += losses
            if streak >= policy.max_consecutive_losses:
                candidate = exit_time + COOLDOWN[timeframe]
                cooldown = candidate if cooldown is None else max(cooldown, candidate)
                streak = 0

        if dd_after <= -policy.hard_drawdown_kill:
            hard_killed = True

    return remaining, equity, peak, streak, cooldown, hard_killed


def apply_causal_risk_overlay_v23(
    ledger: pd.DataFrame,
    spec: Any,
    policy: RiskPsychologyPolicy | None = None,
) -> pd.DataFrame:
    """Apply risk governance without allowing future trade outcomes into entries.

    The function is intentionally versioned instead of replacing v0.20 so all
    previously generated thesis evidence remains reproducible.  The output
    keeps raw ``r_multiple`` untouched and writes v0.23-specific accounting
    columns.
    """

    p = policy or RiskPsychologyPolicy()
    timeframe = _timeframe_from_spec(spec)
    if ledger.empty:
        return ledger.copy()

    required = {"symbol", "entry_time", "exit_time", "r_multiple"}
    missing = required - set(ledger.columns)
    if missing:
        raise ValueError(f"risk overlay missing required columns: {sorted(missing)}")

    x = ledger.sort_values(["entry_time", "symbol"], kind="mergesort").reset_index(drop=True).copy()
    x["entry_time"] = pd.to_datetime(x["entry_time"], utc=True, errors="raise")
    x["exit_time"] = pd.to_datetime(x["exit_time"], utc=True, errors="raise")
    if bool((x["exit_time"] < x["entry_time"]).any()):
        raise ValueError("exit_time precedes entry_time in risk ledger")

    defaults: dict[str, object] = {
        "executed_v23": False,
        "reject_reason_v23": "",
        "risk_scale_drawdown_v23": 0.0,
        "risk_fraction_v23": 0.0,
        "account_return_v23": 0.0,
        "equity_at_entry_v23": np.nan,
        "drawdown_before_v23": np.nan,
        "loss_streak_before_v23": 0,
        "open_positions_before_v23": 0,
        "settlement_equity_v23": np.nan,
        "settlement_drawdown_v23": np.nan,
        "settlement_batch_size_v23": 0,
    }
    for column, default in defaults.items():
        x[column] = default

    equity = 1.0
    peak = 1.0
    streak = 0
    cooldown: pd.Timestamp | None = None
    hard_killed = False
    day_counts: dict[tuple[str, object], int] = {}
    pending: list[tuple[pd.Timestamp, int, int, float]] = []
    sequence = 0

    for row_i, row in x.iterrows():
        entry_time = row["entry_time"]
        pending, equity, peak, streak, cooldown, hard_killed = _settle_batches_before(
            x,
            pending,
            cutoff=entry_time,
            equity=equity,
            peak=peak,
            streak=streak,
            cooldown=cooldown,
            hard_killed=hard_killed,
            timeframe=timeframe,
            policy=p,
        )

        dd = equity / peak - 1.0
        x.at[row_i, "equity_at_entry_v23"] = equity
        x.at[row_i, "drawdown_before_v23"] = dd
        x.at[row_i, "loss_streak_before_v23"] = streak
        x.at[row_i, "open_positions_before_v23"] = len(pending)

        key = (str(row["symbol"]), entry_time.date())
        reason = ""
        if hard_killed or dd <= -p.hard_drawdown_kill:
            hard_killed = True
            reason = "HARD_DRAWDOWN_KILL"
        elif cooldown is not None and entry_time < cooldown:
            reason = "LOSS_STREAK_COOLDOWN"
        elif timeframe in INTRADAY_TIMEFRAMES and day_counts.get(key, 0) >= p.max_daily_trades_intraday:
            reason = "OVERTRADING_DAILY_CAP"

        if reason:
            x.at[row_i, "reject_reason_v23"] = reason
            continue

        dd_scale = p.drawdown_scale_2 if dd <= -p.drawdown_warn_2 else (
            p.drawdown_scale_1 if dd <= -p.drawdown_warn_1 else 1.0
        )
        vol_raw = row.get("risk_scale_volatility", 1.0)
        vol_raw = 1.0 if pd.isna(vol_raw) else float(vol_raw)
        vol_scale = float(np.clip(vol_raw, p.vol_floor_scale, 1.0))
        risk = min(p.max_risk_per_trade, p.base_risk_per_trade * dd_scale * vol_scale)
        account_return = risk * float(row["r_multiple"])
        if not np.isfinite(account_return):
            raise ValueError("non-finite account return produced by risk overlay")
        if account_return <= -1.0:
            raise ValueError("account return <= -100% is invalid for multiplicative equity accounting")

        x.at[row_i, "executed_v23"] = True
        x.at[row_i, "risk_scale_drawdown_v23"] = dd_scale
        x.at[row_i, "risk_fraction_v23"] = risk
        x.at[row_i, "account_return_v23"] = account_return
        day_counts[key] = day_counts.get(key, 0) + 1
        pending.append((row["exit_time"], sequence, row_i, account_return))
        sequence += 1

    pending, equity, peak, streak, cooldown, hard_killed = _settle_batches_before(
        x,
        pending,
        cutoff=None,
        equity=equity,
        peak=peak,
        streak=streak,
        cooldown=cooldown,
        hard_killed=hard_killed,
        timeframe=timeframe,
        policy=p,
    )
    if pending:
        raise AssertionError("internal error: unsettled v0.23 risk events remain")

    x.attrs["v23_risk_audit"] = {
        "final_realized_equity": float(equity),
        "final_realized_drawdown": float(equity / peak - 1.0),
        "hard_killed": bool(hard_killed),
        "policy": asdict(p),
        "timeframe": timeframe,
        "causality_contract": "entry sees settlements with exit_time strictly earlier than entry_time only",
        "live_execution_authorized": False,
    }
    return x


def causality_regression_probe() -> dict[str, object]:
    """Synthetic invariant: an unresolved future outcome cannot alter a new entry."""

    base = pd.DataFrame(
        [
            {
                "symbol": "BTC/USDT",
                "entry_time": "2026-01-01T00:00:00Z",
                "exit_time": "2026-01-02T00:00:00Z",
                "r_multiple": -10.0,
                "risk_scale_volatility": 1.0,
            },
            {
                "symbol": "ETH/USDT",
                "entry_time": "2026-01-01T04:00:00Z",
                "exit_time": "2026-01-01T08:00:00Z",
                "r_multiple": 1.0,
                "risk_scale_volatility": 1.0,
            },
            {
                "symbol": "SOL/USDT",
                "entry_time": "2026-01-02T04:00:00Z",
                "exit_time": "2026-01-02T08:00:00Z",
                "r_multiple": 1.0,
                "risk_scale_volatility": 1.0,
            },
        ]
    )
    alternate = base.copy()
    alternate.loc[0, "r_multiple"] = 10.0

    loss_path = apply_causal_risk_overlay_v23(base, "4h")
    win_path = apply_causal_risk_overlay_v23(alternate, "4h")

    eth_loss = loss_path.loc[loss_path["symbol"] == "ETH/USDT"].iloc[0]
    eth_win = win_path.loc[win_path["symbol"] == "ETH/USDT"].iloc[0]
    sol_loss = loss_path.loc[loss_path["symbol"] == "SOL/USDT"].iloc[0]
    sol_win = win_path.loc[win_path["symbol"] == "SOL/USDT"].iloc[0]

    unresolved_outcome_invariant = bool(
        np.isclose(float(eth_loss["risk_fraction_v23"]), float(eth_win["risk_fraction_v23"]))
        and np.isclose(float(eth_loss["drawdown_before_v23"]), float(eth_win["drawdown_before_v23"]))
    )
    resolved_outcome_can_change_future_risk = bool(
        not np.isclose(float(sol_loss["risk_fraction_v23"]), float(sol_win["risk_fraction_v23"]))
    )
    passed = unresolved_outcome_invariant and resolved_outcome_can_change_future_risk

    return {
        "version": "v0.23",
        "audit": "causal-risk-overlap-regression",
        "passed": passed,
        "unresolved_outcome_invariant": unresolved_outcome_invariant,
        "resolved_outcome_can_change_future_risk": resolved_outcome_can_change_future_risk,
        "eth_risk_if_btc_future_loss": float(eth_loss["risk_fraction_v23"]),
        "eth_risk_if_btc_future_win": float(eth_win["risk_fraction_v23"]),
        "sol_risk_after_btc_loss_is_known": float(sol_loss["risk_fraction_v23"]),
        "sol_risk_after_btc_win_is_known": float(sol_win["risk_fraction_v23"]),
        "live_execution_authorized": False,
    }


if __name__ == "__main__":
    result = causality_regression_probe()
    print(json.dumps(result, indent=2, sort_keys=True))
    if not bool(result["passed"]):
        raise SystemExit(1)
