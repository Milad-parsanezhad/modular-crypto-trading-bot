from __future__ import annotations

"""v0.25 pre-entry portfolio risk allocator.

Research objective
------------------
v0.23 repaired event-time causality and fixed-position cash accounting. v0.24
then showed that many candidates accumulate material concurrent risk even
though per-trade risk is small. v0.25 adds a *pre-entry* portfolio budget before
any new position is admitted.

The policy is deliberately simple and preregistered rather than optimized:
- aggregate open risk-at-stop <= 2.0% of realized equity;
- same-direction open risk-at-stop <= 1.5% of realized equity;
- candidate risk still starts from the frozen v0.20 volatility/drawdown sizing;
- simultaneous entries are scaled pro-rata as one batch, so symbol ordering
  cannot decide who receives the remaining risk budget;
- exits stamped on the same bar are still open at that bar's entry decision,
  preserving the strict v0.23 causal contract.

These limits are portfolio-construction constraints, not performance gates.
The frozen 5% drawdown qualification threshold remains unchanged.

Research/PAPER only. Live execution is not authorized.
"""

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from research_bot.multitimeframe_strategies_v20 import COOLDOWN, RiskPsychologyPolicy


INTRADAY_TIMEFRAMES = {"1m", "5m", "15m", "1h"}


@dataclass(frozen=True)
class PortfolioRiskBudgetV25:
    """Preregistered capital-at-risk limits applied before entry."""

    aggregate_open_risk_cap: float = 0.020
    directional_open_risk_cap: float = 0.015

    def __post_init__(self) -> None:
        if not (0.0 < self.aggregate_open_risk_cap < 1.0):
            raise ValueError("aggregate_open_risk_cap must lie in (0, 1)")
        if not (0.0 < self.directional_open_risk_cap <= self.aggregate_open_risk_cap):
            raise ValueError("directional_open_risk_cap must lie in (0, aggregate cap]")


@dataclass(frozen=True)
class PendingPositionV25:
    exit_time: pd.Timestamp
    sequence: int
    row_index: int
    pnl_cash: float
    account_return_at_entry: float
    risk_cash: float
    side: int


def _timeframe_from_spec(spec: Any) -> str:
    timeframe = str(getattr(spec, "timeframe", spec))
    if timeframe not in COOLDOWN:
        raise ValueError(f"unsupported timeframe for v0.25 allocator: {timeframe!r}")
    return timeframe


def _settle_before(
    x: pd.DataFrame,
    pending: list[PendingPositionV25],
    *,
    cutoff: pd.Timestamp | None,
    equity: float,
    peak: float,
    streak: int,
    cooldown: pd.Timestamp | None,
    hard_killed: bool,
    timeframe: str,
    risk_policy: RiskPsychologyPolicy,
) -> tuple[list[PendingPositionV25], float, float, int, pd.Timestamp | None, bool]:
    """Settle exits strictly before cutoff; None flushes all remaining exits."""

    if cutoff is None:
        due = list(pending)
        remaining: list[PendingPositionV25] = []
    else:
        due = [position for position in pending if position.exit_time < cutoff]
        remaining = [position for position in pending if position.exit_time >= cutoff]

    if not due:
        return remaining, equity, peak, streak, cooldown, hard_killed

    due.sort(key=lambda position: (position.exit_time, position.sequence))
    cursor = 0
    while cursor < len(due):
        exit_time = due[cursor].exit_time
        batch: list[PendingPositionV25] = []
        while cursor < len(due) and due[cursor].exit_time == exit_time:
            batch.append(due[cursor])
            cursor += 1

        pnl = np.asarray([position.pnl_cash for position in batch], dtype=float)
        normalized = np.asarray([position.account_return_at_entry for position in batch], dtype=float)
        if np.any(~np.isfinite(pnl)) or np.any(~np.isfinite(normalized)):
            raise ValueError("non-finite pending settlement value")

        equity_before = float(equity)
        batch_pnl = float(pnl.sum())
        equity = float(equity_before + batch_pnl)
        batch_return = float(batch_pnl / equity_before) if equity_before != 0 else np.nan
        peak = max(float(peak), equity)
        dd_after = float(equity / peak - 1.0) if peak > 0 else -np.inf

        for position in batch:
            row_i = position.row_index
            x.at[row_i, "settlement_equity_before_v25"] = equity_before
            x.at[row_i, "settlement_batch_pnl_v25"] = batch_pnl
            x.at[row_i, "settlement_batch_return_v25"] = batch_return
            x.at[row_i, "settlement_equity_v25"] = equity
            x.at[row_i, "settlement_drawdown_v25"] = dd_after
            x.at[row_i, "settlement_batch_size_v25"] = len(batch)

        positive = int(np.sum(normalized > 0.0))
        losses = int(np.sum(normalized < 0.0))
        if positive:
            streak = 0
        elif losses:
            streak += losses
            if streak >= risk_policy.max_consecutive_losses:
                candidate = exit_time + COOLDOWN[timeframe]
                cooldown = candidate if cooldown is None else max(cooldown, candidate)
                streak = 0

        if not np.isfinite(equity) or equity <= 0 or dd_after <= -risk_policy.hard_drawdown_kill:
            hard_killed = True

    return remaining, equity, peak, streak, cooldown, hard_killed


def _open_risk_state(pending: list[PendingPositionV25]) -> tuple[float, dict[int, float]]:
    total = float(sum(position.risk_cash for position in pending))
    directional = {
        -1: float(sum(position.risk_cash for position in pending if position.side < 0)),
        1: float(sum(position.risk_cash for position in pending if position.side > 0)),
    }
    return total, directional


def _pro_rata_batch_allocation(
    proposals: pd.DataFrame,
    *,
    equity: float,
    open_total_risk_cash: float,
    open_directional_risk_cash: dict[int, float],
    budget: PortfolioRiskBudgetV25,
) -> pd.Series:
    """Allocate simultaneous proposals without arbitrary within-bar priority."""

    if proposals.empty:
        return pd.Series(dtype=float)
    if equity <= 0 or not np.isfinite(equity):
        return pd.Series(0.0, index=proposals.index, dtype=float)

    desired = proposals["proposed_risk_cash_v25"].astype(float).clip(lower=0.0)
    allocated = desired.copy()

    # First constrain long and short buckets independently. This approximates a
    # systematic crypto correlation cluster without fitting a noisy correlation
    # matrix to the same validation sample.
    directional_cap_cash = float(budget.directional_open_risk_cap * equity)
    for side in (-1, 1):
        mask = proposals["side"].astype(int).eq(side)
        wanted = float(desired[mask].sum())
        available = max(0.0, directional_cap_cash - float(open_directional_risk_cash.get(side, 0.0)))
        scale = min(1.0, available / wanted) if wanted > 0 else 1.0
        allocated.loc[mask] = desired.loc[mask] * scale

    # Then enforce the portfolio-wide cap on the already direction-limited batch.
    total_cap_cash = float(budget.aggregate_open_risk_cap * equity)
    available_total = max(0.0, total_cap_cash - float(open_total_risk_cash))
    wanted_total = float(allocated.sum())
    total_scale = min(1.0, available_total / wanted_total) if wanted_total > 0 else 1.0
    allocated *= total_scale

    # Numerical clipping only; not a hidden tuning threshold.
    allocated = allocated.clip(lower=0.0)
    return allocated


def apply_portfolio_allocator_v25(
    ledger: pd.DataFrame,
    spec: Any,
    *,
    risk_policy: RiskPsychologyPolicy | None = None,
    budget: PortfolioRiskBudgetV25 | None = None,
) -> pd.DataFrame:
    """Apply causal risk governance plus pre-entry aggregate risk allocation."""

    p = risk_policy or RiskPsychologyPolicy()
    b = budget or PortfolioRiskBudgetV25()
    timeframe = _timeframe_from_spec(spec)
    if ledger.empty:
        return ledger.copy()

    required = {"symbol", "entry_time", "exit_time", "r_multiple", "side"}
    missing = required - set(ledger.columns)
    if missing:
        raise ValueError(f"v0.25 allocator missing required columns: {sorted(missing)}")

    x = ledger.sort_values(["entry_time", "symbol"], kind="mergesort").reset_index(drop=True).copy()
    x["entry_time"] = pd.to_datetime(x["entry_time"], utc=True, errors="raise")
    x["exit_time"] = pd.to_datetime(x["exit_time"], utc=True, errors="raise")
    x["side"] = pd.to_numeric(x["side"], errors="raise").astype(int)
    if bool((~x["side"].isin([-1, 1])).any()):
        raise ValueError("v0.25 allocator requires side in {-1, +1}")
    if bool((x["exit_time"] < x["entry_time"]).any()):
        raise ValueError("exit_time precedes entry_time in v0.25 ledger")

    defaults: dict[str, object] = {
        "executed_v25": False,
        "reject_reason_v25": "",
        "risk_scale_drawdown_v25": 0.0,
        "proposed_risk_fraction_v25": 0.0,
        "allocated_risk_fraction_v25": 0.0,
        "portfolio_allocation_scale_v25": 0.0,
        "account_return_v25": 0.0,
        "pnl_cash_v25": 0.0,
        "equity_at_entry_v25": np.nan,
        "drawdown_before_v25": np.nan,
        "loss_streak_before_v25": 0,
        "open_positions_before_v25": 0,
        "open_risk_cash_before_v25": 0.0,
        "open_risk_fraction_before_v25": 0.0,
        "open_directional_risk_cash_before_v25": 0.0,
        "open_directional_risk_fraction_before_v25": 0.0,
        "open_risk_cash_after_v25": 0.0,
        "open_risk_fraction_after_v25": 0.0,
        "open_directional_risk_cash_after_v25": 0.0,
        "open_directional_risk_fraction_after_v25": 0.0,
        "settlement_equity_before_v25": np.nan,
        "settlement_batch_pnl_v25": np.nan,
        "settlement_batch_return_v25": np.nan,
        "settlement_equity_v25": np.nan,
        "settlement_drawdown_v25": np.nan,
        "settlement_batch_size_v25": 0,
    }
    for column, default in defaults.items():
        x[column] = default

    equity = 1.0
    peak = 1.0
    streak = 0
    cooldown: pd.Timestamp | None = None
    hard_killed = False
    day_counts: dict[tuple[str, object], int] = {}
    pending: list[PendingPositionV25] = []
    sequence = 0

    # Process simultaneous entries as a batch. This removes arbitrary priority
    # from alphabetical symbol ordering when the risk budget is scarce.
    for entry_time, group in x.groupby("entry_time", sort=True):
        pending, equity, peak, streak, cooldown, hard_killed = _settle_before(
            x,
            pending,
            cutoff=entry_time,
            equity=equity,
            peak=peak,
            streak=streak,
            cooldown=cooldown,
            hard_killed=hard_killed,
            timeframe=timeframe,
            risk_policy=p,
        )

        dd = float(equity / peak - 1.0) if peak > 0 else -np.inf
        open_total_cash, open_dir_cash = _open_risk_state(pending)
        group_indices = list(group.index)

        proposals: list[int] = []
        for row_i in group_indices:
            row = x.loc[row_i]
            side = int(row["side"])
            x.at[row_i, "equity_at_entry_v25"] = equity
            x.at[row_i, "drawdown_before_v25"] = dd
            x.at[row_i, "loss_streak_before_v25"] = streak
            x.at[row_i, "open_positions_before_v25"] = len(pending)
            x.at[row_i, "open_risk_cash_before_v25"] = open_total_cash
            x.at[row_i, "open_risk_fraction_before_v25"] = open_total_cash / equity if equity > 0 else np.inf
            x.at[row_i, "open_directional_risk_cash_before_v25"] = open_dir_cash[side]
            x.at[row_i, "open_directional_risk_fraction_before_v25"] = open_dir_cash[side] / equity if equity > 0 else np.inf

            key = (str(row["symbol"]), entry_time.date())
            reason = ""
            if hard_killed or not np.isfinite(equity) or equity <= 0 or dd <= -p.hard_drawdown_kill:
                hard_killed = True
                reason = "HARD_DRAWDOWN_KILL"
            elif cooldown is not None and entry_time < cooldown:
                reason = "LOSS_STREAK_COOLDOWN"
            elif timeframe in INTRADAY_TIMEFRAMES and day_counts.get(key, 0) >= p.max_daily_trades_intraday:
                reason = "OVERTRADING_DAILY_CAP"

            if reason:
                x.at[row_i, "reject_reason_v25"] = reason
                continue

            dd_scale = p.drawdown_scale_2 if dd <= -p.drawdown_warn_2 else (
                p.drawdown_scale_1 if dd <= -p.drawdown_warn_1 else 1.0
            )
            vol_raw = row.get("risk_scale_volatility", 1.0)
            vol_raw = 1.0 if pd.isna(vol_raw) else float(vol_raw)
            vol_scale = float(np.clip(vol_raw, p.vol_floor_scale, 1.0))
            proposed_fraction = min(p.max_risk_per_trade, p.base_risk_per_trade * dd_scale * vol_scale)
            proposed_cash = float(equity * proposed_fraction)
            x.at[row_i, "risk_scale_drawdown_v25"] = dd_scale
            x.at[row_i, "proposed_risk_fraction_v25"] = proposed_fraction
            x.at[row_i, "proposed_risk_cash_v25"] = proposed_cash
            proposals.append(row_i)

        if not proposals:
            continue

        proposal_frame = x.loc[proposals, ["side", "proposed_risk_cash_v25"]].copy()
        allocated_cash = _pro_rata_batch_allocation(
            proposal_frame,
            equity=equity,
            open_total_risk_cash=open_total_cash,
            open_directional_risk_cash=open_dir_cash,
            budget=b,
        )

        # Admit every proposal with positive budget. Simultaneous entries share
        # the constrained budget proportionally rather than being first-come.
        new_positions: list[PendingPositionV25] = []
        for row_i in proposals:
            row = x.loc[row_i]
            side = int(row["side"])
            proposed_cash = float(row["proposed_risk_cash_v25"])
            risk_cash = float(allocated_cash.loc[row_i])
            if risk_cash <= 1e-15:
                x.at[row_i, "reject_reason_v25"] = "PORTFOLIO_RISK_BUDGET_EXHAUSTED"
                continue

            risk_fraction = float(risk_cash / equity)
            scale = float(risk_cash / proposed_cash) if proposed_cash > 0 else 0.0
            account_return = float(risk_fraction * float(row["r_multiple"]))
            if not np.isfinite(account_return):
                raise ValueError("non-finite v0.25 account return")
            if account_return <= -1.0:
                raise ValueError("single-position entry-normalized loss <= -100% is invalid")
            pnl_cash = float(equity * account_return)

            x.at[row_i, "executed_v25"] = True
            x.at[row_i, "allocated_risk_fraction_v25"] = risk_fraction
            x.at[row_i, "portfolio_allocation_scale_v25"] = scale
            x.at[row_i, "account_return_v25"] = account_return
            x.at[row_i, "pnl_cash_v25"] = pnl_cash
            key = (str(row["symbol"]), entry_time.date())
            day_counts[key] = day_counts.get(key, 0) + 1
            position = PendingPositionV25(
                exit_time=row["exit_time"],
                sequence=sequence,
                row_index=row_i,
                pnl_cash=pnl_cash,
                account_return_at_entry=account_return,
                risk_cash=risk_cash,
                side=side,
            )
            sequence += 1
            new_positions.append(position)

        pending.extend(new_positions)
        after_total_cash, after_dir_cash = _open_risk_state(pending)
        for row_i in group_indices:
            side = int(x.at[row_i, "side"])
            x.at[row_i, "open_risk_cash_after_v25"] = after_total_cash
            x.at[row_i, "open_risk_fraction_after_v25"] = after_total_cash / equity if equity > 0 else np.inf
            x.at[row_i, "open_directional_risk_cash_after_v25"] = after_dir_cash[side]
            x.at[row_i, "open_directional_risk_fraction_after_v25"] = after_dir_cash[side] / equity if equity > 0 else np.inf

        # Strong invariants: they convert the allocator into an executable
        # research contract rather than a descriptive intention.
        tol = 1e-12
        if after_total_cash > b.aggregate_open_risk_cap * equity + tol:
            raise AssertionError("v0.25 aggregate risk budget breached")
        for side in (-1, 1):
            if after_dir_cash[side] > b.directional_open_risk_cap * equity + tol:
                raise AssertionError("v0.25 directional risk budget breached")

    pending, equity, peak, streak, cooldown, hard_killed = _settle_before(
        x,
        pending,
        cutoff=None,
        equity=equity,
        peak=peak,
        streak=streak,
        cooldown=cooldown,
        hard_killed=hard_killed,
        timeframe=timeframe,
        risk_policy=p,
    )
    if pending:
        raise AssertionError("internal error: unsettled v0.25 positions remain")

    x.attrs["v25_portfolio_allocator"] = {
        "final_realized_equity": float(equity),
        "final_realized_return": float(equity - 1.0),
        "final_realized_drawdown": float(equity / peak - 1.0) if peak > 0 else -np.inf,
        "hard_killed": bool(hard_killed),
        "risk_policy": asdict(p),
        "portfolio_budget": asdict(b),
        "timeframe": timeframe,
        "causality_contract": "entry sees settlements with exit_time strictly earlier than entry_time only",
        "allocation_contract": "same-timestamp entries share aggregate and directional budgets pro-rata",
        "live_execution_authorized": False,
    }
    return x
