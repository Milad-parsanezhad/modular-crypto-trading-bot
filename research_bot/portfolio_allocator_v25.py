from __future__ import annotations

"""v0.25 pre-entry portfolio risk allocator.

Research objective
------------------
v0.23 repaired event-time causality and fixed-position cash accounting. v0.24
then showed that many candidates accumulate material concurrent risk even
though per-trade risk is small. v0.25 adds a *pre-entry* portfolio budget before
any new position is admitted.

The policy is deliberately simple and preregistered rather than optimized:
- aggregate cost-adjusted loss-at-stop <= 2.0% of realized equity;
- same-direction cost-adjusted loss-at-stop <= 1.5% of realized equity;
- candidate nominal risk still starts from the frozen v0.20 volatility/drawdown
  sizing;
- the budget includes the frozen 24 bps round-trip friction, so a stop whose
  realized R is slightly below -1 cannot silently breach the intended budget;
- simultaneous entries are scaled pro-rata as one batch, so symbol ordering
  cannot decide who receives the remaining risk budget;
- exits stamped on the same bar are still open at that bar's entry decision,
  preserving the strict v0.23 causal contract.

If realized equity falls while other positions remain open, an inherited
position can temporarily sit above a percentage budget that was respected at
its own entry. v0.25 never rewrites or resizes that historical position. It
freezes new exposure in the affected bucket until exits release enough budget.

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
    round_trip_cost_fraction: float = 0.0024

    def __post_init__(self) -> None:
        if not (0.0 < self.aggregate_open_risk_cap < 1.0):
            raise ValueError("aggregate_open_risk_cap must lie in (0, 1)")
        if not (0.0 < self.directional_open_risk_cap <= self.aggregate_open_risk_cap):
            raise ValueError("directional_open_risk_cap must lie in (0, aggregate cap]")
        if not (0.0 <= self.round_trip_cost_fraction < 0.10):
            raise ValueError("round_trip_cost_fraction must lie in [0, 0.10)")


@dataclass(frozen=True)
class PendingPositionV25:
    exit_time: pd.Timestamp
    sequence: int
    row_index: int
    pnl_cash: float
    account_return_at_entry: float
    stop_loss_budget_cash: float
    side: int


def _timeframe_from_spec(spec: Any) -> str:
    timeframe = str(getattr(spec, "timeframe", spec))
    if timeframe not in COOLDOWN:
        raise ValueError(f"unsupported timeframe for v0.25 allocator: {timeframe!r}")
    return timeframe


def _cost_adjusted_stop_multiple(row: pd.Series, budget: PortfolioRiskBudgetV25) -> float:
    """Absolute stop loss in R units including frozen round-trip friction."""

    entry = float(row["entry"])
    stop = float(row["stop"])
    if not np.isfinite(entry) or not np.isfinite(stop) or entry <= 0:
        raise ValueError("invalid entry/stop for v0.25 cost-adjusted risk budget")
    stop_fraction = abs(stop / entry - 1.0)
    if not np.isfinite(stop_fraction) or stop_fraction <= 0:
        raise ValueError("non-positive stop distance in v0.25 risk budget")
    return float(1.0 + budget.round_trip_cost_fraction / stop_fraction)


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
    total = float(sum(position.stop_loss_budget_cash for position in pending))
    directional = {
        -1: float(sum(position.stop_loss_budget_cash for position in pending if position.side < 0)),
        1: float(sum(position.stop_loss_budget_cash for position in pending if position.side > 0)),
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
    """Allocate simultaneous cost-adjusted stop budgets without symbol priority."""

    if proposals.empty:
        return pd.Series(dtype=float)
    if equity <= 0 or not np.isfinite(equity):
        return pd.Series(0.0, index=proposals.index, dtype=float)

    desired = proposals["proposed_stop_loss_budget_cash_v25"].astype(float).clip(lower=0.0)
    allocated = desired.copy()

    directional_cap_cash = float(budget.directional_open_risk_cap * equity)
    for side in (-1, 1):
        mask = proposals["side"].astype(int).eq(side)
        wanted = float(desired[mask].sum())
        available = max(0.0, directional_cap_cash - float(open_directional_risk_cash.get(side, 0.0)))
        scale = min(1.0, available / wanted) if wanted > 0 else 1.0
        allocated.loc[mask] = desired.loc[mask] * scale

    total_cap_cash = float(budget.aggregate_open_risk_cap * equity)
    available_total = max(0.0, total_cap_cash - float(open_total_risk_cash))
    wanted_total = float(allocated.sum())
    total_scale = min(1.0, available_total / wanted_total) if wanted_total > 0 else 1.0
    allocated *= total_scale
    return allocated.clip(lower=0.0)


def apply_portfolio_allocator_v25(
    ledger: pd.DataFrame,
    spec: Any,
    *,
    risk_policy: RiskPsychologyPolicy | None = None,
    budget: PortfolioRiskBudgetV25 | None = None,
) -> pd.DataFrame:
    """Apply causal risk governance plus cost-aware pre-entry portfolio allocation."""

    p = risk_policy or RiskPsychologyPolicy()
    b = budget or PortfolioRiskBudgetV25()
    timeframe = _timeframe_from_spec(spec)
    if ledger.empty:
        return ledger.copy()

    required = {"symbol", "entry_time", "exit_time", "r_multiple", "side", "entry", "stop"}
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
        "proposed_nominal_risk_cash_v25": 0.0,
        "stop_loss_multiple_abs_v25": np.nan,
        "proposed_stop_loss_budget_cash_v25": 0.0,
        "allocated_stop_loss_budget_cash_v25": 0.0,
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
            proposed_nominal_cash = float(equity * proposed_fraction)
            stop_loss_multiple = _cost_adjusted_stop_multiple(row, b)
            proposed_stop_budget_cash = float(proposed_nominal_cash * stop_loss_multiple)

            x.at[row_i, "risk_scale_drawdown_v25"] = dd_scale
            x.at[row_i, "proposed_risk_fraction_v25"] = proposed_fraction
            x.at[row_i, "proposed_nominal_risk_cash_v25"] = proposed_nominal_cash
            x.at[row_i, "stop_loss_multiple_abs_v25"] = stop_loss_multiple
            x.at[row_i, "proposed_stop_loss_budget_cash_v25"] = proposed_stop_budget_cash
            proposals.append(row_i)

        if proposals:
            proposal_frame = x.loc[proposals, ["side", "proposed_stop_loss_budget_cash_v25"]].copy()
            allocated_stop_budget = _pro_rata_batch_allocation(
                proposal_frame,
                equity=equity,
                open_total_risk_cash=open_total_cash,
                open_directional_risk_cash=open_dir_cash,
                budget=b,
            )

            new_positions: list[PendingPositionV25] = []
            for row_i in proposals:
                row = x.loc[row_i]
                side = int(row["side"])
                proposed_stop_budget_cash = float(row["proposed_stop_loss_budget_cash_v25"])
                stop_loss_multiple = float(row["stop_loss_multiple_abs_v25"])
                allocated_stop_loss_cash = float(allocated_stop_budget.loc[row_i])
                if allocated_stop_loss_cash <= 1e-15:
                    x.at[row_i, "reject_reason_v25"] = "PORTFOLIO_RISK_BUDGET_EXHAUSTED"
                    continue

                nominal_risk_cash = float(allocated_stop_loss_cash / stop_loss_multiple)
                risk_fraction = float(nominal_risk_cash / equity)
                scale = float(allocated_stop_loss_cash / proposed_stop_budget_cash) if proposed_stop_budget_cash > 0 else 0.0
                account_return = float(risk_fraction * float(row["r_multiple"]))
                if not np.isfinite(account_return):
                    raise ValueError("non-finite v0.25 account return")
                if account_return <= -1.0:
                    raise ValueError("single-position entry-normalized loss <= -100% is invalid")
                pnl_cash = float(equity * account_return)

                x.at[row_i, "executed_v25"] = True
                x.at[row_i, "allocated_stop_loss_budget_cash_v25"] = allocated_stop_loss_cash
                x.at[row_i, "allocated_risk_fraction_v25"] = risk_fraction
                x.at[row_i, "portfolio_allocation_scale_v25"] = scale
                x.at[row_i, "account_return_v25"] = account_return
                x.at[row_i, "pnl_cash_v25"] = pnl_cash
                key = (str(row["symbol"]), entry_time.date())
                day_counts[key] = day_counts.get(key, 0) + 1
                new_positions.append(
                    PendingPositionV25(
                        exit_time=row["exit_time"],
                        sequence=sequence,
                        row_index=row_i,
                        pnl_cash=pnl_cash,
                        account_return_at_entry=account_return,
                        stop_loss_budget_cash=allocated_stop_loss_cash,
                        side=side,
                    )
                )
                sequence += 1
            pending.extend(new_positions)

        after_total_cash, after_dir_cash = _open_risk_state(pending)
        for row_i in group_indices:
            side = int(x.at[row_i, "side"])
            x.at[row_i, "open_risk_cash_after_v25"] = after_total_cash
            x.at[row_i, "open_risk_fraction_after_v25"] = after_total_cash / equity if equity > 0 else np.inf
            x.at[row_i, "open_directional_risk_cash_after_v25"] = after_dir_cash[side]
            x.at[row_i, "open_directional_risk_fraction_after_v25"] = after_dir_cash[side] / equity if equity > 0 else np.inf

        # New entries may not make an inherited overage worse. If the bucket was
        # already above its percentage cap solely because realized equity fell,
        # all new risk in that constrained bucket must be zero until exposure exits.
        tol = 1e-12
        aggregate_ceiling = max(open_total_cash, b.aggregate_open_risk_cap * equity)
        if after_total_cash > aggregate_ceiling + tol:
            raise AssertionError("v0.25 aggregate cost-adjusted risk budget worsened")
        for side in (-1, 1):
            directional_ceiling = max(open_dir_cash[side], b.directional_open_risk_cap * equity)
            if after_dir_cash[side] > directional_ceiling + tol:
                raise AssertionError("v0.25 directional cost-adjusted risk budget worsened")

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
        "allocation_contract": "same-timestamp entries share cost-adjusted aggregate and directional stop budgets pro-rata; inherited overage freezes new risk",
        "live_execution_authorized": False,
    }
    return x
