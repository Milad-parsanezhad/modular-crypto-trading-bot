from __future__ import annotations

"""v0.31 pre-entry drawdown firewall.

v0.30 produced several economically positive CoinEx+OKX candidates that failed
primarily because realized drawdown crossed the unchanged 5% hard gate. v0.31
therefore changes *risk admission*, not alpha parameters or qualification
thresholds.

The firewall reserves enough realized-equity headroom so that the aggregate
cost-adjusted loss-at-stop of all currently open positions cannot, at entry
time, push equity below the frozen hard-drawdown floor relative to the realized
peak. The existing v0.25 2.0% aggregate and 1.5% same-direction caps remain in
force. KuCoin remains untouched until a CoinEx+OKX winner is locked.

Research/PAPER only. Live execution is not authorized.
"""

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from research_bot.multitimeframe_strategies_v20 import COOLDOWN, RiskPsychologyPolicy
from research_bot.portfolio_allocator_v25 import (
    INTRADAY_TIMEFRAMES,
    PendingPositionV25,
    PortfolioRiskBudgetV25,
    _cost_adjusted_stop_multiple,
    _open_risk_state,
    _settle_before,
)


PRIOR_EFFECTIVE_TRIALS_V31 = 96
NEW_RISK_VARIANTS_V31 = 12
TOTAL_EFFECTIVE_TRIALS_V31 = PRIOR_EFFECTIVE_TRIALS_V31 + NEW_RISK_VARIANTS_V31


@dataclass(frozen=True)
class DrawdownFirewallV31:
    hard_drawdown_cap: float = 0.050

    def __post_init__(self) -> None:
        if not (0.0 < self.hard_drawdown_cap < 1.0):
            raise ValueError("hard_drawdown_cap must lie in (0, 1)")


def preregistration_manifest_v31() -> dict[str, Any]:
    return {
        "version": "v0.31",
        "experiment": "PRE_ENTRY_HARD_DRAWDOWN_FIREWALL",
        "alpha_registry": "frozen_v0.30_12_candidates",
        "candidate_count": NEW_RISK_VARIANTS_V31,
        "prior_effective_trials": PRIOR_EFFECTIVE_TRIALS_V31,
        "new_risk_variants": NEW_RISK_VARIANTS_V31,
        "total_effective_trials": TOTAL_EFFECTIVE_TRIALS_V31,
        "hard_drawdown_cap": 0.05,
        "aggregate_open_risk_cap": 0.02,
        "directional_open_risk_cap": 0.015,
        "round_trip_cost_fraction": 0.0024,
        "development_venues": ["coinex_consumed", "okx_consumed"],
        "reserved_holdout_venue": "kucoin",
        "alpha_parameter_retuning": False,
        "qualification_threshold_relaxation": False,
        "winner_reselection_after_holdout": False,
        "historical_test_recycling": False,
        "live_execution_authorized": False,
    }


def _drawdown_headroom_cash(*, equity: float, peak: float, cap: float) -> float:
    if not np.isfinite(equity) or not np.isfinite(peak) or equity <= 0 or peak <= 0:
        return 0.0
    floor = float(peak * (1.0 - cap))
    return float(max(0.0, equity - floor))


def _allocate_with_firewall(
    proposals: pd.DataFrame,
    *,
    equity: float,
    peak: float,
    open_total_risk_cash: float,
    open_directional_risk_cash: dict[int, float],
    budget: PortfolioRiskBudgetV25,
    firewall: DrawdownFirewallV31,
) -> pd.Series:
    if proposals.empty or equity <= 0 or not np.isfinite(equity):
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

    aggregate_cap_cash = float(budget.aggregate_open_risk_cap * equity)
    aggregate_available = max(0.0, aggregate_cap_cash - float(open_total_risk_cash))
    headroom_cash = _drawdown_headroom_cash(equity=equity, peak=peak, cap=firewall.hard_drawdown_cap)
    firewall_available = max(0.0, headroom_cash - float(open_total_risk_cash))
    available_total = min(aggregate_available, firewall_available)

    wanted_total = float(allocated.sum())
    scale = min(1.0, available_total / wanted_total) if wanted_total > 0 else 1.0
    return (allocated * scale).clip(lower=0.0)


def apply_drawdown_firewall_v31(
    ledger: pd.DataFrame,
    spec: Any,
    *,
    risk_policy: RiskPsychologyPolicy | None = None,
    budget: PortfolioRiskBudgetV25 | None = None,
    firewall: DrawdownFirewallV31 | None = None,
) -> pd.DataFrame:
    """Apply v0.25 portfolio limits plus a causal pre-entry 5% DD firewall."""
    p = risk_policy or RiskPsychologyPolicy()
    b = budget or PortfolioRiskBudgetV25()
    fw = firewall or DrawdownFirewallV31(hard_drawdown_cap=p.hard_drawdown_kill)
    timeframe = str(getattr(spec, "timeframe", spec))
    if timeframe not in COOLDOWN:
        raise ValueError(f"unsupported timeframe for v0.31 allocator: {timeframe!r}")
    if abs(fw.hard_drawdown_cap - p.hard_drawdown_kill) > 1e-12:
        raise ValueError("v0.31 firewall must equal the frozen hard drawdown gate")
    if ledger.empty:
        return ledger.copy()

    required = {"symbol", "entry_time", "exit_time", "r_multiple", "side", "entry", "stop"}
    missing = required - set(ledger.columns)
    if missing:
        raise ValueError(f"v0.31 allocator missing required columns: {sorted(missing)}")

    x = ledger.sort_values(["entry_time", "symbol"], kind="mergesort").reset_index(drop=True).copy()
    x["entry_time"] = pd.to_datetime(x["entry_time"], utc=True, errors="raise")
    x["exit_time"] = pd.to_datetime(x["exit_time"], utc=True, errors="raise")
    x["side"] = pd.to_numeric(x["side"], errors="raise").astype(int)
    if bool((~x["side"].isin([-1, 1])).any()):
        raise ValueError("v0.31 allocator requires side in {-1,+1}")
    if bool((x["exit_time"] < x["entry_time"]).any()):
        raise ValueError("exit_time precedes entry_time in v0.31 ledger")

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
        "drawdown_floor_equity_v31": np.nan,
        "drawdown_headroom_cash_before_v31": 0.0,
        "firewall_available_cash_before_v31": 0.0,
        "firewall_scale_v31": 0.0,
        "firewall_constrained_v31": False,
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
        headroom_cash = _drawdown_headroom_cash(equity=equity, peak=peak, cap=fw.hard_drawdown_cap)
        firewall_available = max(0.0, headroom_cash - open_total_cash)
        floor_equity = float(peak * (1.0 - fw.hard_drawdown_cap))
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
            x.at[row_i, "drawdown_floor_equity_v31"] = floor_equity
            x.at[row_i, "drawdown_headroom_cash_before_v31"] = headroom_cash
            x.at[row_i, "firewall_available_cash_before_v31"] = firewall_available

            key = (str(row["symbol"]), entry_time.date())
            reason = ""
            if hard_killed or not np.isfinite(equity) or equity <= 0 or dd <= -p.hard_drawdown_kill:
                hard_killed = True
                reason = "HARD_DRAWDOWN_KILL"
            elif firewall_available <= 1e-15:
                reason = "DRAWDOWN_FIREWALL_EXHAUSTED"
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
            allocated = _allocate_with_firewall(
                proposal_frame,
                equity=equity,
                peak=peak,
                open_total_risk_cash=open_total_cash,
                open_directional_risk_cash=open_dir_cash,
                budget=b,
                firewall=fw,
            )
            desired_total = float(proposal_frame["proposed_stop_loss_budget_cash_v25"].sum())
            allocated_total = float(allocated.sum())
            batch_firewall_scale = allocated_total / desired_total if desired_total > 0 else 0.0

            new_positions: list[PendingPositionV25] = []
            for row_i in proposals:
                row = x.loc[row_i]
                side = int(row["side"])
                proposed_stop = float(row["proposed_stop_loss_budget_cash_v25"])
                stop_multiple = float(row["stop_loss_multiple_abs_v25"])
                allocated_stop = float(allocated.loc[row_i])
                x.at[row_i, "firewall_scale_v31"] = batch_firewall_scale
                x.at[row_i, "firewall_constrained_v31"] = bool(batch_firewall_scale < 1.0 - 1e-12)
                if allocated_stop <= 1e-15:
                    x.at[row_i, "reject_reason_v25"] = "DRAWDOWN_FIREWALL_EXHAUSTED" if firewall_available <= 1e-15 else "PORTFOLIO_RISK_BUDGET_EXHAUSTED"
                    continue

                nominal_risk_cash = float(allocated_stop / stop_multiple)
                risk_fraction = float(nominal_risk_cash / equity)
                scale = float(allocated_stop / proposed_stop) if proposed_stop > 0 else 0.0
                account_return = float(risk_fraction * float(row["r_multiple"]))
                if not np.isfinite(account_return) or account_return <= -1.0:
                    raise ValueError("invalid v0.31 account return")
                pnl_cash = float(equity * account_return)

                x.at[row_i, "executed_v25"] = True
                x.at[row_i, "allocated_stop_loss_budget_cash_v25"] = allocated_stop
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
                        stop_loss_budget_cash=allocated_stop,
                        side=side,
                    )
                )
                sequence += 1
            pending.extend(new_positions)

        after_total, after_dir = _open_risk_state(pending)
        for row_i in group_indices:
            side = int(x.at[row_i, "side"])
            x.at[row_i, "open_risk_cash_after_v25"] = after_total
            x.at[row_i, "open_risk_fraction_after_v25"] = after_total / equity if equity > 0 else np.inf
            x.at[row_i, "open_directional_risk_cash_after_v25"] = after_dir[side]
            x.at[row_i, "open_directional_risk_fraction_after_v25"] = after_dir[side] / equity if equity > 0 else np.inf

        tol = 1e-12
        aggregate_ceiling = max(open_total_cash, b.aggregate_open_risk_cap * equity)
        if after_total > aggregate_ceiling + tol:
            raise AssertionError("v0.31 aggregate risk budget worsened")
        for side in (-1, 1):
            directional_ceiling = max(open_dir_cash[side], b.directional_open_risk_cap * equity)
            if after_dir[side] > directional_ceiling + tol:
                raise AssertionError("v0.31 directional risk budget worsened")

        # The new invariant: if the account was not already in inherited
        # overage, new entries may not reserve more stop loss than the remaining
        # realized-equity drawdown headroom.
        firewall_ceiling = max(open_total_cash, headroom_cash)
        if after_total > firewall_ceiling + tol:
            raise AssertionError("v0.31 drawdown headroom worsened")

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
        raise AssertionError("v0.31 settlement flush left pending positions")
    return x


def allocator_diagnostics_v31(attempts: pd.DataFrame) -> dict[str, Any]:
    if attempts.empty:
        return {
            "firewall_scaled_trades": 0,
            "firewall_zero_headroom_rejections": 0,
            "minimum_settlement_drawdown": np.nan,
        }
    executed = attempts[attempts["executed_v25"] == True]  # noqa: E712
    return {
        "firewall_scaled_trades": int(executed.get("firewall_constrained_v31", pd.Series(dtype=bool)).fillna(False).sum()),
        "firewall_zero_headroom_rejections": int(attempts["reject_reason_v25"].astype(str).eq("DRAWDOWN_FIREWALL_EXHAUSTED").sum()),
        "minimum_settlement_drawdown": float(pd.to_numeric(executed.get("settlement_drawdown_v25", pd.Series(dtype=float)), errors="coerce").min()) if len(executed) else np.nan,
    }
