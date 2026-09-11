from __future__ import annotations

"""v0.38 soft constrained portfolio allocation over the frozen v0.36 HistGB event pool.

Research only. CoinEx/OKX/KuCoin are consumed development evidence. Kraken is
sealed. v0.38 replaces v0.37 hard top-k admission with causal fractional risk
allocation; it does not alter the event model or frozen scientific gates.
"""

from dataclasses import asdict, dataclass
from typing import Any, Iterable
import math

import numpy as np
import pandas as pd


PARENT_V36 = "V36_EXPECTED_NETR_HISTGB"
DEVELOPMENT_VENUES_V38 = ("coinex_consumed", "okx_consumed", "kucoin_consumed")
RESERVED_HOLDOUT_VENUE_V38 = "kraken"
PRIOR_EFFECTIVE_TRIALS_V38 = 132
NEW_TRIALS_V38 = 1
TOTAL_EFFECTIVE_TRIALS_V38 = PRIOR_EFFECTIVE_TRIALS_V38 + NEW_TRIALS_V38


@dataclass(frozen=True)
class SoftRiskPolicyV38:
    name: str = "SOFT_CAUSAL_WATERFILL"
    aggregate_open_risk_cap: float = 0.020
    directional_open_risk_cap: float = 0.015
    per_symbol_risk_cap: float = 0.005
    per_event_risk_cap: float = 0.0025
    minimum_risk: float = 0.00025
    hard_drawdown_cap: float = 0.050
    drawdown_warn_1: float = 0.020
    drawdown_warn_2: float = 0.035
    drawdown_scale_1: float = 0.75
    drawdown_scale_2: float = 0.50
    drawdown_scale_3: float = 0.25
    softmax_temperature: float = 1.0


POLICY_V38 = SoftRiskPolicyV38()


def preregistration_manifest_v38() -> dict[str, Any]:
    return {
        "version": "v0.38",
        "experiment": "SOFT_CONSTRAINED_PORTFOLIO_ALLOCATION",
        "parent_v36": PARENT_V36,
        "development_venues": list(DEVELOPMENT_VENUES_V38),
        "reserved_holdout_venue": RESERVED_HOLDOUT_VENUE_V38,
        "prior_effective_trials": PRIOR_EFFECTIVE_TRIALS_V38,
        "new_trials": NEW_TRIALS_V38,
        "total_effective_trials": TOTAL_EFFECTIVE_TRIALS_V38,
        "policy": asdict(POLICY_V38),
        "input": "v0.36 selected HistGB events only; event scores frozen",
        "allocation": "causal batch softmax + capped proportional water filling",
        "settlement_rule": "exit_time < next entry_time; same-time exit remains open",
        "one_open_position_per_symbol": True,
        "min_trades_per_venue": 200,
        "min_profit_factor": 1.05,
        "min_expectancy_r": 0.0,
        "min_positive_asset_fraction": 0.60,
        "min_block_ci_low": 0.0,
        "max_drawdown": 0.05,
        "kraken_touched": False,
        "threshold_relaxation": False,
        "post_result_retuning": False,
        "paper_replacement_authorized": False,
        "live_execution_authorized": False,
    }


def _drawdown_scale(dd: float, p: SoftRiskPolicyV38) -> float:
    depth = abs(min(0.0, dd))
    if depth >= p.hard_drawdown_cap:
        return 0.0
    if depth >= p.drawdown_warn_2:
        return p.drawdown_scale_3
    if depth >= p.drawdown_warn_1:
        # Two deterministic bands inside the warning zone, without tuning.
        mid = 0.5 * (p.drawdown_warn_1 + p.drawdown_warn_2)
        return p.drawdown_scale_2 if depth >= mid else p.drawdown_scale_1
    return 1.0


def _waterfill(weights: np.ndarray, budget: float, cap: float) -> np.ndarray:
    """Capped proportional allocation; deterministic and order independent."""
    w = np.asarray(weights, dtype=float)
    out = np.zeros(len(w), dtype=float)
    if len(w) == 0 or budget <= 0 or cap <= 0:
        return out
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    if float(w.sum()) <= 0:
        w[:] = 1.0
    remaining = np.ones(len(w), dtype=bool)
    left = float(budget)
    for _ in range(len(w) + 1):
        idx = np.flatnonzero(remaining)
        if len(idx) == 0 or left <= 1e-15:
            break
        ww = w[idx]
        if float(ww.sum()) <= 0:
            ww = np.ones(len(idx), dtype=float)
        proposal = left * ww / float(ww.sum())
        hit = proposal >= cap - 1e-15
        if not hit.any():
            out[idx] += proposal
            left = 0.0
            break
        hit_idx = idx[hit]
        for j in hit_idx:
            add = max(0.0, cap - out[j])
            out[j] += add
            left -= add
            remaining[j] = False
        if left <= 1e-15:
            break
    return out


def _settle_before(
    pending: list[dict[str, Any]],
    t: pd.Timestamp,
    equity: float,
    peak: float,
) -> tuple[list[dict[str, Any]], float, float]:
    due = [p for p in pending if pd.Timestamp(p["exit_time"]) < t]
    keep = [p for p in pending if pd.Timestamp(p["exit_time"]) >= t]
    if due:
        frame = pd.DataFrame(due)
        frame["exit_time"] = pd.to_datetime(frame["exit_time"], utc=True)
        for _, g in frame.groupby("exit_time", sort=True):
            batch_ret = float(pd.to_numeric(g["account_return_v38"], errors="coerce").fillna(0).sum())
            equity *= max(1e-12, 1.0 + batch_ret)
            peak = max(peak, equity)
    return keep, equity, peak


def soft_allocate_v38(scored_events: pd.DataFrame, policy: SoftRiskPolicyV38 = POLICY_V38) -> tuple[pd.DataFrame, dict[str, Any]]:
    if scored_events.empty:
        return scored_events.copy(), {"input_selected": 0, "executed": 0}
    x = scored_events.copy()
    if "selected_v36" in x:
        x = x.loc[x["selected_v36"].astype(bool)].copy()
    for c in ("signal_time", "entry_time", "exit_time"):
        x[c] = pd.to_datetime(x[c], utc=True, errors="raise")
    x = x.sort_values(["entry_time", "symbol", "base_strategy_v36"], kind="mergesort").reset_index(drop=True)
    x["executed_v38"] = False
    x["risk_fraction_v38"] = 0.0
    x["account_return_v38"] = 0.0
    x["reject_reason_v38"] = ""
    x["equity_before_v38"] = np.nan
    x["drawdown_before_v38"] = np.nan
    x["soft_weight_v38"] = np.nan

    pending: list[dict[str, Any]] = []
    equity = peak = 1.0
    duplicate_rejections = 0
    min_risk_rejections = 0
    dd_freezes = 0

    for entry_time, group in x.groupby("entry_time", sort=True):
        pending, equity, peak = _settle_before(pending, pd.Timestamp(entry_time), equity, peak)
        dd = equity / peak - 1.0
        scale = _drawdown_scale(dd, policy)
        open_risk = float(sum(float(p["risk_fraction_v38"]) for p in pending))
        open_by_side = {
            1: float(sum(float(p["risk_fraction_v38"]) for p in pending if int(p["side"]) > 0)),
            -1: float(sum(float(p["risk_fraction_v38"]) for p in pending if int(p["side"]) < 0)),
        }
        open_symbols = {str(p["symbol"]) for p in pending}
        headroom = max(0.0, policy.hard_drawdown_cap + dd)
        aggregate_budget = max(0.0, min(policy.aggregate_open_risk_cap - open_risk, headroom)) * scale
        if aggregate_budget <= 1e-15:
            for idx in group.index:
                x.at[idx, "reject_reason_v38"] = "DRAWDOWN_OR_PORTFOLIO_BUDGET"
                x.at[idx, "equity_before_v38"] = equity
                x.at[idx, "drawdown_before_v38"] = dd
            dd_freezes += int(len(group))
            continue

        # One event per symbol per batch; retain the highest frozen v0.36 utility.
        eligible_idx: list[int] = []
        for symbol, sg in group.groupby("symbol", sort=True):
            if str(symbol) in open_symbols:
                for idx in sg.index:
                    x.at[idx, "reject_reason_v38"] = "SYMBOL_ALREADY_OPEN"
                    x.at[idx, "equity_before_v38"] = equity
                    x.at[idx, "drawdown_before_v38"] = dd
                duplicate_rejections += int(len(sg))
                continue
            util = pd.to_numeric(sg["utility_v36"], errors="coerce").fillna(-np.inf)
            chosen = int(util.idxmax())
            eligible_idx.append(chosen)
            for idx in sg.index:
                if int(idx) != chosen:
                    x.at[idx, "reject_reason_v38"] = "DUPLICATE_SYMBOL_SAME_BATCH"
                    x.at[idx, "equity_before_v38"] = equity
                    x.at[idx, "drawdown_before_v38"] = dd
                    duplicate_rejections += 1

        if not eligible_idx:
            continue

        alloc = np.zeros(len(eligible_idx), dtype=float)
        pos = {idx: j for j, idx in enumerate(eligible_idx)}
        # Allocate each side up to directional budget, then enforce aggregate cap pro-rata.
        for side in (1, -1):
            ids = [idx for idx in eligible_idx if int(x.at[idx, "side"]) == side]
            if not ids:
                continue
            side_budget = max(0.0, policy.directional_open_risk_cap - open_by_side[side]) * scale
            side_budget = min(side_budget, aggregate_budget)
            vals = pd.to_numeric(x.loc[ids, "utility_v36"], errors="coerce").to_numpy(dtype=float)
            finite = vals[np.isfinite(vals)]
            center = float(np.median(finite)) if len(finite) else 0.0
            spread = float(np.std(finite)) if len(finite) > 1 else 1.0
            spread = max(spread, 1e-6)
            logits = np.clip((vals - center) / (spread * policy.softmax_temperature), -20, 20)
            weights = np.exp(logits - np.nanmax(logits))
            a = _waterfill(weights, side_budget, min(policy.per_event_risk_cap, policy.per_symbol_risk_cap))
            for idx, amount in zip(ids, a):
                alloc[pos[idx]] = amount

        total = float(alloc.sum())
        if total > aggregate_budget + 1e-15 and total > 0:
            alloc *= aggregate_budget / total

        for idx, risk in zip(eligible_idx, alloc):
            x.at[idx, "equity_before_v38"] = equity
            x.at[idx, "drawdown_before_v38"] = dd
            x.at[idx, "soft_weight_v38"] = float(risk / total) if total > 0 else 0.0
            if risk < policy.minimum_risk - 1e-15:
                x.at[idx, "reject_reason_v38"] = "BELOW_MINIMUM_RISK"
                min_risk_rejections += 1
                continue
            r = float(pd.to_numeric(pd.Series([x.at[idx, "r_multiple"]]), errors="coerce").iloc[0])
            if not np.isfinite(r):
                x.at[idx, "reject_reason_v38"] = "NONFINITE_R"
                continue
            ret = float(risk * r)
            x.at[idx, "executed_v38"] = True
            x.at[idx, "risk_fraction_v38"] = float(risk)
            x.at[idx, "account_return_v38"] = ret
            pending.append({
                "symbol": str(x.at[idx, "symbol"]),
                "side": int(x.at[idx, "side"]),
                "exit_time": x.at[idx, "exit_time"],
                "risk_fraction_v38": float(risk),
                "account_return_v38": ret,
            })

    pending, equity, peak = _settle_before(pending, pd.Timestamp.max.tz_localize("UTC"), equity, peak)
    diag = {
        "input_selected": int(len(x)),
        "executed": int(x["executed_v38"].sum()),
        "duplicate_rejections": int(duplicate_rejections),
        "minimum_risk_rejections": int(min_risk_rejections),
        "drawdown_or_budget_freezes": int(dd_freezes),
        "mean_risk_fraction": float(pd.to_numeric(x.loc[x["executed_v38"], "risk_fraction_v38"], errors="coerce").mean()) if bool(x["executed_v38"].any()) else np.nan,
        "final_equity": float(equity),
    }
    return x, diag


def _profit_factor(values: Iterable[float]) -> float:
    a = np.asarray(list(values), dtype=float)
    a = a[np.isfinite(a)]
    wins = float(a[a > 0].sum())
    losses = float(-a[a < 0].sum())
    return wins / losses if losses > 0 else (np.inf if wins > 0 else np.nan)


def _moving_block_ci(values: Iterable[float], samples: int = 750, block: int = 20, seed: int = 314) -> tuple[float, float]:
    a = np.asarray(list(values), dtype=float)
    a = a[np.isfinite(a)]
    n = len(a)
    if n < max(30, block):
        return np.nan, np.nan
    b = min(block, n)
    starts = np.arange(0, n - b + 1)
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=float)
    need = int(math.ceil(n / b))
    for i in range(samples):
        idx: list[int] = []
        for s in rng.choice(starts, size=need, replace=True):
            idx.extend(range(int(s), int(s) + b))
        means[i] = float(np.mean(a[np.asarray(idx[:n], dtype=int)]))
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def portfolio_metrics_v38(allocated: pd.DataFrame) -> dict[str, Any]:
    z = allocated.loc[allocated.get("executed_v38", False).astype(bool)].copy() if not allocated.empty else pd.DataFrame()
    if z.empty:
        return {"trades": 0, "profit_factor": np.nan, "expectancy_r": np.nan, "positive_asset_fraction": 0.0, "block_ci_low": np.nan, "block_ci_high": np.nan, "max_drawdown": np.nan}
    z["exit_time"] = pd.to_datetime(z["exit_time"], utc=True)
    ret = pd.to_numeric(z["account_return_v38"], errors="coerce").fillna(0.0)
    risk = pd.to_numeric(z["risk_fraction_v38"], errors="coerce").fillna(0.0)
    batch = z.assign(_ret=ret).groupby("exit_time", sort=True)["_ret"].sum()
    equity = (1.0 + batch).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1.0
    by_symbol = z.assign(_ret=ret).groupby("symbol")["_ret"].sum()
    breadth = float((by_symbol > 0).mean()) if len(by_symbol) else 0.0
    lo, hi = _moving_block_ci(batch.to_numpy(dtype=float))
    denom = float(risk.sum())
    expectancy_r = float(ret.sum() / denom) if denom > 0 else np.nan
    return {
        "trades": int(len(z)),
        "profit_factor": float(_profit_factor(ret.to_numpy(dtype=float))),
        "expectancy_r": expectancy_r,
        "positive_asset_fraction": breadth,
        "block_ci_low": lo,
        "block_ci_high": hi,
        "max_drawdown": float(dd.min()) if len(dd) else 0.0,
        "total_return": float(equity.iloc[-1] - 1.0) if len(equity) else 0.0,
        "mean_risk_fraction": float(risk.mean()),
        "max_risk_fraction": float(risk.max()),
    }


def screen_three_venues_v38(metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = [metrics[v] for v in DEVELOPMENT_VENUES_V38]
    eligible = all(
        int(r.get("trades", 0)) >= 200
        and float(r.get("profit_factor", np.nan)) >= 1.05
        and float(r.get("expectancy_r", np.nan)) > 0
        and float(r.get("positive_asset_fraction", np.nan)) >= 0.60
        and abs(float(r.get("max_drawdown", np.nan))) <= 0.05
        and float(r.get("block_ci_low", np.nan)) > 0
        for r in rows
    )
    return {
        "development_eligible_v38": bool(eligible),
        "robust_min_trades_v38": min(int(r.get("trades", 0)) for r in rows),
        "robust_floor_profit_factor_v38": min(float(r.get("profit_factor", np.nan)) for r in rows),
        "robust_floor_expectancy_r_v38": min(float(r.get("expectancy_r", np.nan)) for r in rows),
        "robust_floor_breadth_v38": min(float(r.get("positive_asset_fraction", np.nan)) for r in rows),
        "robust_floor_block_ci_low_v38": min(float(r.get("block_ci_low", np.nan)) for r in rows),
        "robust_worst_drawdown_v38": max(abs(float(r.get("max_drawdown", np.nan))) for r in rows),
    }


def decision_v38(screen: dict[str, Any]) -> dict[str, Any]:
    ok = bool(screen.get("development_eligible_v38", False))
    return {
        "version": "v0.38",
        "decision": "V38_SOFT_PORTFOLIO_CANDIDATE_LOCKED_FOR_WALK_FORWARD" if ok else "NO_V38_ROBUST_SOFT_PORTFOLIO_CANDIDATE",
        "winner": POLICY_V38.name if ok else None,
        "parent_v36": PARENT_V36,
        "development_venues": list(DEVELOPMENT_VENUES_V38),
        "reserved_holdout_venue": RESERVED_HOLDOUT_VENUE_V38,
        "kraken_touched": False,
        "holdout_authorized_to_run": False,
        "paper_replacement_authorized": False,
        "live_execution_authorized": False,
    }
