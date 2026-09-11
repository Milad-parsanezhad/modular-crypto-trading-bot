from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from research_bot.ml_framework_v23r import (
    make_preprocessor,
    probability_score,
    select_feature_columns,
    supervised_model_registry,
)
from research_bot.multitimeframe_strategies_v19 import moving_block_mean_ci
from research_bot.strategy_meta_v24 import economic_metrics


@dataclass(frozen=True)
class V24BContract:
    """Frozen engineering/research contract for the v0.24b follow-up.

    v0.24 test results were already observed before this experiment was designed.
    Therefore the repeated CoinEx terminal segment is explicitly a SHADOW segment
    and can never authorize scientific promotion.  Fresh forward time or a truly
    untouched external venue is required after this engineering stage.
    """

    risk_per_trade: float = 0.0025
    min_development_events: int = 300
    min_validation_events: int = 80
    min_shadow_events: int = 100
    min_validation_selected: int = 25
    min_shadow_selected: int = 25
    threshold_quantiles: tuple[float, ...] = (0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90)
    seeds: tuple[int, ...] = (314, 2718, 1618)
    bootstrap_samples: int = 750
    bootstrap_block: int = 20

    # Portfolio overlap/risk controls.  Costs are already embedded in r_multiple.
    max_concurrent_positions: int = 5
    max_open_risk_fraction: float = 0.0100
    max_strategy_open_risk_fraction: float = 0.0050
    max_directional_open_risk_fraction: float = 0.0075
    hard_realized_drawdown_kill: float = 0.05

    reused_shadow_test: bool = True
    forward_paper_authorized: bool = False
    paper_replacement_authorized: bool = False
    live_execution_authorized: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def _pf_effective(value: float, cap: float = 5.0) -> float:
    v = float(value)
    if np.isposinf(v):
        return float(cap)
    if np.isfinite(v):
        return float(min(v, cap))
    return np.nan


def family_sample_table(dataset: pd.DataFrame, contract: V24BContract | None = None) -> pd.DataFrame:
    c = contract or V24BContract()
    rows: list[dict] = []
    for strategy, g in dataset.groupby("strategy", sort=True):
        counts = g["segment"].value_counts().to_dict()
        dev = int(counts.get("development", 0))
        val = int(counts.get("validation", 0))
        shadow = int(counts.get("test", 0))
        eligible = bool(
            dev >= c.min_development_events
            and val >= c.min_validation_events
            and shadow >= c.min_shadow_events
        )
        rows.append({
            "strategy": strategy,
            "development_events": dev,
            "validation_events": val,
            "shadow_events": shadow,
            "eligible_by_counts_only": eligible,
            "status": "FAMILY_MODEL_ELIGIBLE" if eligible else "DATA_INSUFFICIENT",
        })
    return pd.DataFrame(rows)


def choose_family_threshold(
    validation: pd.DataFrame,
    score: Iterable[float],
    contract: V24BContract | None = None,
) -> tuple[float, dict]:
    """Validation-only threshold selection for one strategy family."""
    c = contract or V24BContract()
    p = np.asarray(list(score), dtype=float)
    if len(p) != len(validation):
        raise ValueError("score length mismatch")
    base = economic_metrics(validation, risk_per_trade=c.risk_per_trade)
    best_t = 1.0
    best = {
        "validation_objective": -np.inf,
        "base": base,
        "filtered": economic_metrics(validation.iloc[0:0], risk_per_trade=c.risk_per_trade),
    }
    for q in c.threshold_quantiles:
        t = float(np.quantile(p, q))
        selected = p >= t
        filt = economic_metrics(validation, selected, risk_per_trade=c.risk_per_trade)
        if int(filt["selected"]) < c.min_validation_selected:
            continue
        mean_r = float(filt["mean_r"])
        if not np.isfinite(mean_r):
            continue
        filt_pf = _pf_effective(float(filt["profit_factor"]))
        base_pf = _pf_effective(float(base["profit_factor"]))
        if not np.isfinite(filt_pf):
            continue
        if not np.isfinite(base_pf):
            base_pf = 0.0
        uplift_r = mean_r - float(base["mean_r"])
        uplift_pf = filt_pf - base_pf
        dd_penalty = max(0.0, abs(float(filt["max_drawdown"])) - abs(float(base["max_drawdown"])))
        objective = uplift_r * np.sqrt(max(int(filt["selected"]), 1)) + 0.12 * uplift_pf - 0.75 * dd_penalty
        if objective > float(best["validation_objective"]):
            best_t = t
            best = {
                "validation_objective": float(objective),
                "base": base,
                "filtered": filt,
                "threshold_quantile": float(q),
                "uplift_mean_r": float(uplift_r),
                "uplift_profit_factor": float(uplift_pf),
            }
    return best_t, best


def _family_uplift_ci(rows: pd.DataFrame, selected: np.ndarray, contract: V24BContract) -> tuple[float, float]:
    base = contract.risk_per_trade * pd.to_numeric(rows["r_multiple"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    filtered = np.where(selected, base, 0.0)
    return moving_block_mean_ci(
        filtered - base,
        samples=contract.bootstrap_samples,
        block=min(contract.bootstrap_block, max(2, len(rows) // 4)),
        seed=2401,
    )


def fit_family_models(
    dataset: pd.DataFrame,
    contract: V24BContract | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Pipeline]]:
    """Fit a separate meta-model for every sample-eligible strategy family.

    Family eligibility is based only on pre-frozen sample counts. Model, seed and
    threshold are chosen on validation. The already-observed terminal test segment
    is reported as SHADOW only and is never used for selection or refit.
    """
    c = contract or V24BContract()
    samples = family_sample_table(dataset, c)
    eligible = samples.loc[samples["eligible_by_counts_only"], "strategy"].tolist()
    leaderboard_rows: list[dict] = []
    ablation_rows: list[dict] = []
    prediction_parts: list[pd.DataFrame] = []
    champions: dict[str, Pipeline] = {}

    for strategy in eligible:
        fam = dataset[dataset["strategy"] == strategy].copy()
        dev = fam[fam["segment"] == "development"].reset_index(drop=True)
        val = fam[fam["segment"] == "validation"].reset_index(drop=True)
        shadow = fam[fam["segment"] == "test"].reset_index(drop=True)
        numeric, categorical = select_feature_columns(dev, include_context=True, include_identity=False)
        # Strategy-parameter descriptors are constant inside a family and add no
        # within-family information. Remove them to avoid meaningless coefficients.
        numeric = [x for x in numeric if not x.startswith("f_strategy_")]
        xcols = numeric + categorical
        search_rows: list[dict] = []

        for seed in c.seeds:
            for name, estimator in supervised_model_registry(seed).items():
                pipe = Pipeline([
                    ("prep", make_preprocessor(numeric, categorical)),
                    ("model", clone(estimator)),
                ])
                try:
                    pipe.fit(dev[xcols], dev["label_meta_execute"].astype(int))
                    val_score = probability_score(pipe, val[xcols])
                    threshold, validation_result = choose_family_threshold(val, val_score, c)
                    search_rows.append({
                        "strategy": strategy,
                        "model": name,
                        "seed": int(seed),
                        "status": "ok",
                        "threshold": float(threshold),
                        "validation_objective": float(validation_result["validation_objective"]),
                        "validation_selected": int(validation_result["filtered"]["selected"]),
                        "validation_mean_r": float(validation_result["filtered"]["mean_r"]),
                        "validation_profit_factor": float(validation_result["filtered"]["profit_factor"]),
                        "validation_total_return": float(validation_result["filtered"]["total_return"]),
                        "validation_max_drawdown": float(validation_result["filtered"]["max_drawdown"]),
                    })
                except Exception as exc:
                    search_rows.append({
                        "strategy": strategy,
                        "model": name,
                        "seed": int(seed),
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    })

        leaderboard_rows.extend(search_rows)
        ok = pd.DataFrame(search_rows)
        ok = ok[(ok["status"] == "ok") & np.isfinite(pd.to_numeric(ok.get("validation_objective"), errors="coerce"))].copy()
        if ok.empty:
            ablation_rows.append({
                "strategy": strategy,
                "status": "FAMILY_MODEL_FAILED",
                "shadow_is_scientifically_untouched": False,
            })
            continue

        champion_row = ok.sort_values(
            ["validation_objective", "validation_selected", "model", "seed"],
            ascending=[False, False, True, True],
        ).iloc[0]
        champion_name = str(champion_row["model"])
        champion_seed = int(champion_row["seed"])
        threshold = float(champion_row["threshold"])
        estimator = supervised_model_registry(champion_seed)[champion_name]
        champion = Pipeline([
            ("prep", make_preprocessor(numeric, categorical)),
            ("model", clone(estimator)),
        ])
        # Deliberately development-only: validation selected the frozen model/
        # threshold and is not folded back into the fit before the shadow read.
        champion.fit(dev[xcols], dev["label_meta_execute"].astype(int))
        champions[strategy] = champion

        shadow_score = probability_score(champion, shadow[xcols])
        selected = shadow_score >= threshold
        base = economic_metrics(shadow, risk_per_trade=c.risk_per_trade)
        filt = economic_metrics(shadow, selected, risk_per_trade=c.risk_per_trade)
        ci_low, ci_high = _family_uplift_ci(shadow, selected, c)
        ablation_rows.append({
            "strategy": strategy,
            "status": "SHADOW_EVALUATED",
            "champion_model": champion_name,
            "champion_seed": champion_seed,
            "frozen_threshold": threshold,
            "development_events": int(len(dev)),
            "validation_events": int(len(val)),
            "shadow_events": int(len(shadow)),
            **{f"base_{k}": v for k, v in base.items()},
            **{f"filtered_{k}": v for k, v in filt.items()},
            "uplift_total_return": float(filt["total_return"]) - float(base["total_return"]),
            "uplift_mean_r": float(filt["mean_r"]) - float(base["mean_r"]) if np.isfinite(float(filt["mean_r"])) else np.nan,
            "paired_block_uplift_ci_low": ci_low,
            "paired_block_uplift_ci_high": ci_high,
            "shadow_is_scientifically_untouched": False,
            "test_used_for_model_or_threshold_selection": False,
            "scientific_promotion_authorized": False,
        })
        pred = shadow[[
            "strategy", "symbol", "signal_time", "entry_time", "exit_time", "side",
            "r_multiple", "entry", "stop", "target", "exit_reason",
        ]].copy()
        pred["family_meta_score"] = shadow_score
        pred["family_meta_selected"] = selected
        pred["family_champion_model"] = champion_name
        pred["family_frozen_threshold"] = threshold
        prediction_parts.append(pred)

    leaderboard = pd.DataFrame(leaderboard_rows)
    ablation = pd.DataFrame(ablation_rows)
    predictions = pd.concat(prediction_parts, ignore_index=True) if prediction_parts else pd.DataFrame()
    return leaderboard, ablation, predictions, champions


def portfolio_overlap_backtest(
    rows: pd.DataFrame,
    selected: Iterable[bool] | None = None,
    *,
    contract: V24BContract | None = None,
    mode: str = "base",
) -> tuple[dict, pd.DataFrame]:
    """Realized-equity overlap-aware portfolio risk simulation.

    The engine allocates risk at entry, keeps overlapping positions open until their
    recorded causal bracket exit, and enforces portfolio/symbol/strategy/directional
    risk caps before admitting a new position.  It does not mark positions to market
    between entry and exit, so the drawdown statistic is realized-equity drawdown,
    not intrabar MTM drawdown.
    """
    c = contract or V24BContract()
    if rows.empty:
        return {
            "mode": mode, "events": 0, "accepted": 0, "total_return": 0.0,
            "max_realized_drawdown": 0.0, "profit_factor": np.nan,
            "max_concurrent": 0, "hard_kill_triggered": False,
        }, pd.DataFrame()

    x = rows.copy().reset_index(drop=True)
    for col in ("entry_time", "exit_time", "signal_time"):
        if col in x:
            x[col] = pd.to_datetime(x[col], utc=True)
    mask = np.ones(len(x), dtype=bool) if selected is None else np.asarray(list(selected), dtype=bool)
    if len(mask) != len(x):
        raise ValueError("portfolio selection mask length mismatch")
    x["_selected"] = mask
    # Priority is deliberately independent of realized outcomes and ML score.
    x = x.sort_values(["entry_time", "strategy", "symbol"], kind="mergesort").reset_index(drop=True)

    equity = 1.0
    peak = 1.0
    min_dd = 0.0
    hard_kill = False
    active: list[dict] = []
    ledger: list[dict] = []
    pnl_realized: list[float] = []
    r_accepted: list[float] = []
    max_concurrent = 0
    max_open_risk_fraction_seen = 0.0

    def close_due(now: pd.Timestamp | None) -> None:
        nonlocal equity, peak, min_dd, active, hard_kill
        due = [p for p in active if now is None or p["exit_time"] <= now]
        if not due:
            return
        due = sorted(due, key=lambda p: (p["exit_time"], p["strategy"], p["symbol"]))
        remaining = [p for p in active if p not in due]
        for p in due:
            pnl = float(p["risk_amount"] * p["r_multiple"])
            equity += pnl
            pnl_realized.append(pnl)
            peak = max(peak, equity)
            dd = equity / peak - 1.0
            min_dd = min(min_dd, dd)
            if dd <= -c.hard_realized_drawdown_kill:
                hard_kill = True
            p["realized_pnl"] = pnl
            p["equity_after_exit"] = equity
        active = remaining

    for _, row in x.iterrows():
        now = row["entry_time"]
        close_due(now)
        reason = ""
        if not bool(row["_selected"]):
            reason = "META_FILTER_REJECT"
        elif hard_kill:
            reason = "HARD_REALIZED_DRAWDOWN_KILL"
        elif any(p["symbol"] == str(row["symbol"]) for p in active):
            reason = "SYMBOL_OVERLAP_CAP"
        elif len(active) >= c.max_concurrent_positions:
            reason = "MAX_CONCURRENT_POSITIONS"
        else:
            planned = equity * c.risk_per_trade
            total_open = sum(float(p["risk_amount"]) for p in active)
            strategy_open = sum(float(p["risk_amount"]) for p in active if p["strategy"] == str(row["strategy"]))
            directional_open = sum(float(p["risk_amount"]) for p in active if int(p["side"]) == int(row["side"]))
            if total_open + planned > equity * c.max_open_risk_fraction + 1e-12:
                reason = "PORTFOLIO_OPEN_RISK_CAP"
            elif strategy_open + planned > equity * c.max_strategy_open_risk_fraction + 1e-12:
                reason = "STRATEGY_OPEN_RISK_CAP"
            elif directional_open + planned > equity * c.max_directional_open_risk_fraction + 1e-12:
                reason = "DIRECTIONAL_OPEN_RISK_CAP"
            else:
                position = {
                    "strategy": str(row["strategy"]),
                    "symbol": str(row["symbol"]),
                    "side": int(row["side"]),
                    "entry_time": now,
                    "exit_time": row["exit_time"],
                    "r_multiple": float(row["r_multiple"]),
                    "risk_amount": float(planned),
                    "entry_equity": float(equity),
                }
                active.append(position)
                r_accepted.append(float(row["r_multiple"]))
                max_concurrent = max(max_concurrent, len(active))
                total_after = sum(float(p["risk_amount"]) for p in active)
                max_open_risk_fraction_seen = max(max_open_risk_fraction_seen, total_after / max(equity, 1e-12))

        ledger.append({
            "mode": mode,
            "strategy": str(row["strategy"]),
            "symbol": str(row["symbol"]),
            "signal_time": row.get("signal_time"),
            "entry_time": now,
            "exit_time": row["exit_time"],
            "side": int(row["side"]),
            "r_multiple": float(row["r_multiple"]),
            "selected_input": bool(row["_selected"]),
            "accepted": reason == "",
            "reject_reason": reason,
            "equity_at_decision": float(equity),
            "open_positions_after_decision": int(len(active)),
        })

    close_due(None)
    accepted = int(sum(bool(z["accepted"]) for z in ledger))
    wins = float(sum(p for p in pnl_realized if p > 0))
    losses = float(-sum(p for p in pnl_realized if p < 0))
    pf = wins / losses if losses > 0 else (np.inf if wins > 0 else np.nan)
    reject_counts = pd.Series([z["reject_reason"] for z in ledger if z["reject_reason"]]).value_counts().to_dict()
    summary = {
        "mode": mode,
        "events": int(len(x)),
        "selected_input": int(mask.sum()),
        "accepted": accepted,
        "acceptance_fraction": float(accepted / len(x)),
        "total_return": float(equity - 1.0),
        "ending_equity": float(equity),
        "max_realized_drawdown": float(min_dd),
        "profit_factor": float(pf),
        "mean_r_accepted": float(np.mean(r_accepted)) if r_accepted else np.nan,
        "max_concurrent": int(max_concurrent),
        "max_open_risk_fraction_seen": float(max_open_risk_fraction_seen),
        "hard_kill_triggered": bool(hard_kill),
        "reject_counts": reject_counts,
        "drawdown_semantics": "realized-equity only; not intrabar mark-to-market",
    }
    return summary, pd.DataFrame(ledger)


def merge_family_predictions(shadow: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """Attach family selections to shadow rows; unsupported families default reject."""
    keys = ["strategy", "symbol", "signal_time"]
    x = shadow.copy()
    x["signal_time"] = pd.to_datetime(x["signal_time"], utc=True)
    if predictions.empty:
        x["family_meta_selected"] = False
        x["family_meta_score"] = np.nan
        return x
    p = predictions.copy()
    p["signal_time"] = pd.to_datetime(p["signal_time"], utc=True)
    keep = keys + ["family_meta_selected", "family_meta_score", "family_champion_model", "family_frozen_threshold"]
    x = x.merge(p[keep], on=keys, how="left", validate="one_to_one")
    x["family_meta_selected"] = x["family_meta_selected"].fillna(False).astype(bool)
    return x
