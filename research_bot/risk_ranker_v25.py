from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from research_bot.ml_framework_v23r import probability_score


V25_STRATEGIES = ("H4_S6_BREAKOUT", "H4_D1_OB_BOS_RISK")


@dataclass(frozen=True)
class V25RankingContract:
    """Pre-registered v0.25 allocator-development contract.

    v0.24d revealed, post outcome-read, that the frozen event filter could retain
    positive event-level economics on two venues while the overlap-aware portfolio
    failed. Therefore all v0.24d OKX/KuCoin observations are now SPENT evidence and
    may be used only for diagnosis/development. Scientific promotion requires data
    at or after ``future_start_utc`` that did not influence this design.
    """

    future_start_utc: str = "2026-09-11T12:00:00Z"
    risk_per_trade: float = 0.0025
    max_concurrent_positions: int = 5
    max_open_risk_fraction: float = 0.0100
    max_strategy_open_risk_fraction: float = 0.0050
    max_directional_open_risk_fraction: float = 0.0075
    hard_realized_drawdown_kill: float = 0.05
    min_pair_abs_r_gap: float = 0.05
    min_validation_admissions: int = 30
    seeds: tuple[int, ...] = (314, 2718, 1618)
    live_execution_authorized: bool = False
    paper_replacement_authorized: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def attach_exact_frozen_gate(rows: pd.DataFrame, bundles: dict) -> pd.DataFrame:
    """Attach the already-frozen v0.24b event gate without refit or retuning."""
    x = rows[rows["strategy"].isin(V25_STRATEGIES)].copy().reset_index(drop=True)
    x["frozen_score"] = np.nan
    x["frozen_threshold"] = np.nan
    x["frozen_selected"] = False
    for strategy in V25_STRATEGIES:
        if strategy not in bundles:
            raise RuntimeError(f"V25_FROZEN_BUNDLE_MISSING {strategy}")
        b = bundles[strategy]
        idx = x.index[x["strategy"] == strategy]
        if len(idx) == 0:
            continue
        part = x.loc[idx]
        missing = [c for c in b.features if c not in part.columns]
        if missing:
            raise RuntimeError(f"V25_FROZEN_FEATURE_MISSING {strategy}: {missing}")
        score = probability_score(b.model, part[list(b.features)])
        x.loc[idx, "frozen_score"] = np.asarray(score, dtype=float)
        x.loc[idx, "frozen_threshold"] = float(b.candidate.threshold)
        x.loc[idx, "frozen_selected"] = np.asarray(score, dtype=float) >= float(b.candidate.threshold)
    x["frozen_score_margin"] = x["frozen_score"] - x["frozen_threshold"]
    return x


def build_rank_features(rows: pd.DataFrame, feature_names: Iterable[str] | None = None) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Build decision-time numeric features for the portfolio ranking layer.

    Outcome, exit-price and realized-return fields are intentionally excluded.
    Frozen event-model scores are permitted because they are available before the
    portfolio admission decision and are themselves produced by immutable models.
    """
    if feature_names is None:
        base = sorted(c for c in rows.columns if c.startswith("f_") and not c.startswith("f_rank_"))
        names = tuple(base + [
            "f_rank_is_s6",
            "f_rank_is_long",
            "f_rank_frozen_score",
            "f_rank_frozen_margin",
        ])
    else:
        names = tuple(str(x) for x in feature_names)

    out = pd.DataFrame(index=rows.index)
    for name in names:
        if name == "f_rank_is_s6":
            out[name] = (rows["strategy"].astype(str) == "H4_S6_BREAKOUT").astype(float)
        elif name == "f_rank_is_long":
            out[name] = (rows["side"].astype(str).str.lower() == "long").astype(float)
        elif name == "f_rank_frozen_score":
            out[name] = pd.to_numeric(rows.get("frozen_score"), errors="coerce")
        elif name == "f_rank_frozen_margin":
            out[name] = pd.to_numeric(rows.get("frozen_score_margin"), errors="coerce")
        else:
            out[name] = pd.to_numeric(rows.get(name), errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    return out, names


def _pf(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=float)
    gp = float(x[x > 0].sum())
    gl = float(-x[x < 0].sum())
    return gp / gl if gl > 0 else (np.inf if gp > 0 else np.nan)


def _realized_priority_replay(
    rows: pd.DataFrame,
    priority: Iterable[float],
    contract: V25RankingContract | None = None,
) -> dict:
    """Validation-only overlap replay used to choose a ranking rule.

    This is deliberately a realized-event allocator, not the final MTM gate. It
    answers the narrow model-selection question: when simultaneously eligible
    events compete, does a ranking rule allocate scarce risk more effectively?
    Final scientific evaluation must use fresh future-time MTM evidence.
    """
    c = contract or V25RankingContract()
    x = rows.copy().reset_index(drop=True)
    score = np.asarray(list(priority), dtype=float)
    if len(score) != len(x):
        raise ValueError("priority length mismatch")
    x["_priority"] = score
    x = x[x["frozen_selected"].astype(bool)].copy()
    if x.empty:
        return {"accepted": 0, "total_return": 0.0, "profit_factor": np.nan, "mean_r": np.nan, "max_realized_drawdown": 0.0}
    for col in ("entry_time", "exit_time"):
        x[col] = pd.to_datetime(x[col], utc=True)
    x = x.sort_values(["entry_time", "_priority", "strategy", "symbol"], ascending=[True, False, True, True], kind="mergesort")

    equity = 1.0
    peak = 1.0
    min_dd = 0.0
    hard_kill = False
    active: list[dict] = []
    realized: list[float] = []
    accepted_r: list[float] = []

    def close_due(now: pd.Timestamp | None) -> None:
        nonlocal equity, peak, min_dd, hard_kill, active
        due = [p for p in active if now is None or p["exit_time"] <= now]
        for p in sorted(due, key=lambda q: (q["exit_time"], q["strategy"], q["symbol"])):
            pnl = float(p["risk"] * p["r"])
            equity += pnl
            realized.append(pnl)
            peak = max(peak, equity)
            min_dd = min(min_dd, equity / max(peak, 1e-12) - 1.0)
            if min_dd <= -c.hard_realized_drawdown_kill:
                hard_kill = True
        active = [p for p in active if p not in due]

    for _, row in x.iterrows():
        close_due(row["entry_time"])
        if hard_kill:
            continue
        if any(p["symbol"] == str(row["symbol"]) for p in active):
            continue
        if len(active) >= c.max_concurrent_positions:
            continue
        planned = equity * c.risk_per_trade
        open_risk = sum(float(p["risk"]) for p in active)
        strategy_risk = sum(float(p["risk"]) for p in active if p["strategy"] == str(row["strategy"]))
        side = str(row["side"]).lower()
        directional_risk = sum(float(p["risk"]) for p in active if p["side"] == side)
        if (open_risk + planned) / max(equity, 1e-12) > c.max_open_risk_fraction + 1e-12:
            continue
        if (strategy_risk + planned) / max(equity, 1e-12) > c.max_strategy_open_risk_fraction + 1e-12:
            continue
        if (directional_risk + planned) / max(equity, 1e-12) > c.max_directional_open_risk_fraction + 1e-12:
            continue
        r = float(row["r_multiple"])
        if not np.isfinite(r):
            continue
        active.append({
            "exit_time": row["exit_time"],
            "risk": float(planned),
            "r": r,
            "strategy": str(row["strategy"]),
            "symbol": str(row["symbol"]),
            "side": side,
        })
        accepted_r.append(r)

    close_due(None)
    rr = np.asarray(accepted_r, dtype=float)
    return {
        "accepted": int(len(rr)),
        "total_return": float(equity - 1.0),
        "profit_factor": float(_pf(np.asarray(realized, dtype=float))),
        "mean_r": float(np.mean(rr)) if len(rr) else np.nan,
        "max_realized_drawdown": float(min_dd),
        "hard_kill_triggered": bool(hard_kill),
    }


def _fit_pairwise_logistic(dev: pd.DataFrame, X: pd.DataFrame, contract: V25RankingContract) -> Pipeline:
    pairs: list[np.ndarray] = []
    labels: list[int] = []
    selected = dev[dev["frozen_selected"].astype(bool)].copy()
    for _, g in selected.groupby("entry_time", sort=True):
        if len(g) < 2:
            continue
        for i, j in combinations(g.index.to_list(), 2):
            yi = float(dev.loc[i, "r_multiple"])
            yj = float(dev.loc[j, "r_multiple"])
            if not np.isfinite(yi) or not np.isfinite(yj) or abs(yi - yj) < contract.min_pair_abs_r_gap:
                continue
            diff = X.loc[i].to_numpy(dtype=float) - X.loc[j].to_numpy(dtype=float)
            pairs.extend([diff, -diff])
            labels.extend([1 if yi > yj else 0, 0 if yi > yj else 1])
    if len(pairs) < 50 or len(set(labels)) < 2:
        raise RuntimeError(f"V25_PAIRWISE_SAMPLE_INSUFFICIENT pairs={len(pairs)}")
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=0.2, max_iter=3000, random_state=314)),
    ])
    model.fit(np.asarray(pairs, dtype=float), np.asarray(labels, dtype=int))
    return model


def fit_ranking_tournament(
    development: pd.DataFrame,
    validation: pd.DataFrame,
    contract: V25RankingContract | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Fit allocator challengers on development and freeze one on validation only.

    The deterministic frozen-score rule is always present as a baseline. A learned
    model may become the frozen v0.25 challenger only when its validation objective
    exceeds that baseline. No test/external/future rows are read here.
    """
    c = contract or V25RankingContract()
    dev = development.copy().reset_index(drop=True)
    val = validation.copy().reset_index(drop=True)
    Xdev, features = build_rank_features(dev)
    Xval, _ = build_rank_features(val, features)
    ydev = pd.to_numeric(dev["r_multiple"], errors="coerce").to_numpy(dtype=float)

    candidates: dict[str, tuple[object | None, np.ndarray]] = {
        "frozen_score": (None, pd.to_numeric(val["frozen_score"], errors="coerce").fillna(-1e9).to_numpy(dtype=float)),
        "frozen_margin": (None, pd.to_numeric(val["frozen_score_margin"], errors="coerce").fillna(-1e9).to_numpy(dtype=float)),
    }

    ridge = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", Ridge(alpha=10.0)),
    ])
    ridge.fit(Xdev, ydev)
    candidates["ridge_expected_r"] = (ridge, ridge.predict(Xval))

    hgb = HistGradientBoostingRegressor(
        max_iter=250, learning_rate=0.04, max_leaf_nodes=15,
        l2_regularization=5.0, random_state=314,
    )
    hgb.fit(Xdev, ydev)
    candidates["hgb_expected_r"] = (hgb, hgb.predict(Xval))

    q25 = HistGradientBoostingRegressor(
        loss="quantile", quantile=0.25, max_iter=250, learning_rate=0.04,
        max_leaf_nodes=15, l2_regularization=5.0, random_state=314,
    )
    q25.fit(Xdev, ydev)
    candidates["hgb_lower_quartile_r"] = (q25, q25.predict(Xval))

    pairwise = _fit_pairwise_logistic(dev, Xdev, c)
    candidates["pairwise_logistic_ranker"] = (pairwise, pairwise.decision_function(Xval))

    rows: list[dict] = []
    for name, (_model, priority) in candidates.items():
        metrics = _realized_priority_replay(val, priority, c)
        pf = float(metrics["profit_factor"])
        pf_eff = min(pf, 5.0) if np.isfinite(pf) else 0.0
        mean_r = float(metrics["mean_r"]) if np.isfinite(metrics["mean_r"]) else -10.0
        dd = abs(float(metrics["max_realized_drawdown"]))
        objective = float(metrics["total_return"]) + 0.02 * (pf_eff - 1.0) + 0.01 * mean_r - 0.25 * max(0.0, dd - 0.05)
        rows.append({"ranker": name, "validation_objective": objective, **metrics})

    leaderboard = pd.DataFrame(rows).sort_values(
        ["validation_objective", "accepted", "ranker"], ascending=[False, False, True]
    ).reset_index(drop=True)
    baseline_obj = float(leaderboard.loc[leaderboard["ranker"] == "frozen_score", "validation_objective"].iloc[0])
    winner = leaderboard.iloc[0]
    champion_name = str(winner["ranker"])
    if float(winner["validation_objective"]) <= baseline_obj + 1e-12:
        champion_name = "frozen_score"

    model = candidates[champion_name][0]
    snapshot = {
        "version": "v0.25",
        "champion": champion_name,
        "features": list(features),
        "model": model,
        "validation_only_selection": True,
        "spent_v24d_external_used_for_selection": False,
        "future_start_utc": c.future_start_utc,
        "forward_paper_authorized": False,
        "paper_replacement_authorized": False,
        "live_execution_authorized": False,
    }
    return leaderboard, snapshot


def score_ranker(rows: pd.DataFrame, snapshot: dict) -> np.ndarray:
    """Score a frozen ranking snapshot without refit."""
    name = str(snapshot["champion"])
    if name == "frozen_score":
        return pd.to_numeric(rows["frozen_score"], errors="coerce").fillna(-1e9).to_numpy(dtype=float)
    if name == "frozen_margin":
        return pd.to_numeric(rows["frozen_score_margin"], errors="coerce").fillna(-1e9).to_numpy(dtype=float)
    X, _ = build_rank_features(rows, snapshot["features"])
    model = snapshot["model"]
    if name == "pairwise_logistic_ranker":
        return np.asarray(model.decision_function(X), dtype=float)
    return np.asarray(model.predict(X), dtype=float)
