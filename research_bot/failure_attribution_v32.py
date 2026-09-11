from __future__ import annotations

"""v0.32 diagnostic attribution of the consumed v0.31 KuCoin failure.

This stage is diagnostic only. KuCoin is consumed evidence and may not be used as
an untouched holdout again. No strategy parameters or qualification thresholds
are changed here.
"""

from typing import Any
import numpy as np
import pandas as pd

REQUIRED_V31_DECISION = "V31_FIREWALL_CANDIDATE_REJECTED_HOLDOUT"


def validate_v31_source(decision: dict[str, Any]) -> None:
    if decision.get("decision") != REQUIRED_V31_DECISION:
        raise ValueError("v0.32 requires the completed rejected v0.31 KuCoin holdout")
    if not bool(decision.get("holdout_used", False)):
        raise ValueError("v0.31 holdout must have been used")
    if bool(decision.get("live_execution_authorized", True)):
        raise ValueError("live execution must remain disabled")


def _pf(x: pd.Series) -> float | None:
    a = pd.to_numeric(x, errors="coerce").dropna()
    gains = float(a[a > 0].sum())
    losses = float(-a[a < 0].sum())
    if losses <= 0:
        return None if gains <= 0 else float("inf")
    return gains / losses


def attribute_holdout_failure(ledger: pd.DataFrame, holdout_metrics: dict[str, Any]) -> dict[str, Any]:
    if ledger.empty:
        raise ValueError("v0.32 requires the v0.31 holdout ledger")
    x = ledger.copy()
    x["entry_time"] = pd.to_datetime(x["entry_time"], utc=True, errors="raise")
    if "executed_v25" in x:
        x = x.loc[x["executed_v25"].fillna(False).astype(bool)].copy()
    if x.empty:
        raise ValueError("no executed v0.31 holdout trades")
    ret_col = "account_return_v25" if "account_return_v25" in x else "account_return"
    x[ret_col] = pd.to_numeric(x[ret_col], errors="coerce").fillna(0.0)
    x["r_multiple"] = pd.to_numeric(x["r_multiple"], errors="coerce")

    rows: list[dict[str, Any]] = []
    for symbol, g in x.groupby("symbol", sort=True):
        pnl = float(g[ret_col].sum())
        rows.append({
            "symbol": str(symbol),
            "trades": int(len(g)),
            "account_return_sum": pnl,
            "mean_r": float(g["r_multiple"].mean()),
            "profit_factor_account_return": _pf(g[ret_col]),
            "positive": bool(pnl > 0),
        })
    assets = pd.DataFrame(rows).sort_values("account_return_sum", ascending=False, kind="mergesort")

    positive = assets.loc[assets["account_return_sum"] > 0, "account_return_sum"]
    negative = -assets.loc[assets["account_return_sum"] < 0, "account_return_sum"]
    positive_total = float(positive.sum())
    negative_total = float(negative.sum())
    top3_positive_share = float(positive.nlargest(3).sum() / positive_total) if positive_total > 0 else None
    bottom3_negative_share = float(negative.nlargest(3).sum() / negative_total) if negative_total > 0 else None

    x["quarter"] = x["entry_time"].dt.tz_convert(None).dt.to_period("Q").astype(str)
    blocks = (
        x.groupby("quarter", sort=True)[ret_col]
        .agg([("trades", "size"), ("account_return_sum", "sum"), ("mean_account_return", "mean")])
        .reset_index()
        .sort_values("account_return_sum", kind="mergesort")
    )

    side_rows = []
    for side, g in x.groupby("side", sort=True):
        side_rows.append({
            "side": int(side),
            "trades": int(len(g)),
            "account_return_sum": float(g[ret_col].sum()),
            "mean_r": float(g["r_multiple"].mean()),
            "profit_factor_account_return": _pf(g[ret_col]),
        })

    observed_breadth = float(assets["positive"].mean()) if len(assets) else 0.0
    reported_breadth = float(holdout_metrics.get("positive_asset_fraction", np.nan))
    return {
        "version": "v0.32",
        "decision": "V32_FAILURE_ATTRIBUTED_NO_PROMOTION",
        "source": "consumed_v0.31_kucoin_holdout",
        "trades": int(len(x)),
        "asset_count": int(len(assets)),
        "observed_positive_asset_fraction": observed_breadth,
        "reported_positive_asset_fraction_v31": reported_breadth if np.isfinite(reported_breadth) else None,
        "negative_assets": assets.loc[~assets["positive"], "symbol"].tolist(),
        "positive_assets": assets.loc[assets["positive"], "symbol"].tolist(),
        "top3_positive_concentration": top3_positive_share,
        "bottom3_negative_concentration": bottom3_negative_share,
        "worst_three_quarters": blocks.head(3).to_dict("records"),
        "best_three_quarters": blocks.tail(3).sort_values("account_return_sum", ascending=False).to_dict("records"),
        "side_attribution": side_rows,
        "asset_attribution": assets.to_dict("records"),
        "diagnosed_failures": list(holdout_metrics.get("holdout_failures", [])),
        "kucoin_consumed": True,
        "threshold_relaxation": False,
        "strategy_parameter_retuning": False,
        "forward_paper_candidate_authorized": False,
        "live_execution_authorized": False,
    }
