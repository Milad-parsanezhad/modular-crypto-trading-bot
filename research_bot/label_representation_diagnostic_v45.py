from __future__ import annotations

"""Frozen diagnostic utilities for v0.45 Route C.

This module is deliberately non-promotional. It analyzes already-consumed v0.44
OOS/prepared evidence to localize label, censoring, calibration and representation
failure modes. It never changes a trading threshold, fits a rescue model, fetches
market data, touches Kraken, or authorizes PAPER/LIVE execution.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


V45_PROBABILITY_BIN_EDGES = np.linspace(0.0, 1.0, 11)
V45_MAX_HOLD_BARS = 30
V45_TIMEFRAME_HOURS = 4
V45_MIN_REPORT_N = 50
V45_MIN_FEATURE_STRATUM_N = 200
V45_MIN_FEATURE_CLASS_N = 20


def _finite_binary_pair(y: Sequence[float], p: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    yy = np.asarray(y, dtype=float)
    pp = np.asarray(p, dtype=float)
    if yy.shape != pp.shape:
        raise ValueError("y and p must have identical shape")
    keep = np.isfinite(yy) & np.isfinite(pp)
    yy = yy[keep]
    pp = np.clip(pp[keep], 0.0, 1.0)
    if not np.isin(yy, [0.0, 1.0]).all():
        raise ValueError("binary outcome must contain only 0/1")
    return yy, pp


def brier_decomposition_v45(
    y: Sequence[float],
    p: Sequence[float],
) -> tuple[dict[str, float | int | None], pd.DataFrame]:
    """Murphy-style Brier decomposition using ten preregistered equal-width bins."""

    yy, pp = _finite_binary_pair(y, p)
    if len(yy) == 0:
        return {
            "n": 0,
            "prevalence": None,
            "brier": None,
            "uncertainty": None,
            "reliability": None,
            "resolution": None,
            "reconstructed_brier": None,
            "reconstruction_residual": None,
        }, pd.DataFrame(columns=["bin", "lower", "upper", "n", "mean_prediction", "empirical_frequency"])

    prevalence = float(np.mean(yy))
    uncertainty = prevalence * (1.0 - prevalence)
    brier = float(np.mean((pp - yy) ** 2))

    # Search over the nine internal cut points. p=1.0 remains in the final bin.
    bin_id = np.searchsorted(V45_PROBABILITY_BIN_EDGES[1:-1], pp, side="right")
    reliability = 0.0
    resolution = 0.0
    rows: list[dict] = []
    n_total = float(len(yy))

    for k in range(10):
        take = bin_id == k
        nk = int(np.sum(take))
        lower = float(V45_PROBABILITY_BIN_EDGES[k])
        upper = float(V45_PROBABILITY_BIN_EDGES[k + 1])
        if nk == 0:
            rows.append({
                "bin": k,
                "lower": lower,
                "upper": upper,
                "n": 0,
                "mean_prediction": None,
                "empirical_frequency": None,
            })
            continue
        pk = float(np.mean(pp[take]))
        yk = float(np.mean(yy[take]))
        weight = nk / n_total
        reliability += weight * (pk - yk) ** 2
        resolution += weight * (yk - prevalence) ** 2
        rows.append({
            "bin": k,
            "lower": lower,
            "upper": upper,
            "n": nk,
            "mean_prediction": pk,
            "empirical_frequency": yk,
        })

    reconstructed = float(reliability - resolution + uncertainty)
    return {
        "n": int(len(yy)),
        "prevalence": prevalence,
        "brier": brier,
        "uncertainty": float(uncertainty),
        "reliability": float(reliability),
        "resolution": float(resolution),
        "reconstructed_brier": reconstructed,
        "reconstruction_residual": float(brier - reconstructed),
    }, pd.DataFrame(rows)


def duration_bars_v45(events: pd.DataFrame) -> pd.Series:
    required = {"entry_time", "exit_time"}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"missing duration columns: {sorted(missing)}")
    entry = pd.to_datetime(events["entry_time"], utc=True, errors="raise")
    exit_ = pd.to_datetime(events["exit_time"], utc=True, errors="raise")
    bars = ((exit_ - entry).dt.total_seconds() / (V45_TIMEFRAME_HOURS * 3600.0)).round().astype(int) + 1
    return bars.clip(lower=1, upper=V45_MAX_HOLD_BARS).astype(int)


def assign_common_test_folds_v45(events: pd.DataFrame, fold_index: pd.DataFrame) -> pd.DataFrame:
    """Return S0 events that fall in exactly one frozen v0.44 common test window."""

    required_events = {"signal_time", "v44_sample_s0"}
    missing = required_events - set(events.columns)
    if missing:
        raise ValueError(f"missing event columns: {sorted(missing)}")
    required_folds = {"fold", "test_start", "test_end"}
    missing_folds = required_folds - set(fold_index.columns)
    if missing_folds:
        raise ValueError(f"missing fold columns: {sorted(missing_folds)}")
    if len(fold_index) != 5 or fold_index["fold"].astype(int).nunique() != 5:
        raise ValueError("v0.45 requires exactly five frozen common folds")

    x = events.loc[events["v44_sample_s0"].astype(bool)].copy()
    signal = pd.to_datetime(x["signal_time"], utc=True, errors="raise")
    assigned = pd.Series(np.zeros(len(x), dtype=np.int16), index=x.index)
    membership = pd.Series(np.zeros(len(x), dtype=np.int8), index=x.index)

    for _, row in fold_index.sort_values("fold").iterrows():
        start = pd.Timestamp(row["test_start"])
        end = pd.Timestamp(row["test_end"])
        if start.tzinfo is None:
            start = start.tz_localize("UTC")
        else:
            start = start.tz_convert("UTC")
        if end.tzinfo is None:
            end = end.tz_localize("UTC")
        else:
            end = end.tz_convert("UTC")
        mask = (signal >= start) & (signal <= end)
        membership.loc[mask] += 1
        assigned.loc[mask] = int(row["fold"])

    if bool((membership > 1).any()):
        raise RuntimeError("common v0.44 test windows overlap")
    x["fold"] = assigned.to_numpy(dtype=np.int16)
    x = x.loc[membership.eq(1)].copy()
    x["duration_bars_v45"] = duration_bars_v45(x)
    return x.sort_values(["fold", "signal_time", "symbol", "venue"], kind="mergesort").reset_index(drop=True)


def _group_label_row(scope: str, value: str, g: pd.DataFrame) -> dict:
    outcome = g["outcome"].astype(str)
    n = int(len(g))
    duration = duration_bars_v45(g) if n else pd.Series(dtype=float)
    net_r = pd.to_numeric(g["net_r"], errors="coerce") if "net_r" in g else pd.Series(dtype=float)
    return {
        "scope": scope,
        "group_value": value,
        "n": n,
        "support_status": "OK" if n >= V45_MIN_REPORT_N else "LOW_SUPPORT",
        "target_n": int(outcome.eq("TARGET").sum()),
        "stop_n": int(outcome.eq("STOP").sum()),
        "time_n": int(outcome.eq("TIME").sum()),
        "target_fraction": float(outcome.eq("TARGET").mean()) if n else None,
        "stop_fraction": float(outcome.eq("STOP").mean()) if n else None,
        "time_fraction": float(outcome.eq("TIME").mean()) if n else None,
        "duration_mean_bars": float(duration.mean()) if n else None,
        "duration_median_bars": float(duration.median()) if n else None,
        "net_r_mean": float(net_r.mean()) if net_r.notna().any() else None,
    }


def label_distribution_v45(events: pd.DataFrame) -> pd.DataFrame:
    required = {"outcome", "entry_time", "exit_time"}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"missing label-distribution columns: {sorted(missing)}")
    rows = [_group_label_row("overall", "ALL", events)]
    scopes = [
        ("fold", "fold"),
        ("venue", "venue"),
        ("side", "side"),
        ("regime", "regime"),
        ("event_family", "event_family_v41"),
    ]
    for scope, column in scopes:
        if column not in events.columns:
            continue
        for value, g in events.groupby(column, sort=True, dropna=False):
            rows.append(_group_label_row(scope, str(value), g))
    return pd.DataFrame(rows)


def cause_age_hazard_v45(
    events: pd.DataFrame,
    *,
    scope: str = "overall",
    group_value: str = "ALL",
) -> pd.DataFrame:
    required = {"outcome", "entry_time", "exit_time"}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"missing hazard columns: {sorted(missing)}")
    x = events.copy()
    durations = duration_bars_v45(x).to_numpy(dtype=int)
    outcomes = x["outcome"].astype(str).to_numpy()
    n0 = int(len(x))
    cum_target = 0
    cum_stop = 0
    rows: list[dict] = []
    for age in range(1, V45_MAX_HOLD_BARS + 1):
        at_risk = durations >= age
        risk_n = int(np.sum(at_risk))
        terminal = durations == age
        target_n = int(np.sum(terminal & (outcomes == "TARGET")))
        stop_n = int(np.sum(terminal & (outcomes == "STOP")))
        time_n = int(np.sum(terminal & (outcomes == "TIME")))
        cum_target += target_n
        cum_stop += stop_n
        rows.append({
            "scope": scope,
            "group_value": group_value,
            "age_bar": age,
            "initial_n": n0,
            "at_risk_n": risk_n,
            "target_terminal_n": target_n,
            "stop_terminal_n": stop_n,
            "time_terminal_n": time_n,
            "target_hazard": float(target_n / risk_n) if risk_n else None,
            "stop_hazard": float(stop_n / risk_n) if risk_n else None,
            "time_terminal_fraction_of_risk": float(time_n / risk_n) if risk_n else None,
            "cumulative_target_incidence_proxy": float(cum_target / n0) if n0 else None,
            "cumulative_stop_incidence_proxy": float(cum_stop / n0) if n0 else None,
        })
    return pd.DataFrame(rows)


def cause_age_hazard_strata_v45(events: pd.DataFrame) -> pd.DataFrame:
    parts = [cause_age_hazard_v45(events)]
    for scope, column in (("regime", "regime"), ("event_family", "event_family_v41")):
        if column not in events.columns:
            continue
        for value, g in events.groupby(column, sort=True, dropna=False):
            parts.append(cause_age_hazard_v45(g, scope=scope, group_value=str(value)))
    return pd.concat(parts, ignore_index=True, sort=False)


def timeout_diagnostics_v45(events: pd.DataFrame) -> pd.DataFrame:
    time_rows = events.loc[events["outcome"].astype(str).eq("TIME")].copy()
    scopes: list[tuple[str, str | None]] = [
        ("overall", None),
        ("fold", "fold"),
        ("regime", "regime"),
        ("event_family", "event_family_v41"),
    ]
    rows: list[dict] = []
    for scope, column in scopes:
        groups = [("ALL", time_rows)] if column is None else list(time_rows.groupby(column, sort=True, dropna=False))
        for value, g in groups:
            duration = duration_bars_v45(g) if len(g) else pd.Series(dtype=float)
            net_r = pd.to_numeric(g["net_r"], errors="coerce") if "net_r" in g else pd.Series(dtype=float)
            rows.append({
                "scope": scope,
                "group_value": str(value),
                "n": int(len(g)),
                "support_status": "OK" if len(g) >= V45_MIN_REPORT_N else "LOW_SUPPORT",
                "duration_mean_bars": float(duration.mean()) if len(g) else None,
                "duration_median_bars": float(duration.median()) if len(g) else None,
                "net_r_mean": float(net_r.mean()) if net_r.notna().any() else None,
                "net_r_median": float(net_r.median()) if net_r.notna().any() else None,
                "net_r_positive_fraction": float((net_r > 0.0).mean()) if net_r.notna().any() else None,
            })
    return pd.DataFrame(rows)


def _safe_auc(y: np.ndarray, x: np.ndarray, *, min_class_n: int) -> float | None:
    finite = np.isfinite(y) & np.isfinite(x)
    yy = y[finite].astype(int)
    xx = x[finite].astype(float)
    if len(yy) == 0:
        return None
    positives = int(np.sum(yy == 1))
    negatives = int(np.sum(yy == 0))
    if positives < min_class_n or negatives < min_class_n:
        return None
    return float(roc_auc_score(yy, xx))


def _safe_spearman(y: np.ndarray, x: np.ndarray) -> float | None:
    frame = pd.DataFrame({"y": y, "x": x}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 3 or frame["x"].nunique() < 2 or frame["y"].nunique() < 2:
        return None
    value = frame["x"].corr(frame["y"], method="spearman")
    return float(value) if pd.notna(value) else None


def feature_information_v45(
    events: pd.DataFrame,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    missing = sorted(set(feature_columns) - set(events.columns))
    if missing:
        raise ValueError(f"missing frozen features: {missing}")
    if "fold" not in events or "outcome" not in events:
        raise ValueError("feature diagnostics require fold and outcome")

    rows: list[dict] = []
    for fold, fold_data in events.groupby("fold", sort=True):
        strata: list[tuple[str, str, pd.DataFrame]] = [("overall", "ALL", fold_data)]
        for scope, column in (("regime", "regime"), ("event_family", "event_family_v41")):
            if column in fold_data.columns:
                for value, g in fold_data.groupby(column, sort=True, dropna=False):
                    strata.append((scope, str(value), g))

        for scope, value, g in strata:
            outcome = g["outcome"].astype(str)
            y_target = outcome.eq("TARGET").to_numpy(dtype=int)
            y_stop = outcome.eq("STOP").to_numpy(dtype=int)
            n = int(len(g))
            for feature in feature_columns:
                x = pd.to_numeric(g[feature], errors="coerce").to_numpy(dtype=float)
                target_auc = _safe_auc(y_target, x, min_class_n=V45_MIN_FEATURE_CLASS_N) if n >= V45_MIN_FEATURE_STRATUM_N else None
                stop_auc = _safe_auc(y_stop, x, min_class_n=V45_MIN_FEATURE_CLASS_N) if n >= V45_MIN_FEATURE_STRATUM_N else None
                target_supported = target_auc is not None
                stop_supported = stop_auc is not None
                rows.append({
                    "fold": int(fold),
                    "scope": scope,
                    "group_value": value,
                    "feature": feature,
                    "n": n,
                    "target_n": int(np.sum(y_target)),
                    "stop_n": int(np.sum(y_stop)),
                    "target_supported": bool(target_supported),
                    "stop_supported": bool(stop_supported),
                    "target_auc": target_auc,
                    "target_auc_abs_edge": abs(float(target_auc) - 0.5) if target_supported else None,
                    "stop_auc": stop_auc,
                    "stop_auc_abs_edge": abs(float(stop_auc) - 0.5) if stop_supported else None,
                    "target_spearman": _safe_spearman(y_target.astype(float), x) if n >= V45_MIN_FEATURE_STRATUM_N else None,
                    "stop_spearman": _safe_spearman(y_stop.astype(float), x) if n >= V45_MIN_FEATURE_STRATUM_N else None,
                })
    return pd.DataFrame(rows)
