from __future__ import annotations

from hashlib import sha256

import numpy as np
import pandas as pd

from research_bot.multitimeframe_strategies_v19 import moving_block_mean_ci


def canonical_ohlcv_frame(frame: pd.DataFrame) -> pd.DataFrame:
    cols = ["timestamp", "open", "high", "low", "close", "volume", "source"]
    x = frame.copy()
    for c in cols:
        if c not in x.columns:
            x[c] = np.nan if c != "source" else ""
    x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    return x[cols].sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)


def frame_sha256(frame: pd.DataFrame) -> str:
    x = canonical_ohlcv_frame(frame).copy()
    x["timestamp"] = x["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    raw = x.to_csv(index=False, float_format="%.12g", lineterminator="\n").encode("utf-8")
    return sha256(raw).hexdigest()


def merge_append_only(previous: pd.DataFrame, fresh: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Preserve first-observed OHLCV rows and append only new timestamps.

    If a venue later restates an overlapping bar, the archived first observation
    remains unchanged and the difference is counted for provenance/audit purposes.
    """
    prev = canonical_ohlcv_frame(previous) if len(previous) else pd.DataFrame()
    new = canonical_ohlcv_frame(fresh) if len(fresh) else pd.DataFrame()
    if prev.empty:
        return new, 0
    if new.empty:
        return prev, 0

    numeric = ["open", "high", "low", "close", "volume"]
    overlap = prev[["timestamp", *numeric]].merge(new[["timestamp", *numeric]], on="timestamp", suffixes=("_old", "_new"))
    revisions = 0
    for _, row in overlap.iterrows():
        changed = False
        for c in numeric:
            a = float(row[f"{c}_old"]) if pd.notna(row[f"{c}_old"]) else np.nan
            b = float(row[f"{c}_new"]) if pd.notna(row[f"{c}_new"]) else np.nan
            if (np.isnan(a) != np.isnan(b)) or (
                np.isfinite(a)
                and np.isfinite(b)
                and not np.isclose(a, b, rtol=1e-10, atol=1e-12)
            ):
                changed = True
                break
        revisions += int(changed)

    last_prev = prev["timestamp"].max()
    appended = new[new["timestamp"] > last_prev]
    merged = pd.concat([prev, appended], ignore_index=True)
    return canonical_ohlcv_frame(merged), revisions


def stress_r_multiple(events: pd.DataFrame, roundtrip_bps: float) -> pd.Series:
    """Recompute event R under a frozen higher round-trip cost assumption."""
    required = {"entry", "stop", "gross_return"}
    missing = required - set(events.columns)
    if missing:
        raise RuntimeError(f"V25_COST_STRESS_COLUMNS_MISSING {sorted(missing)}")
    entry = pd.to_numeric(events["entry"], errors="coerce")
    stop = pd.to_numeric(events["stop"], errors="coerce")
    gross = pd.to_numeric(events["gross_return"], errors="coerce")
    stop_fraction = (entry - stop).abs() / entry.abs().replace(0, np.nan)
    return (gross - float(roundtrip_bps) / 10_000.0) / stop_fraction.replace(0, np.nan)


def paired_curve_uplift_ci(base_curve: pd.DataFrame, ranked_curve: pd.DataFrame) -> dict:
    """Moving-block CI for aligned close-MTM return uplift of ranker vs baseline."""
    if base_curve.empty or ranked_curve.empty:
        return {"low": np.nan, "high": np.nan, "observations": 0}
    left = base_curve[["timestamp", "close_mtm_equity"]].copy()
    right = ranked_curve[["timestamp", "close_mtm_equity"]].copy()
    left["timestamp"] = pd.to_datetime(left["timestamp"], utc=True)
    right["timestamp"] = pd.to_datetime(right["timestamp"], utc=True)
    merged = left.merge(right, on="timestamp", suffixes=("_base", "_ranked"), validate="one_to_one").sort_values("timestamp")
    if len(merged) < 20:
        return {"low": np.nan, "high": np.nan, "observations": int(max(0, len(merged) - 1))}
    base_ret = merged["close_mtm_equity_base"].pct_change()
    ranked_ret = merged["close_mtm_equity_ranked"].pct_change()
    delta = (ranked_ret - base_ret).replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if len(delta) < 20:
        return {"low": np.nan, "high": np.nan, "observations": int(len(delta))}
    block = max(5, min(20, max(5, len(delta) // 8)))
    low, high = moving_block_mean_ci(delta, samples=1200, block=block, seed=250911)
    return {"low": float(low), "high": float(high), "observations": int(len(delta)), "block": int(block)}
