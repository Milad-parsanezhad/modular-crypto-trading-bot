from __future__ import annotations

"""Prepare the frozen v0.43 screened event table without fitting any model."""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_bot.asset_crossvenue_v43 import (
    DataQualityPolicyV43,
    V43_DEVELOPMENT_VENUES,
    data_quality_diagnostics_v43,
    preregistration_manifest_v43,
)
from research_bot.event_competing_risk_v41 import V41_FEATURES, attach_event_family_v41
from research_bot.financial_system_v39 import causal_robust_normalize
from research_bot.mother_strategy_v39 import (
    NEURAL_FEATURES_V39,
    MotherStrategyPolicyV39,
    build_mother_features_v39,
)

ROOT = Path(__file__).resolve().parents[1]
V42 = ROOT / "scripts" / "run_v42_breadth_cluster_characterization.py"
spec = importlib.util.spec_from_file_location("v42_for_v43_prepare", V42)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load frozen v0.42 helper")
v42 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v42)
base = v42.base


def _label_safe_cutoff(frame: pd.DataFrame, horizon_bars: int) -> pd.Timestamp:
    """Latest signal timestamp that still has `horizon_bars` actual future bars.

    v0.39 labels enter on t+1 and may inspect through t+horizon_bars.  A wall-
    clock subtraction is insufficient when the quality policy permits small gaps,
    so this cutoff is based on observed bar position rather than elapsed hours.
    """
    x = frame.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    if len(x) <= int(horizon_bars):
        raise RuntimeError("series too short to reserve a complete label horizon")
    return pd.Timestamp(pd.to_datetime(x["timestamp"], utc=True).iloc[-(int(horizon_bars) + 1)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    manifest = preregistration_manifest_v43()
    if manifest["kraken_touched"] is not False or manifest["reserved_holdout"] != "kraken":
        raise RuntimeError("v0.43 holdout governance invalid")
    if not (ROOT / "docs" / "V42_RESULTS_2026-09-12.md").exists():
        raise RuntimeError("v0.42 frozen result must exist before v0.43 preparation")

    policy = DataQualityPolicyV43()
    frames, _, availability = v42._fetch_availability()
    availability.to_csv(outdir / "availability_v43.csv", index=False)

    quality_rows: list[dict] = []
    accepted_frames: dict[tuple[str, str], pd.DataFrame] = {}
    for venue in V43_DEVELOPMENT_VENUES:
        for symbol in v42.V42_SYMBOL_CANDIDATES:
            frame = frames.get((venue, symbol))
            if frame is None:
                quality_rows.append({
                    "venue": venue,
                    "symbol": symbol,
                    "accepted": False,
                    "reason_codes": "MISSING_OR_UNFETCHABLE_SERIES",
                    "bars": 0,
                })
                continue
            diag = data_quality_diagnostics_v43(frame, policy)
            row = {"venue": venue, "symbol": symbol, **diag}
            row["reason_codes"] = ";".join(diag.get("reason_codes", []))
            quality_rows.append(row)
            if diag["accepted"]:
                accepted_frames[(venue, symbol)] = frame.copy()

    quality = pd.DataFrame(quality_rows)
    quality.to_csv(outdir / "data_quality_manifest_v43.csv", index=False)

    accepted_counts = quality.loc[quality["accepted"].eq(True)].groupby("symbol")["venue"].nunique()
    eligible_assets = tuple(
        symbol for symbol in v42.V42_SYMBOL_CANDIDATES if int(accepted_counts.get(symbol, 0)) >= policy.minimum_training_venues
    )
    if not eligible_assets:
        status = {
            "status": "DATA_UNAVAILABLE",
            "reason": "no asset has quality-passed data on >=2 development venues",
            "kraken_touched": False,
        }
        (outdir / "prep_status_v43.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        return

    pd.DataFrame({"eligible_asset": eligible_assets}).to_csv(outdir / "eligible_assets_v43.csv", index=False)

    cfg = MotherStrategyPolicyV39()
    panels: list[pd.DataFrame] = []
    events: list[pd.DataFrame] = []
    series_rows: list[dict] = []

    for (venue, symbol), frame in accepted_frames.items():
        if symbol not in eligible_assets:
            continue
        f = build_mother_features_v39(frame, cfg).copy()
        f["venue"] = venue
        f["symbol"] = symbol
        f["series_id"] = f"{venue}::{symbol}"
        series_id = f"{venue}::{symbol}"
        cutoff = _label_safe_cutoff(frame, policy.label_horizon_bars)
        panels.append(f)
        e = base._event_labels(f, venue, symbol, cfg)
        if not e.empty:
            e = e[pd.to_datetime(e["signal_time"], utc=True) <= cutoff].copy()
            # Fail closed: every retained signal must have at least the frozen
            # number of *actual observed bars* after signal time in its series.
            ts = pd.DatetimeIndex(pd.to_datetime(frame["timestamp"], utc=True))
            for signal_time in pd.to_datetime(e["signal_time"], utc=True).unique():
                pos = int(ts.searchsorted(pd.Timestamp(signal_time), side="left"))
                if len(ts) - pos - 1 < policy.label_horizon_bars:
                    raise RuntimeError(
                        f"incomplete label horizon retained: {series_id} {signal_time}"
                    )
            events.append(e)
        series_rows.append({
            "venue": venue,
            "symbol": symbol,
            "series_id": series_id,
            "bars": int(len(frame)),
            "label_cutoff": cutoff,
            "label_horizon_actual_bars": int(policy.label_horizon_bars),
            "mother_events": int(f["mother_event_v39"].sum()),
        })

    panel = pd.concat(panels, ignore_index=True, sort=False)
    event_frame = pd.concat(events, ignore_index=True, sort=False) if events else pd.DataFrame()
    if event_frame.empty:
        raise RuntimeError("v0.43 generated no settled mother events")

    normalized = causal_robust_normalize(
        panel,
        NEURAL_FEATURES_V39,
        timestamp_col="timestamp",
        group_col="series_id",
    ).rename(columns={"timestamp": "signal_time"})
    feature_cols: list[str] = []
    for c in NEURAL_FEATURES_V39:
        feature_cols.extend([c, f"{c}__missing"])
    event_frame = event_frame.merge(
        normalized[["signal_time", "series_id", *feature_cols]],
        on=["signal_time", "series_id"],
        how="left",
        validate="many_to_one",
    )
    event_frame = event_frame.replace([np.inf, -np.inf], np.nan)
    event_frame[feature_cols] = event_frame[feature_cols].fillna(0.0).astype("float32")
    event_frame = attach_event_family_v41(panel, event_frame)
    missing = sorted(set(V41_FEATURES) - set(event_frame.columns))
    if missing:
        raise RuntimeError(f"missing frozen v0.41 features: {missing}")
    event_frame.loc[:, list(V41_FEATURES)] = (
        event_frame.loc[:, list(V41_FEATURES)].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32")
    )
    for venue in V43_DEVELOPMENT_VENUES:
        event_frame[f"venue_{venue}_v43"] = event_frame["venue"].astype(str).eq(venue).astype("float32")

    event_frame = event_frame.sort_values(["signal_time", "symbol", "venue"], kind="mergesort").reset_index(drop=True)
    event_frame.to_pickle(outdir / "events_v43.pkl.gz", compression="gzip")
    pd.DataFrame(series_rows).to_csv(outdir / "data_manifest_v43.csv", index=False)

    folds = base._folds(event_frame)
    fold_rows = []
    for fold in folds:
        fold_rows.append({
            "fold": int(fold["fold"]),
            "fit_events": int(len(fold["fit"])),
            "calibration_events": int(len(fold["cal"])),
            "test_events": int(len(fold["test"])),
            "test_start": fold["test_start"],
            "test_end": fold["test_end"],
        })
    pd.DataFrame(fold_rows).to_csv(outdir / "fold_index_v43.csv", index=False)
    status = {
        "status": "READY",
        "eligible_asset_count": len(eligible_assets),
        "eligible_assets": list(eligible_assets),
        "settled_events": int(len(event_frame)),
        "folds": len(folds),
        "kraken_touched": False,
    }
    (outdir / "prep_status_v43.json").write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(status, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
