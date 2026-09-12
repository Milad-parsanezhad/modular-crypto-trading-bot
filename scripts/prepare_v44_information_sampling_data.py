from __future__ import annotations

"""Prepare frozen v0.44 data once and freeze common S0 fold boundaries.

No model is fit here.  Sampling flags use only causal OHLCV/mother-event data.
Kraken is never instantiated or fetched.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_bot.asset_crossvenue_v43 import (
    DataQualityPolicyV43,
    V43_DEVELOPMENT_VENUES,
    data_quality_diagnostics_v43,
)
from research_bot.event_competing_risk_v41 import V41_FEATURES
from research_bot.event_family_priority_v43 import attach_event_family_priority_v43
from research_bot.financial_system_v39 import causal_robust_normalize
from research_bot.information_sampling_v44 import (
    V44_RESERVED_HOLDOUT,
    V44_VARIANTS,
    build_sampling_flags_v44,
    preregistration_manifest_v44,
    variant_column_v44,
)
from research_bot.mother_strategy_v39 import (
    NEURAL_FEATURES_V39,
    MotherStrategyPolicyV39,
    build_mother_features_v39,
)

ROOT = Path(__file__).resolve().parents[1]
V42_FAST = ROOT / "scripts" / "run_v42_breadth_cluster_characterization_fast.py"
spec = importlib.util.spec_from_file_location("v42_fast_for_v44_prepare", V42_FAST)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load frozen v0.42 fast data helper")
v42fast = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v42fast)
v42 = v42fast.runner
base = v42.base

EXPECTED_PREREG_SHA256 = "4b198a6572c7b3dfbefac633bb673eef530a0d52324f81f432a3e4302651e332"
STEP = pd.Timedelta(hours=4)
EMBARGO_BARS = 30


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _label_safe_cutoff(frame: pd.DataFrame, horizon_bars: int) -> pd.Timestamp:
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

    prereg = ROOT / "docs" / "V44_INFORMATION_DRIVEN_SAMPLING_PREREGISTRATION.md"
    if _sha256(prereg) != EXPECTED_PREREG_SHA256:
        raise RuntimeError("v0.44 preregistration digest changed after freeze")
    manifest = preregistration_manifest_v44()
    if manifest["reserved_holdout"] != V44_RESERVED_HOLDOUT or manifest["kraken_touched"] is not False:
        raise RuntimeError("v0.44 holdout governance invalid")
    if V44_RESERVED_HOLDOUT != "kraken":
        raise RuntimeError("unexpected v0.44 holdout")

    dq = DataQualityPolicyV43()
    frames, _, availability = v42._fetch_availability()
    availability.to_csv(outdir / "availability_v44.csv", index=False)

    quality_rows: list[dict] = []
    accepted_frames: dict[tuple[str, str], pd.DataFrame] = {}
    for venue in V43_DEVELOPMENT_VENUES:
        if venue == "kraken":
            raise RuntimeError("Kraken is sealed")
        for symbol in v42.V42_SYMBOL_CANDIDATES:
            frame = frames.get((venue, symbol))
            if frame is None:
                quality_rows.append({
                    "venue": venue, "symbol": symbol, "accepted": False,
                    "reason_codes": "MISSING_OR_UNFETCHABLE_SERIES", "bars": 0,
                })
                continue
            diag = data_quality_diagnostics_v43(frame, dq)
            row = {"venue": venue, "symbol": symbol, **diag}
            row["reason_codes"] = ";".join(diag.get("reason_codes", []))
            quality_rows.append(row)
            if bool(diag["accepted"]):
                accepted_frames[(venue, symbol)] = frame.copy()

    quality = pd.DataFrame(quality_rows)
    quality.to_csv(outdir / "data_quality_manifest_v44.csv", index=False)
    accepted_counts = quality.loc[quality["accepted"].eq(True)].groupby("symbol")["venue"].nunique()
    eligible_assets = tuple(
        s for s in v42.V42_SYMBOL_CANDIDATES
        if int(accepted_counts.get(s, 0)) >= dq.minimum_training_venues
    )
    if not eligible_assets:
        status = {"status": "DATA_UNAVAILABLE", "reason": "no eligible assets", "kraken_touched": False}
        (outdir / "prep_status_v44.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        return
    pd.DataFrame({"eligible_asset": eligible_assets}).to_csv(outdir / "eligible_assets_v44.csv", index=False)

    cfg = MotherStrategyPolicyV39()
    panels: list[pd.DataFrame] = []
    events: list[pd.DataFrame] = []
    series_rows: list[dict] = []
    sampling_rows: list[dict] = []

    for (venue, symbol), frame in accepted_frames.items():
        if symbol not in eligible_assets:
            continue
        f = build_mother_features_v39(frame, cfg).copy()
        f["venue"] = venue
        f["symbol"] = symbol
        f["series_id"] = f"{venue}::{symbol}"
        series_id = f"{venue}::{symbol}"
        cutoff = _label_safe_cutoff(frame, dq.label_horizon_bars)

        sampling_input = f[["timestamp", "open", "high", "low", "close", "volume", "mother_event_v39"]].copy()
        flags = build_sampling_flags_v44(sampling_input)
        for c in flags.columns:
            f[c] = flags[c].to_numpy()
        panels.append(f)

        e = base._event_labels(f, venue, symbol, cfg)
        if not e.empty:
            e = e[pd.to_datetime(e["signal_time"], utc=True) <= cutoff].copy()
            ts = pd.DatetimeIndex(pd.to_datetime(frame["timestamp"], utc=True))
            for signal_time in pd.to_datetime(e["signal_time"], utc=True).unique():
                pos = int(ts.searchsorted(pd.Timestamp(signal_time), side="left"))
                if len(ts) - pos - 1 < dq.label_horizon_bars:
                    raise RuntimeError(f"incomplete label horizon retained: {series_id} {signal_time}")
            lookup = f[["timestamp", "series_id", "v44_sample_s0", "v44_sample_s1", "v44_sample_s2"]].rename(
                columns={"timestamp": "signal_time"}
            )
            e = e.merge(lookup, on=["signal_time", "series_id"], how="left", validate="many_to_one")
            for c in ("v44_sample_s0", "v44_sample_s1", "v44_sample_s2"):
                e[c] = e[c].fillna(False).astype(bool)
            if not e["v44_sample_s0"].all():
                raise RuntimeError("settled event table contains non-mother S0 rows")
            events.append(e)

        series_rows.append({
            "venue": venue, "symbol": symbol, "series_id": series_id,
            "bars": int(len(frame)), "label_cutoff": cutoff,
            "label_horizon_actual_bars": int(dq.label_horizon_bars),
            "mother_events": int(f["mother_event_v39"].sum()),
        })
        sampling_rows.append({
            "venue": venue, "symbol": symbol,
            "s0_clock_mother": int(flags["v44_sample_s0"].sum()),
            "s1_cusum": int(flags["v44_sample_s1"].sum()),
            "s2_directional_change": int(flags["v44_sample_s2"].sum()),
            "cusum_raw_events": int(flags["v44_cusum_event"].sum()),
            "dc_raw_events": int(flags["v44_dc_event"].sum()),
        })

    panel = pd.concat(panels, ignore_index=True, sort=False)
    event_frame = pd.concat(events, ignore_index=True, sort=False) if events else pd.DataFrame()
    if event_frame.empty:
        raise RuntimeError("v0.44 generated no settled S0 mother events")

    normalized = causal_robust_normalize(
        panel, NEURAL_FEATURES_V39, timestamp_col="timestamp", group_col="series_id"
    ).rename(columns={"timestamp": "signal_time"})
    feature_cols: list[str] = []
    for c in NEURAL_FEATURES_V39:
        feature_cols.extend([c, f"{c}__missing"])
    event_frame = event_frame.merge(
        normalized[["signal_time", "series_id", *feature_cols]],
        on=["signal_time", "series_id"], how="left", validate="many_to_one",
    )
    event_frame = event_frame.replace([np.inf, -np.inf], np.nan)
    event_frame[feature_cols] = event_frame[feature_cols].fillna(0.0).astype("float32")
    event_frame = attach_event_family_priority_v43(panel, event_frame)
    missing = sorted(set(V41_FEATURES) - set(event_frame.columns))
    if missing:
        raise RuntimeError(f"missing frozen v0.41 features: {missing}")
    event_frame.loc[:, list(V41_FEATURES)] = (
        event_frame.loc[:, list(V41_FEATURES)].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32")
    )
    for venue in V43_DEVELOPMENT_VENUES:
        event_frame[f"venue_{venue}_v43"] = event_frame["venue"].astype(str).eq(venue).astype("float32")

    event_frame = event_frame.sort_values(["signal_time", "symbol", "venue"], kind="mergesort").reset_index(drop=True)
    event_path = outdir / "events_v44.pkl.gz"
    event_frame.to_pickle(event_path, compression="gzip")
    pd.DataFrame(series_rows).to_csv(outdir / "data_manifest_v44.csv", index=False)
    pd.DataFrame(sampling_rows).to_csv(outdir / "sampling_counts_v44.csv", index=False)

    # IMPORTANT: common folds are frozen exactly once from S0 before any S1/S2 filtering.
    folds = base._folds(event_frame)
    fold_rows: list[dict] = []
    for fold in folds:
        test_start = pd.Timestamp(fold["test_start"])
        pretest_cut = test_start - EMBARGO_BARS * STEP
        fold_rows.append({
            "fold": int(fold["fold"]),
            "cal_start": pd.Timestamp(fold["cal_start"]),
            "pretest_cut": pretest_cut,
            "test_start": test_start,
            "test_end": pd.Timestamp(fold["test_end"]),
            "s0_fit_events": int(len(fold["fit"])),
            "s0_calibration_events": int(len(fold["cal"])),
            "s0_test_events": int(len(fold["test"])),
        })
    if len(fold_rows) != 5:
        raise RuntimeError(f"expected five common folds, got {len(fold_rows)}")
    fold_index = pd.DataFrame(fold_rows)
    fold_index.to_csv(outdir / "fold_index_v44.csv", index=False)

    # Descriptive counts only; no outcome is inspected.
    variant_counts = []
    for variant in V44_VARIANTS:
        col = variant_column_v44(variant)
        variant_counts.append({"variant": variant, "settled_events": int(event_frame[col].sum())})
    pd.DataFrame(variant_counts).to_csv(outdir / "variant_counts_v44.csv", index=False)

    hashes = {
        "preregistration_sha256": EXPECTED_PREREG_SHA256,
        "events_v44_pickle_sha256": _sha256(event_path),
        "fold_index_v44_sha256": _sha256(outdir / "fold_index_v44.csv"),
        "data_quality_manifest_v44_sha256": _sha256(outdir / "data_quality_manifest_v44.csv"),
        "data_manifest_v44_sha256": _sha256(outdir / "data_manifest_v44.csv"),
        "sampling_counts_v44_sha256": _sha256(outdir / "sampling_counts_v44.csv"),
    }
    (outdir / "prepared_hashes_v44.json").write_text(json.dumps(hashes, indent=2, sort_keys=True), encoding="utf-8")

    status = {
        "status": "READY",
        "eligible_asset_count": len(eligible_assets),
        "eligible_assets": list(eligible_assets),
        "s0_settled_events": int(len(event_frame)),
        "variant_counts": {r["variant"]: r["settled_events"] for r in variant_counts},
        "folds": 5,
        "common_fold_source": "S0_CLOCK_MOTHER_BASELINE",
        "kraken_touched": False,
        "paper_execution": False,
        "live_execution": False,
    }
    (outdir / "prep_status_v44.json").write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(status, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
