from __future__ import annotations

"""Prepare the frozen v0.42 development dataset once for parallel fold execution.

This is an execution/checkpoint optimization only. Eligibility, features, labels,
normalization, event families, folds, and holdout governance are delegated to the
already frozen v0.42 runner.
"""

import argparse
import importlib.util
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FAST = ROOT / "scripts" / "run_v42_breadth_cluster_characterization_fast.py"
spec = importlib.util.spec_from_file_location("v42_fast_prepare", FAST)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load v0.42 fast runner")
fast = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fast)
r = fast.runner


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    manifest = r.preregistration_manifest_v42()
    if manifest["kraken_touched"] is not False or manifest["reserved_holdout"] != "kraken":
        raise RuntimeError("v0.42 holdout governance invalid")

    frames, bars, availability = r._fetch_availability()
    availability.to_csv(outdir / "availability_v42.csv", index=False)

    try:
        eligible_symbols = r.select_common_symbols_v42(bars)
    except RuntimeError as exc:
        status = {
            "status": "DATA_UNAVAILABLE",
            "reason": str(exc),
            "kraken_touched": False,
            "paper_execution": False,
            "live_execution": False,
        }
        (outdir / "prep_status_v42.json").write_text(
            json.dumps(status, indent=2, sort_keys=True), encoding="utf-8"
        )
        pd.DataFrame(columns=["eligible_symbol"]).to_csv(
            outdir / "eligible_symbols_v42.csv", index=False
        )
        print(json.dumps(status, indent=2, sort_keys=True))
        return

    pd.DataFrame({"eligible_symbol": list(eligible_symbols)}).to_csv(
        outdir / "eligible_symbols_v42.csv", index=False
    )
    _, events, data_manifest = r._prepare_expanded_events(frames, eligible_symbols)
    data_manifest.to_csv(outdir / "data_manifest_v42.csv", index=False)
    if events.empty:
        raise RuntimeError("v0.42 generated no mother events")

    folds = r.base._folds(events)
    if len(folds) != r.PURGED_FOLDS:
        raise RuntimeError("v0.42 requires all five frozen folds")

    events.to_pickle(outdir / "events_v42.pkl.gz", compression="gzip")
    fold_index = [
        {
            "fold": int(f["fold"]),
            "cal_start": pd.Timestamp(f["cal_start"]).isoformat(),
            "test_start": pd.Timestamp(f["test_start"]).isoformat(),
            "test_end": pd.Timestamp(f["test_end"]).isoformat(),
            "fit_events": int(len(f["fit"])),
            "calibration_events": int(len(f["cal"])),
            "test_events": int(len(f["test"])),
        }
        for f in folds
    ]
    (outdir / "fold_index_v42.json").write_text(
        json.dumps(fold_index, indent=2, sort_keys=True), encoding="utf-8"
    )
    status = {
        "status": "READY",
        "eligible_symbol_count": int(len(eligible_symbols)),
        "event_count": int(len(events)),
        "fold_count": int(len(folds)),
        "implementation_route": "PARALLEL_FOLD_CHECKPOINTED",
        "kraken_touched": False,
        "paper_execution": False,
        "live_execution": False,
    }
    (outdir / "prep_status_v42.json").write_text(
        json.dumps(status, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(status, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
