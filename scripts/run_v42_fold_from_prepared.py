from __future__ import annotations

"""Run exactly one frozen v0.42 purged walk-forward fold from prepared data."""

import argparse
import importlib.util
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FAST = ROOT / "scripts" / "run_v42_breadth_cluster_characterization_fast.py"
spec = importlib.util.spec_from_file_location("v42_fast_fold", FAST)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load v0.42 fast runner")
fast = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fast)
r = fast.runner


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--fold", type=int, required=True, choices=(1, 2, 3, 4, 5))
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    indir = Path(args.input_dir)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    status = json.loads((indir / "prep_status_v42.json").read_text(encoding="utf-8"))
    if status.get("status") != "READY":
        (outdir / f"fold_{args.fold}_skipped.json").write_text(
            json.dumps({"fold": args.fold, "status": "SKIPPED_DATA_UNAVAILABLE"}, indent=2),
            encoding="utf-8",
        )
        return

    events = pd.read_pickle(indir / "events_v42.pkl.gz", compression="gzip")
    folds = r.base._folds(events)
    fold = next((x for x in folds if int(x["fold"]) == int(args.fold)), None)
    if fold is None:
        raise RuntimeError(f"missing frozen fold {args.fold}")

    pred, seed_exps, seed_diags, fold_diag = r._predict_fold(fold)
    pred.to_pickle(outdir / f"oos_fold_{args.fold}_v42.pkl.gz", compression="gzip")
    pd.DataFrame(seed_diags).to_csv(
        outdir / f"seed_diags_fold_{args.fold}_v42.csv", index=False
    )
    (outdir / f"seed_expectancies_fold_{args.fold}_v42.json").write_text(
        json.dumps({str(k): r._jsonable(v) for k, v in seed_exps.items()}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (outdir / f"fold_diag_{args.fold}_v42.json").write_text(
        json.dumps(r._jsonable(fold_diag), indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({
        "fold": int(args.fold),
        "test_events": int(len(pred)),
        "selected_events": int(pred["selected_model"].sum()),
        "kraken_touched": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
