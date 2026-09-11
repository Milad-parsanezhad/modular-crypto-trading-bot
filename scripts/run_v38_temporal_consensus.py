from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from research_bot.temporal_consensus_v38 import (
    DEVELOPMENT_VENUES_V38,
    MODEL_NAMES_V38,
    V38_CANDIDATES,
    apply_candidate_v38,
    build_consensus_panel_v38,
    decision_v38,
    metrics_v38,
    preregistration_manifest_v38,
    screen_three_venues_v38,
    select_v38_winner,
)


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_sealed(v36_dir: Path, v34_dir: Path) -> None:
    d36 = _read_json(v36_dir / "decision_v36.json")
    if bool(d36.get("kraken_touched", True)):
        raise RuntimeError("v0.36 evidence does not preserve Kraken seal")
    found = False
    for p in sorted(v34_dir.glob("*.json")):
        try:
            x = _read_json(p)
        except Exception:
            continue
        if "holdout_used" in x or "kraken_touched" in x or x.get("version") == "v0.34":
            if bool(x.get("holdout_used", False)) or bool(x.get("kraken_touched", False)):
                raise RuntimeError("v0.34 evidence indicates Kraken was touched")
            found = True
    if not found:
        # v0.36 itself still guarantees no Kraken use; keep this diagnostic explicit.
        print("warning: no explicit v0.34 seal field found; relying on frozen v0.36 kraken_touched=false")


def _load_models(v36_dir: Path, venue: str) -> dict[str, pd.DataFrame]:
    out = {}
    for model in MODEL_NAMES_V38:
        p = v36_dir / f"scored_{model}_{venue}.csv.gz"
        if not p.exists():
            raise FileNotFoundError(p)
        out[model] = pd.read_csv(p)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v36-artifact-dir", required=True)
    ap.add_argument("--v34-artifact-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    v36_dir = Path(args.v36_artifact_dir)
    v34_dir = Path(args.v34_artifact_dir)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    _assert_sealed(v36_dir, v34_dir)

    prereg = preregistration_manifest_v38()
    (outdir / "preregistration_v38.json").write_text(json.dumps(prereg, indent=2, sort_keys=True), encoding="utf-8")

    rows = []
    all_metrics = {}
    for candidate in V38_CANDIDATES:
        metrics_by_venue = {}
        for venue in DEVELOPMENT_VENUES_V38:
            panel = build_consensus_panel_v38(_load_models(v36_dir, venue))
            scored = apply_candidate_v38(panel, candidate)
            metrics = metrics_v38(scored)
            metrics_by_venue[venue] = metrics
            scored.to_csv(outdir / f"scored_{candidate.name}_{venue}.csv.gz", index=False, compression="gzip")

        screen = screen_three_venues_v38(metrics_by_venue)
        row = {"strategy": candidate.name, **screen}
        for venue, metrics in metrics_by_venue.items():
            prefix = venue.replace("_consumed", "")
            for k, v in metrics.items():
                row[f"{prefix}_{k}"] = v
        rows.append(row)
        all_metrics[candidate.name] = metrics_by_venue

    summary = pd.DataFrame(rows).sort_values(
        ["development_eligible_v38", "robust_floor_block_ci_low_v38", "robust_floor_profit_factor_v38"],
        ascending=[False, False, False],
        kind="mergesort",
    )
    summary.to_csv(outdir / "development_summary_v38.csv", index=False)
    (outdir / "development_metrics_v38.json").write_text(json.dumps(all_metrics, indent=2, sort_keys=True), encoding="utf-8")

    winner = select_v38_winner(rows)
    decision = decision_v38(winner)
    decision["eligible_count"] = int(sum(bool(r["development_eligible_v38"]) for r in rows))
    (outdir / "decision_v38.json").write_text(json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps({"decision": decision, "top": summary.head(5).to_dict(orient="records")}, indent=2, default=str))


if __name__ == "__main__":
    main()
