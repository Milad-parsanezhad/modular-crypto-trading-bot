from __future__ import annotations

import argparse, json
from pathlib import Path
import pandas as pd

from research_bot.failure_attribution_v32 import attribute_holdout_failure, validate_v31_source


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v31-artifact-dir", required=True)
    ap.add_argument("--output-dir", default="artifacts/v32-failure-attribution")
    args = ap.parse_args()
    src, out = Path(args.v31_artifact_dir), Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    decision = json.loads((src / "decision_v31.json").read_text(encoding="utf-8"))
    metrics = json.loads((src / "holdout_metrics_v31.json").read_text(encoding="utf-8"))
    metrics["holdout_failures"] = decision.get("holdout_failures", [])
    validate_v31_source(decision)
    ledger = pd.read_csv(src / "kucoin_holdout_ledger_v31.csv")
    result = attribute_holdout_failure(ledger, metrics)
    result["source_v31_decision"] = decision
    (out / "failure_attribution_v32.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    pd.DataFrame(result["asset_attribution"]).to_csv(out / "asset_attribution_v32.csv", index=False)
    print(json.dumps({k: result[k] for k in ["decision","trades","asset_count","observed_positive_asset_fraction","negative_assets","worst_three_quarters"]}, indent=2))

if __name__ == "__main__":
    main()
