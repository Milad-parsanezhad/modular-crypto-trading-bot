from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from research_bot.soft_portfolio_v38 import (
    DEVELOPMENT_VENUES_V38,
    PARENT_V36,
    TOTAL_EFFECTIVE_TRIALS_V38,
    decision_v38,
    portfolio_metrics_v38,
    preregistration_manifest_v38,
    screen_three_venues_v38,
    soft_allocate_v38,
)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding='utf-8')


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument('--v36-artifact-dir', required=True)
    ap.add_argument('--v37-artifact-dir', required=True)
    ap.add_argument('--output-dir', default='artifacts/v38-soft-portfolio')
    args=ap.parse_args()
    s36=Path(args.v36_artifact_dir); s37=Path(args.v37_artifact_dir); out=Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    d36=json.loads((s36/'decision_v36.json').read_text())
    d37=json.loads((s37/'decision_v37.json').read_text())
    if d36.get('decision')!='NO_V36_ROBUST_EVENT_UTILITY_CANDIDATE':
        raise RuntimeError('v0.38 is preregistered against the frozen v0.36 rejection state')
    if d37.get('decision')!='NO_V37_JOINT_EVENT_PORTFOLIO_CANDIDATE':
        raise RuntimeError('v0.38 requires the frozen v0.37 hard-admission rejection')
    if bool(d36.get('kraken_touched')) or bool(d37.get('kraken_touched')):
        raise RuntimeError('Kraken seal violated before v0.38')

    write_json(out/'preregistration_v38.json', preregistration_manifest_v38())
    metrics={}; diagnostics={}
    for venue in DEVELOPMENT_VENUES_V38:
        src=s36/f'scored_{PARENT_V36}_{venue}.csv.gz'
        frame=pd.read_csv(src, compression='gzip')
        allocated,diag=soft_allocate_v38(frame)
        m=portfolio_metrics_v38(allocated)
        metrics[venue]=m; diagnostics[venue]=diag
        allocated.to_csv(out/f'allocated_v38_{venue}.csv.gz', index=False, compression='gzip')

    screen=screen_three_venues_v38(metrics)
    decision=decision_v38(screen)
    write_json(out/'metrics_v38.json', metrics)
    write_json(out/'diagnostics_v38.json', diagnostics)
    write_json(out/'screen_v38.json', screen)
    write_json(out/'decision_v38.json', decision)
    write_json(out/'evidence_v38.json', {
        'version':'v0.38','parent_v36':PARENT_V36,'total_effective_trials':TOTAL_EFFECTIVE_TRIALS_V38,
        'metrics':metrics,'screen':screen,'decision':decision,
    })
    print(json.dumps({'decision':decision,'screen':screen,'metrics':metrics,'diagnostics':diagnostics}, indent=2, default=str))


if __name__=='__main__':
    main()
