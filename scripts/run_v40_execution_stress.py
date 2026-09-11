from __future__ import annotations

import argparse,json
from pathlib import Path
from typing import Any
import pandas as pd

from research_bot.execution_stress_v40 import DEVELOPMENT_VENUES_V40, SCENARIOS_V40, apply_execution_scenario_v40, metrics_v40, preregistration_manifest_v40, screen_v40, decision_v40


def w(path:Path,p:Any): path.write_text(json.dumps(p,indent=2,sort_keys=True,default=str),encoding='utf-8')


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--v38-artifact-dir',required=True); ap.add_argument('--v39-artifact-dir',required=True); ap.add_argument('--output-dir',default='artifacts/v40-execution-stress')
    a=ap.parse_args(); s38=Path(a.v38_artifact_dir); s39=Path(a.v39_artifact_dir); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    d39=json.loads((s39/'decision_v39.json').read_text()); parent=d39.get('decision')=='V39_TEMPORALLY_STABLE_CANDIDATE_LOCKED_FOR_EXECUTION_STRESS'
    if bool(d39.get('kraken_touched')): raise RuntimeError('Kraken seal violated before v0.40')
    w(out/'preregistration_v40.json',preregistration_manifest_v40())
    results={}
    for v in DEVELOPMENT_VENUES_V40:
        base=pd.read_csv(s38/f'allocated_v38_{v}.csv.gz',compression='gzip')
        results[v]={}
        for scenario in SCENARIOS_V40:
            stressed=apply_execution_scenario_v40(base,scenario)
            results[v][scenario.name]=metrics_v40(stressed)
    screen=screen_v40(parent,results); d=decision_v40(screen)
    w(out/'stress_metrics_v40.json',results); w(out/'screen_v40.json',screen); w(out/'decision_v40.json',d)
    w(out/'evidence_v40.json',{'version':'v0.40','parent_v39':d39,'stress':results,'screen':screen,'decision':d})
    print(json.dumps({'decision':d,'screen':screen,'stress':results},indent=2,default=str))

if __name__=='__main__': main()
