from __future__ import annotations

import argparse, json
from pathlib import Path
from typing import Any
import pandas as pd

from research_bot.walk_forward_v39 import DEVELOPMENT_VENUES_V39, decision_v39, preregistration_manifest_v39, walk_forward_v39


def write_json(path:Path,payload:Any)->None:
    path.write_text(json.dumps(payload,indent=2,sort_keys=True,default=str),encoding='utf-8')


def main()->None:
    ap=argparse.ArgumentParser(); ap.add_argument('--v38-artifact-dir',required=True); ap.add_argument('--output-dir',default='artifacts/v39-walk-forward')
    a=ap.parse_args(); src=Path(a.v38_artifact_dir); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    d38=json.loads((src/'decision_v38.json').read_text())
    if bool(d38.get('kraken_touched')): raise RuntimeError('Kraken seal violated before v0.39')
    screen=json.loads((src/'screen_v38.json').read_text())
    parent=bool(screen.get('development_eligible_v38',False))
    write_json(out/'preregistration_v39.json',preregistration_manifest_v39())
    results={}
    for venue in DEVELOPMENT_VENUES_V39:
        f=pd.read_csv(src/f'allocated_v38_{venue}.csv.gz',compression='gzip')
        results[venue]=walk_forward_v39(f)
    d=decision_v39(parent,results)
    write_json(out/'walk_forward_v39.json',results); write_json(out/'decision_v39.json',d)
    write_json(out/'evidence_v39.json',{'version':'v0.39','parent_v38_screen':screen,'walk_forward':results,'decision':d})
    print(json.dumps({'decision':d,'walk_forward':results},indent=2,default=str))

if __name__=='__main__': main()
