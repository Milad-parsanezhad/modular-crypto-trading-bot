from __future__ import annotations

import argparse,json
from pathlib import Path
from typing import Any

from research_bot.readiness_gate_v42 import decision_v42, operations_runbook_v42, preregistration_manifest_v42, readiness_matrix_v42


def load(path:Path)->dict[str,Any]: return json.loads(path.read_text(encoding='utf-8'))
def write(path:Path,p:Any)->None: path.write_text(json.dumps(p,indent=2,sort_keys=True,default=str),encoding='utf-8')


def main():
    ap=argparse.ArgumentParser()
    for v in ('v38','v39','v40','v41'): ap.add_argument(f'--{v}-artifact-dir',required=True)
    ap.add_argument('--output-dir',default='artifacts/v42-production-readiness')
    a=ap.parse_args(); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    d38=load(Path(a.v38_artifact_dir)/'decision_v38.json')
    d39=load(Path(a.v39_artifact_dir)/'decision_v39.json')
    d40=load(Path(a.v40_artifact_dir)/'decision_v40.json')
    d41=load(Path(a.v41_artifact_dir)/'decision_v41.json')
    matrix=readiness_matrix_v42(d38=d38,d39=d39,d40=d40,d41=d41); decision=decision_v42(matrix)
    write(out/'preregistration_v42.json',preregistration_manifest_v42())
    write(out/'readiness_matrix_v42.json',matrix); write(out/'decision_v42.json',decision)
    (out/'ROLLBACK_AND_KEY_MANAGEMENT_V42.md').write_text(operations_runbook_v42(),encoding='utf-8')
    write(out/'evidence_v42.json',{'version':'v0.42','parents':{'v38':d38,'v39':d39,'v40':d40,'v41':d41},'readiness_matrix':matrix,'decision':decision})
    print(json.dumps({'readiness_matrix':matrix,'decision':decision},indent=2,default=str))

if __name__=='__main__': main()
