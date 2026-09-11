from __future__ import annotations

import argparse,json
from pathlib import Path

from research_bot.shadow_execution_v41 import LiveExecutionForbidden, OrderIntentV41, ShadowExecutionEngineV41, decision_v41, preregistration_manifest_v41


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--v40-artifact-dir',required=True); ap.add_argument('--output-dir',default='artifacts/v41-shadow-execution')
    a=ap.parse_args(); src=Path(a.v40_artifact_dir); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    d40=json.loads((src/'decision_v40.json').read_text()); parent=d40.get('decision')=='V40_EXECUTION_ROBUST_CANDIDATE_LOCKED_FOR_SHADOW'
    if bool(d40.get('kraken_touched')): raise RuntimeError('Kraken seal violated before v0.41')
    (out/'preregistration_v41.json').write_text(json.dumps(preregistration_manifest_v41(),indent=2,sort_keys=True),encoding='utf-8')
    e=ShadowExecutionEngineV41()
    shadow=OrderIntentV41('BTC/USDT','buy',0.01,50000.0,'2026-09-11T20:00:00Z','V41_SHADOW_SMOKE')
    paper=OrderIntentV41('ETH/USDT','sell',0.10,3000.0,'2026-09-11T20:00:01Z','V41_PAPER_SMOKE')
    shadow_event=e.submit_shadow(shadow)
    fill=e.submit_paper(paper)
    rec=e.reconcile({'ETH/USDT':-0.1})
    live_blocked=False
    try:
        e.submit_live(shadow)
    except LiveExecutionForbidden:
        live_blocked=True
    engine_valid=bool(shadow_event.get('transmitted') is False and fill.mode=='paper' and rec.get('ok') and live_blocked)
    d=decision_v41(parent,engine_valid)
    (out/'audit_v41.jsonl').write_text(e.audit_jsonl(),encoding='utf-8')
    (out/'engine_validation_v41.json').write_text(json.dumps({'engine_validated':engine_valid,'live_blocked':live_blocked,'shadow_event':shadow_event,'paper_fill':fill.__dict__,'reconciliation':rec},indent=2,sort_keys=True,default=str),encoding='utf-8')
    (out/'decision_v41.json').write_text(json.dumps(d,indent=2,sort_keys=True),encoding='utf-8')
    print(json.dumps({'decision':d,'engine_validated':engine_valid,'live_blocked':live_blocked},indent=2))

if __name__=='__main__': main()
