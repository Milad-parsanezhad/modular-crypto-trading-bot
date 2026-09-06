from __future__ import annotations

import argparse, json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd

from research_bot.cross_sectional_v06 import (
    CrossSectionConfig, build_symbol_panel, run_cross_sectional_experiment,
    fetch_um_monthly_raw_klines,
)
from research_bot.binance_vision import fetch_um_daily_metrics
from research_bot.alpha_ablation import build_point_in_time_features
from research_bot.regime_robustness import RegimeRobustnessConfig, add_point_in_time_regimes, run_regime_horizon_panel

DEFAULT_UNIVERSE=[
    'BTCUSDT','ETHUSDT','BNBUSDT','SOLUSDT','XRPUSDT','DOGEUSDT',
    'ADAUSDT','LINKUSDT','AVAXUSDT','LTCUSDT','BCHUSDT','DOTUSDT'
]

def clean(o):
    if isinstance(o,dict): return {k:clean(v) for k,v in o.items()}
    if isinstance(o,list): return [clean(v) for v in o]
    if isinstance(o,(np.floating,float)): return None if not np.isfinite(o) else float(o)
    if isinstance(o,(np.integer,int)): return int(o)
    if isinstance(o,pd.Timestamp): return o.isoformat()
    return o


def independent_oi_confirmation(start_month:str,end_month:str|None,fee_bps:float,slippage_bps:float):
    px,meta=fetch_um_monthly_raw_klines('ETHUSDT','4h',start_month,end_month,True,'klines')
    if px.empty: return {'error':'ETHUSDT kline archive unavailable'}
    start=px.timestamp.min().date(); end=px.timestamp.max().date()
    oi,oimeta=fetch_um_daily_metrics('ETHUSDT',start,end,max_workers=12,verify_checksum=False)
    frame=pd.DataFrame({'timestamp':px.timestamp,'spot_close':px.close,'futures_close':px.close})
    if not oi.empty:
        frame['timestamp']=pd.to_datetime(frame['timestamp'],utc=True).astype('datetime64[ns, UTC]')
        oi=oi.copy(); oi['timestamp']=pd.to_datetime(oi['timestamp'],utc=True).astype('datetime64[ns, UTC]')
        frame=pd.merge_asof(frame.sort_values('timestamp'),oi.sort_values('timestamp'),on='timestamp',direction='backward')
    feat,fams=build_point_in_time_features(frame); feat=add_point_in_time_regimes(feat)
    if not fams.get('open_interest') or feat[fams['open_interest']].notna().any(axis=1).sum()<700:
        return {'error':'insufficient ETHUSDT OI rows','kline_meta':meta,'oi_meta':oimeta}
    cfg=RegimeRobustnessConfig(horizons=(1,),n_splits=3,min_train_size=700,fee_bps=fee_bps,slippage_bps=slippage_bps,bootstrap_runs=500,min_regime_obs=40)
    variants={'baseline':['baseline'],'open_interest':['baseline','open_interest']}
    summ,cond,comp,trans,rc,thr,pred,robust=run_regime_horizon_panel(feat,fams,variants,cfg)
    bear=comp[(comp.horizon==1)&(comp.regime=='bear')&(comp.variant=='open_interest')]
    return {'asset':'ETHUSDT','frozen_hypothesis':'OI × Bear regime × 4h','summary':summ.to_dict('records'),'bear_incremental_comparison':bear.to_dict('records'),'robustness':robust,'kline_meta':meta,'oi_meta':oimeta,'note':'Independent asset confirmation only; same venue family and frozen methodology as discovery.'}


def main():
    p=argparse.ArgumentParser(); p.add_argument('--start-month',default='2022-01'); p.add_argument('--end-month',default=None); p.add_argument('--output-dir',default='artifacts/v06'); p.add_argument('--symbols',default=','.join(DEFAULT_UNIVERSE)); p.add_argument('--fee-bps',type=float,default=4.0); p.add_argument('--slippage-bps',type=float,default=2.0); p.add_argument('--max-workers',type=int,default=6)
    a=p.parse_args(); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    symbols=[s.strip().upper() for s in a.symbols.split(',') if s.strip()]
    panels=[]; archive_meta={}; failures={}

    def load_symbol(s):
        try:
            fr,meta=build_symbol_panel(s,a.start_month,a.end_month,True)
            return s,fr,meta,None
        except Exception as exc:
            return s,pd.DataFrame(),{},f'{type(exc).__name__}: {exc}'

    with ThreadPoolExecutor(max_workers=max(1,min(a.max_workers,len(symbols)))) as ex:
        futs=[ex.submit(load_symbol,s) for s in symbols]
        for f in as_completed(futs):
            s,fr,meta,err=f.result()
            if err is not None:
                failures[s]=err
                continue
            archive_meta[s]=meta
            if fr.empty: failures[s]='empty panel'
            else: panels.append(fr)

    if len(panels)<6: raise RuntimeError(f'Need >=6 usable perpetuals, got {len(panels)}; failures={failures}')
    panel=pd.concat(panels,ignore_index=True).sort_values(['timestamp','symbol']); panel.to_csv(out/'cross_sectional_raw_panel.csv',index=False)
    cfg=CrossSectionConfig(start_month=a.start_month,end_month=a.end_month,n_splits=3,top_quantile=.20,one_way_cost_bps=a.fee_bps+a.slippage_bps)
    results=[]; preds=[]
    for h in (1,3):
        r,pr=run_cross_sectional_experiment(panel,cfg,horizon=h); results.append(r); preds.append(pr)
    summary=pd.concat(results,ignore_index=True); predictions=pd.concat(preds,ignore_index=True)
    summary.to_csv(out/'cross_sectional_summary.csv',index=False); predictions.to_csv(out/'cross_sectional_predictions.csv',index=False)
    oi=independent_oi_confirmation('2025-09',a.end_month,a.fee_bps,a.slippage_bps)
    report={'research_status':'v0.6_cross_sectional_perpetuals_independent_confirmation_not_production','universe_requested':symbols,'universe_usable':sorted(panel.symbol.unique().tolist()),'archive_failures':failures,'coverage':{'start':panel.timestamp.min().isoformat(),'end':panel.timestamp.max().isoformat(),'rows':int(len(panel)),'settlements':int(panel.timestamp.nunique())},'cross_sectional_design':{'clock':'completed 8h perpetual bars aligned to settlement availability','variants':['baseline market/liquidity/order-flow','baseline + funding/premium crowding'],'models':['ridge logistic','HistGradientBoosting'],'portfolio':'dollar-neutral top/bottom 20% cross-sectional score','one_way_cost_bps':a.fee_bps+a.slippage_bps,'labels':['8h forward return','24h forward return'],'validation':'expanding chronological folds; no test tuning'},'cross_sectional_summary':summary.to_dict('records'),'oi_independent_confirmation':oi,'archive_meta':archive_meta,'warnings':['Cross-sectional v0.6 is a research replication, not a live strategy.','The first panel uses 12 liquid perpetual candidates and will only scale to 19-30 after data-quality validation.','No deep model is allowed to replace simple baselines until incremental information is demonstrated.']}
    (out/'v06_report.json').write_text(json.dumps(clean(report),indent=2),encoding='utf-8'); print('===V06_REPORT==='); print(json.dumps(clean(report),indent=2)); print('===END_V06_REPORT===')

if __name__=='__main__': main()
