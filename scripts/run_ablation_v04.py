from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from research_bot.alpha_ablation import AblationConfig,build_point_in_time_features,run_ablation_panel
from research_bot.binance_vision import fetch_um_monthly_klines,fetch_um_daily_metrics,merge_archives_point_in_time

def clean(o):
    if isinstance(o,dict):return {k:clean(v) for k,v in o.items()}
    if isinstance(o,list):return [clean(v) for v in o]
    if isinstance(o,(np.floating,float)):return None if not np.isfinite(o) else float(o)
    if isinstance(o,(np.integer,int)):return int(o)
    if isinstance(o,pd.Timestamp):return o.isoformat()
    return o

def status(d):
    b=(d or {}).get('paired_block_bootstrap',{}); lo=b.get('ci_low'); q=(d or {}).get('bh_adjusted_p'); ds=(d or {}).get('delta_sharpe')
    if all(v is not None and np.isfinite(v) for v in (lo,q,ds)):
        if ds>0 and lo>0 and q<=.10:return 'candidate_incremental_alpha'
        if ds<=0 and q>.10:return 'no_incremental_evidence'
    return 'inconclusive'

def main():
    p=argparse.ArgumentParser(); p.add_argument('--v03-dir',default='artifacts/v03'); p.add_argument('--output-dir',default='artifacts/v04'); p.add_argument('--archive-start-month',default='2023-09'); p.add_argument('--oi-days',type=int,default=365); p.add_argument('--fee-bps',type=float,default=10); p.add_argument('--slippage-bps',type=float,default=2); p.add_argument('--threshold',type=float,default=.56); a=p.parse_args(); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    base=pd.read_csv(Path(a.v03_dir)/'alpha_frame_v03.csv',parse_dates=['timestamp']); errors={}; of=pd.DataFrame(); ofmeta={}; oi=pd.DataFrame(); oimeta={}
    try: of,ofmeta=fetch_um_monthly_klines('BTCUSDT','4h',a.archive_start_month,verify_checksum=True); of.to_csv(out/'binance_vision_orderflow_4h.csv',index=False)
    except Exception as e:errors['orderflow']=f'{type(e).__name__}: {e}'
    try:
        end=(pd.Timestamp.now(tz='UTC').normalize()-pd.Timedelta(days=1)).date(); start=end-pd.Timedelta(days=max(30,a.oi_days-1)); oi,oimeta=fetch_um_daily_metrics('BTCUSDT',start,end,12,False); oi.to_csv(out/'binance_vision_open_interest.csv',index=False)
    except Exception as e:errors['open_interest']=f'{type(e).__name__}: {e}'
    merged=merge_archives_point_in_time(base,of,oi); merged.to_csv(out/'alpha_frame_v04.csv',index=False); feat,fams=build_point_in_time_features(merged)
    cfg=AblationConfig(n_splits=5,min_train_size=2200,purge_bars=2,probability_threshold=a.threshold,fee_bps=a.fee_bps,slippage_bps=a.slippage_bps); variants={'baseline':['baseline'],'funding':['baseline','funding'],'basis':['baseline','basis'],'funding_basis':['baseline','funding','basis']}
    if fams['orderflow']:variants.update({'orderflow':['baseline','orderflow'],'funding_basis_orderflow':['baseline','funding','basis','orderflow']})
    summ,pred,stats=run_ablation_panel(feat,fams,variants,cfg); summ.to_csv(out/'long_panel_ablation_summary.csv',index=False); pred.to_csv(out/'long_panel_predictions.csv',index=False)
    oiresult=None
    if fams['open_interest']:
        n=int(feat[fams['open_interest']].notna().any(axis=1).sum())
        if n>=450:
            try:
                ocfg=AblationConfig(n_splits=3,min_train_size=max(250,min(1200,int(n*.55))),purge_bars=2,probability_threshold=a.threshold,fee_bps=a.fee_bps,slippage_bps=a.slippage_bps); os,op,ost=run_ablation_panel(feat,fams,{'baseline':['baseline'],'open_interest':['baseline','open_interest'],'funding_oi':['baseline','funding','open_interest']},ocfg); os.to_csv(out/'oi_panel_ablation_summary.csv',index=False); op.to_csv(out/'oi_panel_predictions.csv',index=False); oiresult={'summary':os.to_dict('records'),'stats':ost}
            except Exception as e:errors['oi_ablation']=f'{type(e).__name__}: {e}'
        else:errors['oi_ablation']=f'Insufficient point-in-time OI rows: {n}'
    famstatus={k:status(v) for k,v in stats.get('variants',{}).items()}
    if oiresult:
        famstatus.update({'oi_panel:'+k:status(v) for k,v in oiresult['stats'].get('variants',{}).items()})
    report={'research_status':'v0.4_historical_archive_alpha_ablation_not_production','method':{'target':'next 4h spot return > one-turnover fee+slippage hurdle','validation':'expanding walk-forward + purge; identical common dates inside each panel','model':'fixed HistGradientBoostingClassifier; no test-set tuning','execution':'long/flat fixed threshold with explicit turnover cost','inference':'paired moving-block bootstrap + Benjamini-Hochberg','point_in_time':'external derivatives lagged; daily OI conservatively available next day; no backfill'},'archive':{'orderflow':ofmeta,'open_interest':oimeta,'errors':errors},'feature_counts':{k:len(v) for k,v in fams.items()},'long_panel':{'summary':summ.to_dict('records'),'stats':stats},'oi_panel':oiresult,'family_status':famstatus,'warnings':['Ablation is not a profitability/live-trading claim.','Binance Vision data remain explicitly cross-venue.','Kline taker-buy imbalance is a coarse order-flow proxy, not L2/L3 microstructure.','Any candidate must later survive CPCV/PBO/Deflated Sharpe, regime stress, capacity and forward paper trading.']}; (out/'v04_ablation_report.json').write_text(json.dumps(clean(report),indent=2),encoding='utf-8'); print('===V04_ABLATION_REPORT==='); print(json.dumps(clean(report),indent=2)); print('===END_V04_ABLATION_REPORT===')
if __name__=='__main__':main()
