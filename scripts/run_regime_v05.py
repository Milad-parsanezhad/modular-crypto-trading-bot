from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from research_bot.alpha_ablation import build_point_in_time_features
from research_bot.regime_robustness import (
    RegimeRobustnessConfig, add_point_in_time_regimes,
    run_regime_horizon_panel, cpcv_sanity_panel,
)

def clean(o):
    if isinstance(o,dict): return {k:clean(v) for k,v in o.items()}
    if isinstance(o,list): return [clean(v) for v in o]
    if isinstance(o,(np.floating,float)): return None if not np.isfinite(o) else float(o)
    if isinstance(o,(np.integer,int)): return int(o)
    if isinstance(o,pd.Timestamp): return o.isoformat()
    return o

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--v04-dir',default='artifacts/v04')
    p.add_argument('--output-dir',default='artifacts/v05')
    p.add_argument('--fee-bps',type=float,default=10)
    p.add_argument('--slippage-bps',type=float,default=2)
    p.add_argument('--bootstrap-runs',type=int,default=700)
    a=p.parse_args(); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    raw=pd.read_csv(Path(a.v04_dir)/'alpha_frame_v04.csv',parse_dates=['timestamp'])
    feat,fams=build_point_in_time_features(raw)
    regimes=add_point_in_time_regimes(feat)
    regime_counts=regimes['regime'].value_counts(dropna=False).to_dict()

    long_variants={
        'baseline':['baseline'],
        'basis':['baseline','basis'],
        'orderflow':['baseline','orderflow'],
        'funding':['baseline','funding'],
        'basis_orderflow':['baseline','basis','orderflow'],
    }
    cfg=RegimeRobustnessConfig(
        horizons=(1,3,6), n_splits=4, min_train_size=2400,
        fee_bps=a.fee_bps, slippage_bps=a.slippage_bps,
        bootstrap_runs=a.bootstrap_runs,
    )
    summ,cond,comp,trans,rc,thr,pred,robust=run_regime_horizon_panel(feat,fams,long_variants,cfg)
    summ.to_csv(out/'long_summary.csv',index=False); cond.to_csv(out/'regime_conditional_summary.csv',index=False)
    comp.to_csv(out/'incremental_comparisons.csv',index=False); trans.to_csv(out/'transition_audit.csv',index=False)
    rc.to_csv(out/'risk_coverage_diagnostic.csv',index=False); thr.to_csv(out/'train_only_thresholds.csv',index=False)
    pred.to_csv(out/'oos_predictions.csv',index=False)

    cpcv={}
    for h in (1,3,6):
        panel=cpcv_sanity_panel(feat,fams,long_variants,horizon=h,fee_bps=a.fee_bps,slippage_bps=a.slippage_bps,n_groups=6,test_groups=2)
        panel.to_csv(out/f'cpcv_h{h}.csv',index=False)
        cpcv[f'h{h}']={v:{'median_sharpe':float(g.sharpe.median()),'positive_split_share':float((g.sharpe>0).mean()),'n_splits':int(len(g))} for v,g in panel.groupby('variant')}

    oi_report=None
    if fams.get('open_interest') and int(feat[fams['open_interest']].notna().any(axis=1).sum())>=900:
        oi_variants={'baseline':['baseline'],'open_interest':['baseline','open_interest'],'funding_oi':['baseline','funding','open_interest']}
        ocfg=RegimeRobustnessConfig(horizons=(1,3,6),n_splits=3,min_train_size=850,fee_bps=a.fee_bps,slippage_bps=a.slippage_bps,bootstrap_runs=max(350,a.bootstrap_runs//2),min_regime_obs=40)
        os,oc,oi_cmp,ot,orc,oth,op,orob=run_regime_horizon_panel(feat,fams,oi_variants,ocfg)
        os.to_csv(out/'oi_summary.csv',index=False); oc.to_csv(out/'oi_regime_conditional_summary.csv',index=False); oi_cmp.to_csv(out/'oi_incremental_comparisons.csv',index=False); op.to_csv(out/'oi_oos_predictions.csv',index=False)
        oi_report={'summary':os.to_dict('records'),'comparisons':oi_cmp.to_dict('records'),'robustness':orob}

    candidates=[]
    if not comp.empty:
        candidates=comp[(comp.status=='candidate')].sort_values(['bh_adjusted_p','delta_mean_net_return'],ascending=[True,False]).to_dict('records')
    top_clues=[]
    if not comp.empty:
        top_clues=comp.sort_values('delta_mean_net_return',ascending=False).head(12).to_dict('records')
    report={
        'research_status':'v0.5_regime_conditional_robustness_not_production',
        'pre_specification':{
            'regimes':'trailing 42-bar trend/volatility with lagged rolling quantile cutoffs: bull, bear, sideways, high_vol, crisis',
            'horizons':{'1':'4h','3':'12h','6':'24h'},
            'variants':long_variants,
            'selection':'model fit, Platt calibration and trade-threshold selection are all performed inside outer-training data only',
            'execution':'non-overlapping horizon events; long/flat; round-trip fee+slippage charged per selected event',
            'inference':'paired moving-block bootstrap, BH correction, coverage-matched selective diagnostic, transition audit, CPCV sanity, PBO and DSR diagnostics',
        },
        'regime_counts':regime_counts,
        'long_summary':summ.to_dict('records'),
        'conditional_summary':cond.to_dict('records'),
        'incremental_comparisons':comp.to_dict('records'),
        'validated_candidates':candidates,
        'top_exploratory_clues':top_clues,
        'robustness':robust,
        'cpcv':cpcv,
        'oi_panel':oi_report,
        'warnings':[
            'A regime label is a hypothesis, not evidence of alpha.',
            'Risk-coverage curves are diagnostic only and never feed threshold selection.',
            'Coverage-matched comparisons are included to detect selective-prediction/persistence illusions.',
            'No feature family or regime may be promoted unless multiplicity-adjusted evidence and economic robustness agree.',
            'No live trading is authorized by v0.5.'
        ]
    }
    (out/'v05_regime_report.json').write_text(json.dumps(clean(report),indent=2),encoding='utf-8')
    print('===V05_REGIME_REPORT==='); print(json.dumps(clean(report),indent=2)); print('===END_V05_REGIME_REPORT===')

if __name__=='__main__': main()
