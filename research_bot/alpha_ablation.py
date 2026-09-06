from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline

@dataclass(frozen=True)
class AblationConfig:
    n_splits:int=5; min_train_size:int=1800; purge_bars:int=2; test_size:int|None=None
    probability_threshold:float=0.56; fee_bps:float=10.0; slippage_bps:float=2.0; random_state:int=42

def build_point_in_time_features(frame:pd.DataFrame):
    x=frame.copy(); x['timestamp']=pd.to_datetime(x['timestamp'],utc=True); x=x.sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)
    spot=pd.to_numeric(x['spot_close'],errors='coerce'); fut=pd.to_numeric(x.get('futures_close'),errors='coerce'); ret=spot.pct_change()
    x['mkt_ret_1']=ret; x['mkt_logret_1']=np.log(spot).diff()
    for w in (3,6,12,24,48): x[f'mkt_mom_{w}']=spot.pct_change(w)
    for w in (6,12,24,48): x[f'mkt_vol_{w}']=ret.rolling(w).std()
    for w in (12,24,48,96): x[f'mkt_ma_gap_{w}']=spot/spot.rolling(w).mean()-1
    x['mkt_fut_mom_6']=fut.pct_change(6); x['mkt_fut_mom_24']=fut.pct_change(24)
    baseline=[c for c in x if c.startswith('mkt_')]
    funding=[]
    if 'actual_funding_rate' in x:
        raw=pd.to_numeric(x['actual_funding_rate'],errors='coerce').ffill().shift(1); theo=pd.to_numeric(x.get('theoretical_funding_rate'),errors='coerce').ffill().shift(1)
        x['fund_last']=raw; x['fund_theoretical_last']=theo; x['fund_spread']=theo-raw
        for w in (3,6,12,42): x[f'fund_mean_{w}']=raw.rolling(w).mean(); x[f'fund_sum_{w}']=raw.rolling(w).sum()
        funding=[c for c in x if c.startswith('fund_')]
    basis=[]
    if 'realized_basis_rate' in x:
        b=pd.to_numeric(x['realized_basis_rate'],errors='coerce').shift(1); x['basis_last']=b; x['basis_change_1']=b.diff()
        for w in (6,24,42,126):
            mu,sd=b.rolling(w).mean(),b.rolling(w).std(); x[f'basis_z_{w}']=(b-mu)/sd.replace(0,np.nan); x[f'basis_mean_{w}']=mu
        basis=['basis_last','basis_change_1']+[c for c in x if c.startswith('basis_z_') or c.startswith('basis_mean_')]
    orderflow=[]
    if 'binance_of_quote_imbalance' in x:
        q=pd.to_numeric(x['binance_of_quote_imbalance'],errors='coerce').shift(1); x['of_quote_imbalance']=q; x['of_change_1']=q.diff()
        for w in (3,6,24,42): x[f'of_mean_{w}']=q.rolling(w).mean(); x[f'of_std_{w}']=q.rolling(w).std()
        if 'binance_taker_buy_quote' in x and 'binance_quote_volume' in x:
            buy=pd.to_numeric(x['binance_taker_buy_quote'],errors='coerce').shift(1); total=pd.to_numeric(x['binance_quote_volume'],errors='coerce').shift(1); x['of_buy_share']=buy/total.replace(0,np.nan)
        orderflow=[c for c in x if c.startswith('of_')]
    oi=[]
    if 'binance_open_interest' in x:
        raw=pd.to_numeric(x['binance_open_interest'],errors='coerce').shift(1); val=pd.to_numeric(x.get('binance_open_interest_value'),errors='coerce').shift(1)
        x['oi_log']=np.log(raw.where(raw>0)); x['oi_log_change_1']=x['oi_log'].diff(); x['oi_value_log']=np.log(val.where(val>0)); x['oi_value_change_1']=x['oi_value_log'].diff()
        for w in (6,24,42):
            mu,sd=x['oi_log'].rolling(w).mean(),x['oi_log'].rolling(w).std(); x[f'oi_z_{w}']=(x['oi_log']-mu)/sd.replace(0,np.nan)
        oi=[c for c in x if c.startswith('oi_')]
    x['future_return']=spot.shift(-1)/spot-1; x['target_up']=(x['future_return']>0).astype(int); x=x.replace([np.inf,-np.inf],np.nan)
    return x,{'baseline':baseline,'funding':funding,'basis':basis,'orderflow':orderflow,'open_interest':oi}

def _walk(n,cfg):
    gap=max(1,cfg.purge_bars); avail=n-cfg.min_train_size-gap
    if avail<=50: raise ValueError(f'Not enough rows ({n})')
    test=cfg.test_size or max(50,avail//cfg.n_splits); end=cfg.min_train_size; k=0
    while k<cfg.n_splits:
        s=end+gap; e=min(n,s+test)
        if e-s<25: break
        yield np.arange(end),np.arange(s,e); end=e; k+=1
        if end+gap>=n: break

def _perf(r,pos,ppy=6*365):
    r=pd.Series(r).fillna(0).astype(float); pos=pd.Series(pos,index=r.index).fillna(0); eq=(1+r).cumprod(); dd=eq/eq.cummax()-1; sd=r.std(ddof=1)
    return {'n':len(r),'total_return':float(eq.iloc[-1]-1),'sharpe':float(r.mean()/sd*math.sqrt(ppy)) if sd>0 else np.nan,'max_drawdown':float(dd.min()),'hit_rate':float((r[pos>0]>0).mean()) if (pos>0).any() else np.nan,'trade_exposure':float(pos.abs().mean()),'turnover':float(pos.diff().abs().fillna(pos.abs()).sum())}

def _bh(p):
    vals=sorted([(k,float(v)) for k,v in p.items() if np.isfinite(v)],key=lambda z:z[1]); m=len(vals); out={}; prev=1.0
    for rank,(k,v) in reversed(list(enumerate(vals,1))): out[k]=prev=min(prev,v*m/rank,1.0)
    return out

def moving_block_bootstrap_mean_diff(diff,block_size=24,n_boot=1000,seed=42):
    x=pd.Series(diff).dropna().to_numpy(float); n=len(x)
    if n<max(50,2*block_size): return {'n':n,'mean':float(x.mean()) if n else np.nan,'ci_low':np.nan,'ci_high':np.nan,'p_one_sided':np.nan}
    rng=np.random.default_rng(seed); starts=np.arange(n-block_size+1); means=[]; nb=int(np.ceil(n/block_size))
    for _ in range(n_boot):
        chosen=rng.choice(starts,size=nb,replace=True); sample=np.concatenate([x[s:s+block_size] for s in chosen])[:n]; means.append(sample.mean())
    means=np.asarray(means)
    return {'n':n,'mean':float(x.mean()),'ci_low':float(np.quantile(means,.025)),'ci_high':float(np.quantile(means,.975)),'p_one_sided':float((1+(means<=0).sum())/(n_boot+1))}

def run_ablation_panel(features,family_map,variants,cfg):
    common=features.dropna(subset=['future_return']).copy(); common['target_up']=(common['future_return']>(cfg.fee_bps+cfg.slippage_bps)/10000).astype(int)
    req=sorted(set(f for fams in variants.values() for f in fams))
    for fam in req:
        cols=family_map.get(fam,[])
        if not cols: raise ValueError(f"Family '{fam}' has no available features")
        common=common[common[cols].notna().any(axis=1)]
    common=common.reset_index(drop=True)
    if len(common)<=cfg.min_train_size+cfg.purge_bars+50: raise ValueError(f'Common sample too short: {len(common)}')
    rows=[]
    for fold,(tr,te) in enumerate(_walk(len(common),cfg),1):
        train,test=common.iloc[tr],common.iloc[te]; y=train['target_up'].astype(int)
        if y.nunique()<2: continue
        for name,fams in variants.items():
            cols=[]
            for fam in fams: cols+=family_map[fam]
            cols=list(dict.fromkeys(cols)); pipe=Pipeline([('imputer',SimpleImputer(strategy='median')),('model',HistGradientBoostingClassifier(learning_rate=.04,max_iter=250,max_leaf_nodes=15,l2_regularization=1,random_state=cfg.random_state))])
            pipe.fit(train[cols],y); prob=pipe.predict_proba(test[cols])[:,1]; pos=(prob>=cfg.probability_threshold).astype(float); turnover=np.abs(np.diff(np.r_[0.,pos])); cost=turnover*(cfg.fee_bps+cfg.slippage_bps)/10000; net=pos*test['future_return'].to_numpy(float)-cost
            for i,idx in enumerate(te): rows.append({'timestamp':common.iloc[idx]['timestamp'],'fold':fold,'variant':name,'prob_up':float(prob[i]),'position':float(pos[i]),'gross_return':float(pos[i]*test.iloc[i]['future_return']),'cost':float(cost[i]),'net_return':float(net[i]),'target_up':int(test.iloc[i]['target_up']),'future_return':float(test.iloc[i]['future_return'])})
    pred=pd.DataFrame(rows)
    if pred.empty: raise RuntimeError('No walk-forward predictions produced')
    summary=[]
    for name,g in pred.groupby('variant'):
        d=_perf(g['net_return'],g['position']); yy=g['target_up'].to_numpy(); pp=np.clip(g['prob_up'].to_numpy(),1e-6,1-1e-6); d.update({'variant':name,'rows':len(g),'auc':float(roc_auc_score(yy,pp)) if len(np.unique(yy))>1 else np.nan,'brier':float(brier_score_loss(yy,pp)),'log_loss':float(log_loss(yy,pp,labels=[0,1]))}); summary.append(d)
    summary=pd.DataFrame(summary).set_index('variant').sort_index(); stats={'common_rows':len(common),'coverage_start':common['timestamp'].min().isoformat(),'coverage_end':common['timestamp'].max().isoformat(),'variants':{}}
    if 'baseline' in summary.index:
        base=pred[pred.variant=='baseline'].set_index(['timestamp','fold'])['net_return']; pvals={}
        for name in summary.index:
            if name=='baseline': continue
            vr=pred[pred.variant==name].set_index(['timestamp','fold'])['net_return']; a=pd.concat([base.rename('b'),vr.rename('v')],axis=1).dropna(); boot=moving_block_bootstrap_mean_diff(a.v-a.b,seed=cfg.random_state)
            stats['variants'][name]={'delta_total_return':float(summary.loc[name,'total_return']-summary.loc['baseline','total_return']),'delta_sharpe':float(summary.loc[name,'sharpe']-summary.loc['baseline','sharpe']),'delta_auc':float(summary.loc[name,'auc']-summary.loc['baseline','auc']),'paired_block_bootstrap':boot}; pvals[name]=boot['p_one_sided']
        for name,q in _bh(pvals).items(): stats['variants'][name]['bh_adjusted_p']=float(q)
    return summary.reset_index(),pred,stats
