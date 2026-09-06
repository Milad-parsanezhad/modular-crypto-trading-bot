from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from statistics import NormalDist
import math
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class RegimeRobustnessConfig:
    horizons: tuple[int, ...] = (1, 3, 6)
    n_splits: int = 4
    min_train_size: int = 2400
    regime_window: int = 756
    fee_bps: float = 10.0
    slippage_bps: float = 2.0
    threshold_grid: tuple[float, ...] = (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65)
    random_state: int = 42
    min_regime_obs: int = 60
    bootstrap_block: int = 18
    bootstrap_runs: int = 700


def add_point_in_time_regimes(frame: pd.DataFrame, window: int = 756) -> pd.DataFrame:
    """Deterministic regimes using only trailing observations and lagged rolling quantiles."""
    x = frame.copy()
    x['timestamp'] = pd.to_datetime(x['timestamp'], utc=True)
    x = x.sort_values('timestamp').reset_index(drop=True)
    close = pd.to_numeric(x['spot_close'], errors='coerce')
    ret1 = close.pct_change()
    trend = close.pct_change(42)
    vol = ret1.rolling(42).std()
    mp = max(252, window // 3)
    q10 = trend.rolling(window, min_periods=mp).quantile(.10).shift(1)
    q30 = trend.rolling(window, min_periods=mp).quantile(.30).shift(1)
    q70 = trend.rolling(window, min_periods=mp).quantile(.70).shift(1)
    v80 = vol.rolling(window, min_periods=mp).quantile(.80).shift(1)
    v90 = vol.rolling(window, min_periods=mp).quantile(.90).shift(1)
    crisis = (trend <= q10) & (vol >= v90)
    high_vol = (vol >= v80) & ~crisis
    bull = (trend >= q70) & ~crisis & ~high_vol
    bear = (trend <= q30) & ~crisis & ~high_vol
    regime = np.select([crisis, high_vol, bull, bear], ['crisis','high_vol','bull','bear'], default='sideways')
    available = q30.notna() & v80.notna()
    x['regime'] = regime
    x.loc[~available, 'regime'] = 'unknown'
    x['regime_transition'] = (x['regime'] != x['regime'].shift(1)) & (x['regime'] != 'unknown')
    x['regime_trend_42'] = trend.shift(1)
    x['regime_vol_42'] = vol.shift(1)
    return x


def _outer_walk(n: int, n_splits: int, min_train: int, gap: int):
    avail = n - min_train - gap
    if avail < 200:
        raise ValueError(f'Not enough rows for outer walk-forward: {n}')
    test = max(80, avail // n_splits)
    train_end = min_train
    for fold in range(1, n_splits + 1):
        s = train_end + gap
        e = min(n, s + test)
        if e - s < 50:
            break
        yield fold, np.arange(train_end), np.arange(s, e)
        train_end = e
        if train_end + gap >= n:
            break


def _three_way_train_split(idx: np.ndarray, gap: int):
    n = len(idx)
    fit_end = int(n * .65)
    cal_end = int(n * .82)
    fit = idx[:fit_end]
    cal = idx[min(n, fit_end + gap):cal_end]
    threshold = idx[min(n, cal_end + gap):]
    if min(len(fit), len(cal), len(threshold)) < 80:
        raise ValueError('Insufficient rows for fit/calibration/threshold temporal split')
    return fit, cal, threshold


def _pipe(seed: int):
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', HistGradientBoostingClassifier(learning_rate=.045, max_iter=160, max_leaf_nodes=15, l2_regularization=1.0, random_state=seed)),
    ])


def _fit_platt(raw_prob: np.ndarray, y: np.ndarray):
    p = np.clip(np.asarray(raw_prob, float), 1e-6, 1-1e-6)
    if len(np.unique(y)) < 2:
        return None
    z = np.log(p / (1-p)).reshape(-1,1)
    lr = LogisticRegression(C=1.0, solver='lbfgs', random_state=0)
    lr.fit(z, y)
    return lr


def _apply_platt(model, raw_prob):
    p = np.clip(np.asarray(raw_prob, float), 1e-6, 1-1e-6)
    if model is None:
        return p
    z = np.log(p / (1-p)).reshape(-1,1)
    return model.predict_proba(z)[:,1]


def _ann_sharpe(r: np.ndarray, horizon: int) -> float:
    x = pd.Series(r).dropna().astype(float)
    if len(x) < 2 or x.std(ddof=1) <= 0:
        return np.nan
    return float(x.mean()/x.std(ddof=1)*math.sqrt((6*365)/horizon))


def _choose_threshold(prob, future_return, row_positions, horizon, cfg):
    p = np.asarray(prob, float); fr = np.asarray(future_return, float); rp = np.asarray(row_positions, int)
    keep = (rp % horizon) == 0
    p, fr = p[keep], fr[keep]
    roundtrip = 2.0*(cfg.fee_bps + cfg.slippage_bps)/10000.0
    best = None
    min_trades = max(10, int(.04*len(p)))
    dynamic = [float(np.quantile(p,q)) for q in (.50,.60,.70,.80,.90,.95)] if len(p) else []
    thresholds = sorted(set(round(float(t),6) for t in (*cfg.threshold_grid,*dynamic) if .05 < float(t) < .95))
    for t in thresholds:
        pos = (p >= t).astype(float)
        trades = int(pos.sum())
        if trades < min_trades:
            continue
        net = pos*fr - pos*roundtrip
        sh = _ann_sharpe(net, horizon)
        if not np.isfinite(sh):
            continue
        item = (sh, float(t), trades, float(pos.mean()))
        if best is None or item[0] > best[0] or (item[0] == best[0] and item[1] > best[1]):
            best = item
    if best is None:
        return float(np.quantile(p,.80)) if len(p) else 0.50, {'selected_sharpe': np.nan, 'selected_trades': 0, 'selected_coverage': 0.0}
    return best[1], {'selected_sharpe':best[0], 'selected_trades':best[2], 'selected_coverage':best[3]}


def _ece(y, p, bins=10):
    y=np.asarray(y,int); p=np.asarray(p,float); e=0.0
    edges=np.linspace(0,1,bins+1)
    for a,b in zip(edges[:-1],edges[1:]):
        m=(p>=a)&((p<b) if b<1 else (p<=b))
        if m.any(): e += m.mean()*abs(p[m].mean()-y[m].mean())
    return float(e)


def _perf(g: pd.DataFrame, horizon: int):
    r=g['net_return'].to_numpy(float); pos=g['position'].to_numpy(float)
    eq=np.cumprod(1+r); dd=eq/np.maximum.accumulate(eq)-1
    y=g['target_up'].to_numpy(int); p=np.clip(g['prob_up'].to_numpy(float),1e-6,1-1e-6)
    return {
        'rows':int(len(g)), 'trades':int(pos.sum()), 'coverage':float(pos.mean()),
        'total_return':float(eq[-1]-1) if len(eq) else np.nan,
        'sharpe':_ann_sharpe(r,horizon), 'max_drawdown':float(dd.min()) if len(dd) else np.nan,
        'auc':float(roc_auc_score(y,p)) if len(np.unique(y))>1 else np.nan,
        'brier':float(brier_score_loss(y,p)) if len(y) else np.nan,
        'log_loss':float(log_loss(y,p,labels=[0,1])) if len(y) else np.nan,
        'ece':_ece(y,p),
        'selective_accuracy':float((y[pos>0]==1).mean()) if (pos>0).any() else np.nan,
    }


def moving_block_bootstrap(diff, block=18, runs=700, seed=42):
    x=pd.Series(diff).dropna().to_numpy(float); n=len(x)
    if n < max(40,2*block): return {'n':n,'mean':float(x.mean()) if n else np.nan,'ci_low':np.nan,'ci_high':np.nan,'p_one_sided':np.nan}
    rng=np.random.default_rng(seed); starts=np.arange(n-block+1); means=[]; nb=int(np.ceil(n/block))
    for _ in range(runs):
        s=rng.choice(starts,size=nb,replace=True); sample=np.concatenate([x[i:i+block] for i in s])[:n]; means.append(sample.mean())
    means=np.asarray(means)
    return {'n':n,'mean':float(x.mean()),'ci_low':float(np.quantile(means,.025)),'ci_high':float(np.quantile(means,.975)),'p_one_sided':float((1+(means<=0).sum())/(runs+1))}


def benjamini_hochberg(pvals: dict[str,float]):
    vals=sorted([(k,float(v)) for k,v in pvals.items() if np.isfinite(v)],key=lambda z:z[1]); m=len(vals); out={}; prev=1.0
    for rank,(k,v) in reversed(list(enumerate(vals,1))):
        prev=min(prev,v*m/rank,1.0); out[k]=prev
    return out


def deflated_sharpe_probability(returns, num_trials:int, trial_sharpe_std:float=0.0):
    r=pd.Series(returns).dropna().astype(float); n=len(r)
    if n<30 or r.std(ddof=1)<=0: return np.nan
    sr=float(r.mean()/r.std(ddof=1)); skew=float(r.skew()); kurt=float(r.kurt()+3.0)
    if num_trials<=1 or not np.isfinite(trial_sharpe_std) or trial_sharpe_std<=0:
        benchmark=0.0
    else:
        nd=NormalDist(); gamma=.5772156649015329; N=float(num_trials)
        benchmark=trial_sharpe_std*((1-gamma)*nd.inv_cdf(1-1/N)+gamma*nd.inv_cdf(1-1/(N*math.e)))
    denom=math.sqrt(max(1e-12,1-skew*sr+((kurt-1)/4.0)*sr*sr))
    z=(sr-benchmark)*math.sqrt(n-1)/denom
    return float(NormalDist().cdf(z))


def probability_backtest_overfitting(return_matrix: pd.DataFrame, n_groups:int=6):
    r=return_matrix.dropna().copy()
    if len(r)<n_groups*20 or r.shape[1]<2: return {'pbo':np.nan,'splits':0}
    groups=np.array_split(np.arange(len(r)),n_groups); lambdas=[]
    k=n_groups//2
    for tr_groups in combinations(range(n_groups),k):
        tr_idx=np.concatenate([groups[i] for i in tr_groups]); te_groups=[i for i in range(n_groups) if i not in tr_groups]; te_idx=np.concatenate([groups[i] for i in te_groups])
        is_sr=r.iloc[tr_idx].mean()/r.iloc[tr_idx].std(ddof=1).replace(0,np.nan); os_sr=r.iloc[te_idx].mean()/r.iloc[te_idx].std(ddof=1).replace(0,np.nan)
        if is_sr.dropna().empty or os_sr.dropna().empty: continue
        winner=is_sr.idxmax(); vals=os_sr.dropna().sort_values(); rank=int(np.where(vals.index==winner)[0][0])+1 if winner in vals.index else 1
        omega=(rank-.5)/len(vals); omega=min(max(omega,1e-6),1-1e-6); lambdas.append(math.log(omega/(1-omega)))
    if not lambdas: return {'pbo':np.nan,'splits':0}
    return {'pbo':float((np.asarray(lambdas)<=0).mean()),'splits':len(lambdas),'median_logit_rank':float(np.median(lambdas))}


def run_regime_horizon_panel(features:pd.DataFrame, family_map:dict, variants:dict, cfg:RegimeRobustnessConfig):
    x=add_point_in_time_regimes(features,cfg.regime_window)
    all_rows=[]; threshold_rows=[]
    req=sorted(set(f for fs in variants.values() for f in fs))
    for h in cfg.horizons:
        d=x.copy(); d[f'future_return_{h}']=pd.to_numeric(d['spot_close'],errors='coerce').shift(-h)/pd.to_numeric(d['spot_close'],errors='coerce')-1
        roundtrip=2*(cfg.fee_bps+cfg.slippage_bps)/10000.0
        d[f'target_{h}']=(d[f'future_return_{h}']>roundtrip).astype(int)
        d=d[(d['regime']!='unknown') & d[f'future_return_{h}'].notna()].copy()
        for fam in req:
            cols=family_map.get(fam,[])
            if not cols: raise ValueError(f'Missing feature family: {fam}')
            d=d[d[cols].notna().any(axis=1)]
        d=d.reset_index(drop=True)
        gap=h+2
        for fold,tr,te in _outer_walk(len(d),cfg.n_splits,cfg.min_train_size,gap):
            fit_idx,cal_idx,thr_idx=_three_way_train_split(tr,gap)
            for vname,family_names in variants.items():
                cols=[]
                for fam in family_names: cols+=family_map[fam]
                cols=list(dict.fromkeys(cols))
                yfit=d.iloc[fit_idx][f'target_{h}'].astype(int)
                if yfit.nunique()<2: continue
                model=_pipe(cfg.random_state); model.fit(d.iloc[fit_idx][cols],yfit)
                raw_cal=model.predict_proba(d.iloc[cal_idx][cols])[:,1]; calibrator=_fit_platt(raw_cal,d.iloc[cal_idx][f'target_{h}'].to_numpy(int))
                raw_thr=model.predict_proba(d.iloc[thr_idx][cols])[:,1]; p_thr=_apply_platt(calibrator,raw_thr)
                threshold,tdiag=_choose_threshold(p_thr,d.iloc[thr_idx][f'future_return_{h}'].to_numpy(float),thr_idx,h,cfg)
                threshold_rows.append({'horizon_bars':h,'fold':fold,'variant':vname,'threshold':threshold,**tdiag})
                raw=model.predict_proba(d.iloc[te][cols])[:,1]; prob=_apply_platt(calibrator,raw)
                keep=(te % h)==0; te2=te[keep]; prob=prob[keep]
                if len(te2)==0: continue
                pos=(prob>=threshold).astype(float); fr=d.iloc[te2][f'future_return_{h}'].to_numpy(float); net=pos*fr-pos*roundtrip
                for j,idx in enumerate(te2):
                    row=d.iloc[idx]
                    all_rows.append({'timestamp':row['timestamp'],'fold':fold,'horizon_bars':h,'variant':vname,'prob_up':float(prob[j]),'threshold':float(threshold),'position':float(pos[j]),'future_return':float(fr[j]),'net_return':float(net[j]),'target_up':int(row[f'target_{h}']),'regime':row['regime'],'regime_transition':bool(row['regime_transition'])})
    pred=pd.DataFrame(all_rows); th=pd.DataFrame(threshold_rows)
    if pred.empty: raise RuntimeError('No v0.5 predictions produced')
    summary=[]; conditional=[]; transition=[]
    for (h,v),g in pred.groupby(['horizon_bars','variant']):
        summary.append({'horizon_bars':h,'variant':v,**_perf(g,h)})
        for reg,rg in g.groupby('regime'):
            if len(rg)>=cfg.min_regime_obs: conditional.append({'horizon_bars':h,'variant':v,'regime':reg,**_perf(rg,h)})
        for flag,tg in g.groupby('regime_transition'):
            if len(tg)>=30: transition.append({'horizon_bars':h,'variant':v,'transition':bool(flag),**_perf(tg,h)})
    summary=pd.DataFrame(summary); conditional=pd.DataFrame(conditional); transition=pd.DataFrame(transition)
    comparisons=[]; pvals={}
    for h in cfg.horizons:
        base=pred[(pred.horizon_bars==h)&(pred.variant=='baseline')].set_index(['timestamp','fold'])
        for v in [x for x in variants if x!='baseline']:
            vg=pred[(pred.horizon_bars==h)&(pred.variant==v)].set_index(['timestamp','fold'])
            for reg in ['ALL','bull','bear','sideways','high_vol','crisis']:
                b=base if reg=='ALL' else base[base.regime==reg]; q=vg if reg=='ALL' else vg[vg.regime==reg]
                a=b[['net_return','prob_up','position','future_return']].rename(columns={'net_return':'b','prob_up':'bp','position':'bpos','future_return':'fr'}).join(q[['net_return','prob_up','position']].rename(columns={'net_return':'v','prob_up':'vp','position':'vpos'}),how='inner')
                if len(a)<cfg.min_regime_obs: continue
                boot=moving_block_bootstrap(a.v-a.b,cfg.bootstrap_block,cfg.bootstrap_runs,cfg.random_state+h)
                key=f'h{h}:{v}:{reg}'; pvals[key]=boot['p_one_sided']
                nsel=int(a.vpos.sum()); matched=np.zeros(len(a),dtype=float)
                if nsel>0:
                    matched[np.argsort(-a.bp.to_numpy())[:min(nsel,len(a))]]=1.0
                roundtrip=2*(cfg.fee_bps+cfg.slippage_bps)/10000.0
                matched_net=matched*a.fr.to_numpy(float)-matched*roundtrip
                comparisons.append({'key':key,'horizon_bars':h,'variant':v,'regime':reg,'n':len(a),'delta_mean_net_return':float((a.v-a.b).mean()),'ci_low':boot['ci_low'],'ci_high':boot['ci_high'],'p_one_sided':boot['p_one_sided'],'candidate_coverage':float(a.vpos.mean()),'baseline_coverage':float(a.bpos.mean()),'coverage_matched_baseline_mean_net':float(np.mean(matched_net)),'coverage_matched_delta_mean_net':float(np.mean(a.v.to_numpy(float)-matched_net))})
    qvals=benjamini_hochberg(pvals)
    for r in comparisons:
        r['bh_adjusted_p']=qvals.get(r['key'],np.nan)
        r['status']='candidate' if np.isfinite(r['ci_low']) and r['ci_low']>0 and r['bh_adjusted_p']<=.10 else ('no_incremental_evidence' if np.isfinite(r['ci_high']) and r['ci_high']<=0 else 'inconclusive')
    comparisons=pd.DataFrame(comparisons)
    rc=[]
    for (h,v),g in pred.groupby(['horizon_bars','variant']):
        for t in np.arange(.50,.76,.05):
            pos=(g.prob_up.to_numpy()>=t); y=g.target_up.to_numpy(int); fr=g.future_return.to_numpy(float); cost=2*(cfg.fee_bps+cfg.slippage_bps)/10000.0; net=pos*fr-pos*cost
            rc.append({'horizon_bars':h,'variant':v,'diagnostic_threshold':float(t),'coverage':float(pos.mean()),'selective_accuracy':float((y[pos]==1).mean()) if pos.any() else np.nan,'mean_net_return':float(net.mean()),'sharpe':_ann_sharpe(net,h)})
    rc=pd.DataFrame(rc)
    robust={}; raw_srs=[]
    for (h,v),g in pred.groupby(['horizon_bars','variant']):
        r=g.net_return.to_numpy(float); sr=float(np.mean(r)/np.std(r,ddof=1)) if len(r)>2 and np.std(r,ddof=1)>0 else np.nan; raw_srs.append(sr)
    trial_std=float(np.nanstd(raw_srs,ddof=1)) if len(raw_srs)>1 else 0.0; num_trials=max(1,len(raw_srs))
    for h in cfg.horizons:
        hh=pred[pred.horizon_bars==h]; pivot=hh.pivot_table(index=['timestamp','fold'],columns='variant',values='net_return',aggfunc='first').dropna()
        robust[f'h{h}']={'pbo':probability_backtest_overfitting(pivot,6),'dsr':{}}
        for v in pivot.columns:
            robust[f'h{h}']['dsr'][v]=deflated_sharpe_probability(pivot[v],num_trials,trial_std)
    return summary,conditional,comparisons,transition,rc,th,pred,robust


def cpcv_sanity_panel(features:pd.DataFrame,family_map:dict,variants:dict,horizon:int=1,fee_bps:float=10,slippage_bps:float=2,n_groups:int=6,test_groups:int=2,seed:int=42):
    x=add_point_in_time_regimes(features.copy())
    x[f'future_return_{horizon}']=pd.to_numeric(x['spot_close'],errors='coerce').shift(-horizon)/pd.to_numeric(x['spot_close'],errors='coerce')-1
    cost=2*(fee_bps+slippage_bps)/10000.0; x[f'target_{horizon}']=(x[f'future_return_{horizon}']>cost).astype(int); x=x[(x.regime!='unknown')&x[f'future_return_{horizon}'].notna()].copy()
    req=sorted(set(f for fs in variants.values() for f in fs))
    for fam in req: x=x[x[family_map[fam]].notna().any(axis=1)]
    x=x.reset_index(drop=True); groups=np.array_split(np.arange(len(x)),n_groups); rows=[]; purge=horizon+2
    for split_id,te_g in enumerate(combinations(range(n_groups),test_groups),1):
        te=np.concatenate([groups[i] for i in te_g]); train_mask=np.ones(len(x),dtype=bool); train_mask[te]=False
        for idx in te:
            lo=max(0,idx-purge); hi=min(len(x),idx+purge+1); train_mask[lo:hi]=False
        tr=np.where(train_mask)[0]
        if len(tr)<500: continue
        for v,fns in variants.items():
            cols=[]
            for f in fns: cols+=family_map[f]
            cols=list(dict.fromkeys(cols)); y=x.iloc[tr][f'target_{horizon}'].astype(int)
            if y.nunique()<2: continue
            m=_pipe(seed); m.fit(x.iloc[tr][cols],y); p=m.predict_proba(x.iloc[te][cols])[:,1]; pos=(p>=.60).astype(float); fr=x.iloc[te][f'future_return_{horizon}'].to_numpy(float); net=pos*fr-pos*cost
            rows.append({'split':split_id,'variant':v,'n_test':len(te),'sharpe':_ann_sharpe(net,horizon),'mean_net_return':float(net.mean()),'coverage':float(pos.mean())})
    return pd.DataFrame(rows)
