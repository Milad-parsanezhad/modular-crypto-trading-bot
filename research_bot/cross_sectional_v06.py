from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .binance_vision import BASE_URL, _download, _read_zip, _utc, _verify

FUNDING_COLUMNS = ['calc_time','funding_interval_hours','last_funding_rate']
KLINE_COLUMNS = ['open_time','open','high','low','close','volume','close_time','quote_volume','trade_count','taker_buy_base','taker_buy_quote','ignore']

@dataclass(frozen=True)
class CrossSectionConfig:
    start_month: str = '2022-01'
    end_month: str | None = None
    interval: str = '8h'
    n_splits: int = 3
    top_quantile: float = 0.20
    one_way_cost_bps: float = 6.0
    random_state: int = 42


def _canonical_utc_timestamp(values):
    """Normalize all event clocks to one merge-safe dtype.

    pandas 3 can preserve source resolution (us/ms/ns) even when every series is
    timezone-aware. merge_asof requires the dtype units to be identical, so the
    research clock is canonicalized to UTC nanoseconds before any alignment.
    """
    s = pd.to_datetime(values, utc=True, errors='coerce')
    if isinstance(s, pd.Series):
        return s.astype('datetime64[ns, UTC]')
    return pd.Series(s, dtype='datetime64[ns, UTC]')


def filter_regime_comparison(comp: pd.DataFrame, horizon_bars: int, regime: str, variant: str) -> pd.DataFrame:
    """Validate and filter regime-engine comparison output.

    This is intentionally in the package (not scripts/) so tests and callers use
    the same contract and CI does not depend on script-package import behavior.
    """
    required={'horizon_bars','regime','variant'}
    missing=required-set(comp.columns)
    if missing:
        raise RuntimeError(f'Regime comparison schema mismatch: missing={sorted(missing)} columns={list(comp.columns)}')
    return comp[(comp['horizon_bars']==horizon_bars)&(comp['regime']==regime)&(comp['variant']==variant)].copy()


def _month_range(start_month:str,end_month:str|None=None):
    now=pd.Timestamp.now(tz='UTC')
    end_month=end_month or (now.tz_localize(None).to_period('M')-1).strftime('%Y-%m')
    return pd.period_range(pd.Period(start_month,'M'),pd.Period(end_month,'M'),freq='M')


def fetch_um_monthly_raw_klines(symbol:str,interval:str='8h',start_month:str='2022-01',end_month:str|None=None,verify_checksum:bool=True,kind:str='klines'):
    symbol=symbol.upper(); frames=[]; files=[]
    if kind not in {'klines','premiumIndexKlines'}: raise ValueError(kind)
    for per in _month_range(start_month,end_month):
        ym=per.strftime('%Y-%m'); url=f'{BASE_URL}/futures/um/monthly/{kind}/{symbol}/{interval}/{symbol}-{interval}-{ym}.zip'
        payload=_download(url,missing_ok=True)
        if payload is None:
            files.append({'file':ym,'status':'missing'}); continue
        verified=_verify(url,payload,verify_checksum); raw=_read_zip(payload)
        if raw.shape[1] < 5: raise RuntimeError(f'Unexpected {kind} schema {symbol} {ym}: {raw.shape}')
        raw=raw.iloc[:,:min(len(KLINE_COLUMNS),raw.shape[1])].copy(); raw.columns=KLINE_COLUMNS[:raw.shape[1]]
        raw['timestamp_open']=_canonical_utc_timestamp(_utc(raw['open_time']))
        raw['timestamp']=_canonical_utc_timestamp(_utc(raw['close_time']))
        raw['timestamp']=raw['timestamp']+pd.Timedelta(milliseconds=1)
        for c in ('open','high','low','close','volume','quote_volume','taker_buy_quote'):
            if c in raw: raw[c]=pd.to_numeric(raw[c],errors='coerce')
        keep=['timestamp','open','high','low','close','volume']+[c for c in ('quote_volume','taker_buy_quote') if c in raw]
        frames.append(raw[keep]); files.append({'file':ym,'status':'ok','checksum_verified':verified,'rows':len(raw)})
    if not frames:return pd.DataFrame(),{'files':files,'symbol':symbol,'kind':kind}
    df=pd.concat(frames,ignore_index=True).dropna(subset=['timestamp']).drop_duplicates('timestamp').sort_values('timestamp').reset_index(drop=True)
    df['timestamp']=_canonical_utc_timestamp(df['timestamp'])
    return df,{'files':files,'symbol':symbol,'kind':kind,'rows':len(df),'coverage_start':df.timestamp.min().isoformat(),'coverage_end':df.timestamp.max().isoformat()}


def fetch_um_monthly_funding(symbol:str,start_month:str='2022-01',end_month:str|None=None,verify_checksum:bool=True):
    symbol=symbol.upper(); frames=[]; files=[]
    for per in _month_range(start_month,end_month):
        ym=per.strftime('%Y-%m'); url=f'{BASE_URL}/futures/um/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{ym}.zip'
        payload=_download(url,missing_ok=True)
        if payload is None:
            files.append({'file':ym,'status':'missing'}); continue
        verified=_verify(url,payload,verify_checksum); raw=_read_zip(payload)
        raw.columns=[str(c).strip().lower() for c in raw.columns]
        if 'calc_time' not in raw.columns:
            raw=raw.iloc[:,:min(3,raw.shape[1])].copy(); raw.columns=FUNDING_COLUMNS[:raw.shape[1]]
        rate_col='last_funding_rate' if 'last_funding_rate' in raw.columns else raw.columns[-1]
        raw['timestamp']=_canonical_utc_timestamp(_utc(raw['calc_time'])); raw['funding_rate']=pd.to_numeric(raw[rate_col],errors='coerce')
        if 'funding_interval_hours' in raw.columns: raw['funding_interval_hours']=pd.to_numeric(raw['funding_interval_hours'],errors='coerce')
        else: raw['funding_interval_hours']=8.0
        frames.append(raw[['timestamp','funding_rate','funding_interval_hours']]); files.append({'file':ym,'status':'ok','checksum_verified':verified,'rows':len(raw)})
    if not frames:return pd.DataFrame(),{'files':files,'symbol':symbol}
    df=pd.concat(frames,ignore_index=True).dropna(subset=['timestamp']).drop_duplicates('timestamp').sort_values('timestamp').reset_index(drop=True)
    df['timestamp']=_canonical_utc_timestamp(df['timestamp'])
    return df,{'files':files,'symbol':symbol,'rows':len(df),'coverage_start':df.timestamp.min().isoformat(),'coverage_end':df.timestamp.max().isoformat()}


def build_symbol_panel(symbol:str,start_month:str,end_month:str|None=None,verify_checksum:bool=True):
    px,pm=fetch_um_monthly_raw_klines(symbol,'8h',start_month,end_month,verify_checksum,'klines')
    prem,prm=fetch_um_monthly_raw_klines(symbol,'8h',start_month,end_month,verify_checksum,'premiumIndexKlines')
    fund,fm=fetch_um_monthly_funding(symbol,start_month,end_month,verify_checksum)
    if px.empty:return pd.DataFrame(),{'price':pm,'premium':prm,'funding':fm}
    x=px.rename(columns={'close':'futures_close','volume':'futures_volume','quote_volume':'futures_quote_volume','taker_buy_quote':'futures_taker_buy_quote'})
    x['symbol']=symbol.upper(); x['timestamp']=_canonical_utc_timestamp(x['timestamp']); x=x.sort_values('timestamp')
    if not prem.empty:
        pp=prem[['timestamp','close']].rename(columns={'close':'premium_index_close'}).copy(); pp['timestamp']=_canonical_utc_timestamp(pp['timestamp']); pp=pp.sort_values('timestamp')
        x=pd.merge_asof(x,pp,on='timestamp',direction='backward',tolerance=pd.Timedelta('8h'))
    if not fund.empty:
        fund=fund.copy(); fund['timestamp']=_canonical_utc_timestamp(fund['timestamp']); fund=fund.sort_values('timestamp')
        x=pd.merge_asof(x,fund,on='timestamp',direction='backward',tolerance=pd.Timedelta('8h'))
    return x,{'price':pm,'premium':prm,'funding':fm}


def add_cross_sectional_features(panel:pd.DataFrame):
    x=panel.copy(); x['timestamp']=_canonical_utc_timestamp(x.timestamp); x=x.sort_values(['symbol','timestamp']).reset_index(drop=True)
    g=x.groupby('symbol',group_keys=False)
    x['ret_1']=g['futures_close'].pct_change(); x['mom_3']=g['futures_close'].pct_change(3); x['mom_9']=g['futures_close'].pct_change(9)
    x['vol_21']=g['ret_1'].rolling(21,min_periods=10).std().reset_index(level=0,drop=True)
    x['liq_log_quote']=np.log1p(pd.to_numeric(x.get('futures_quote_volume'),errors='coerce'))
    total=pd.to_numeric(x.get('futures_quote_volume'),errors='coerce'); buy=pd.to_numeric(x.get('futures_taker_buy_quote'),errors='coerce')
    x['orderflow_imbalance']=(2*buy-total)/total.replace(0,np.nan)
    x['funding_rate']=pd.to_numeric(x.get('funding_rate'),errors='coerce')
    x['premium_index_close']=pd.to_numeric(x.get('premium_index_close'),errors='coerce')
    for col,prefix in [('funding_rate','fund'),('premium_index_close','prem')]:
        mu=g[col].rolling(21,min_periods=10).mean().reset_index(level=0,drop=True); sd=g[col].rolling(21,min_periods=10).std().reset_index(level=0,drop=True)
        x[f'{prefix}_z21']=(x[col]-mu)/sd.replace(0,np.nan)
        x[f'{prefix}_mean21']=mu
    x['fund_cum21']=g['funding_rate'].rolling(21,min_periods=10).sum().reset_index(level=0,drop=True)
    rank_cols=['funding_rate','fund_z21','premium_index_close','prem_z21','mom_3','mom_9','vol_21','liq_log_quote','orderflow_imbalance']
    for c in rank_cols: x[f'cs_{c}_pct']=x.groupby('timestamp')[c].rank(pct=True,method='average')
    x['future_ret_8h']=g['futures_close'].shift(-1)/x['futures_close']-1
    x['future_ret_24h']=g['futures_close'].shift(-3)/x['futures_close']-1
    return x.replace([np.inf,-np.inf],np.nan)

BASELINE_FEATURES=['mom_3','mom_9','vol_21','liq_log_quote','orderflow_imbalance','cs_mom_3_pct','cs_mom_9_pct','cs_vol_21_pct','cs_liq_log_quote_pct','cs_orderflow_imbalance_pct']
FUNDING_PREMIUM_FEATURES=['funding_rate','fund_z21','fund_mean21','fund_cum21','premium_index_close','prem_z21','prem_mean21','cs_funding_rate_pct','cs_fund_z21_pct','cs_premium_index_close_pct','cs_prem_z21_pct']


def _time_folds(timestamps:Iterable[pd.Timestamp],n_splits:int=3):
    ts=np.array(sorted(pd.unique(pd.to_datetime(pd.Series(timestamps),utc=True))))
    start=max(120,int(len(ts)*0.50)); remaining=len(ts)-start
    step=max(30,remaining//n_splits)
    for k in range(n_splits):
        s=start+k*step; e=len(ts) if k==n_splits-1 else min(len(ts),s+step)
        if e-s<20: continue
        yield ts[:s],ts[s:e]


def _model(kind:str,seed:int):
    if kind=='ridge_logit':
        return Pipeline([('imputer',SimpleImputer(strategy='median')),('scale',StandardScaler()),('model',LogisticRegression(C=0.25,solver='lbfgs',max_iter=1500,random_state=seed))])
    if kind=='hgb':
        return Pipeline([('imputer',SimpleImputer(strategy='median')),('model',HistGradientBoostingClassifier(learning_rate=.04,max_iter=220,max_leaf_nodes=15,l2_regularization=1.0,random_state=seed))])
    raise ValueError(kind)


def _portfolio_from_scores(test:pd.DataFrame,score_col:str,target_col:str,q:float,cost_bps:float,rebalance_every:int=1):
    if rebalance_every < 1: raise ValueError('rebalance_every must be >=1')
    rows=[]; prev={}; grouped=list(test.groupby('timestamp',sort=True))
    for i,(ts,g) in enumerate(grouped):
        if i % rebalance_every != 0: continue
        z=g.dropna(subset=[score_col,target_col]).copy(); n=len(z)
        if n<6: continue
        k=max(1,int(np.floor(n*q))); z=z.sort_values(score_col)
        short=z.head(k); long=z.tail(k); weights={}
        for _,r in long.iterrows(): weights[r.symbol]=0.5/k
        for _,r in short.iterrows(): weights[r.symbol]=-0.5/k
        gross=sum(weights.get(r.symbol,0.0)*float(r[target_col]) for _,r in z.iterrows())
        syms=set(prev)|set(weights); turnover=sum(abs(weights.get(s,0)-prev.get(s,0)) for s in syms)
        cost=turnover*cost_bps/10000.0; net=gross-cost
        rows.append({'timestamp':ts,'gross_return':gross,'net_return':net,'turnover':turnover,'n_assets':n,'n_long':k,'n_short':k}); prev=weights
    return pd.DataFrame(rows)


def _perf(r:pd.Series,periods_per_year:float):
    r=pd.Series(r).dropna().astype(float)
    if r.empty:return {'n':0}
    eq=(1+r).cumprod(); dd=eq/eq.cummax()-1; sd=r.std(ddof=1)
    return {'n':int(len(r)),'total_return':float(eq.iloc[-1]-1),'sharpe':float(r.mean()/sd*sqrt(periods_per_year)) if sd>0 else np.nan,'max_drawdown':float(dd.min()),'mean_return':float(r.mean()),'volatility':float(sd) if np.isfinite(sd) else np.nan}


def run_cross_sectional_experiment(panel:pd.DataFrame,cfg:CrossSectionConfig,horizon:int=1):
    if horizon not in (1,3): raise ValueError('v0.6 supports horizon=1 (8h) or horizon=3 (24h)')
    x=add_cross_sectional_features(panel); target='future_ret_8h' if horizon==1 else 'future_ret_24h'; results=[]; predictions=[]
    variants={'baseline':BASELINE_FEATURES,'funding_premium':BASELINE_FEATURES+FUNDING_PREMIUM_FEATURES}
    for fold,(train_ts,test_ts) in enumerate(_time_folds(x.timestamp,cfg.n_splits),1):
        tr=x[x.timestamp.isin(train_ts)].dropna(subset=[target]).copy(); te=x[x.timestamp.isin(test_ts)].dropna(subset=[target]).copy()
        y=(tr[target]>0).astype(int)
        if y.nunique()<2: continue
        for model_kind in ('ridge_logit','hgb'):
            for variant,cols in variants.items():
                cols=[c for c in cols if c in x.columns]
                m=_model(model_kind,cfg.random_state); m.fit(tr[cols],y); p=m.predict_proba(te[cols])[:,1]
                tmp=te[['timestamp','symbol',target]].copy(); tmp['score']=p; tmp['fold']=fold; tmp['variant']=variant; tmp['model']=model_kind; predictions.append(tmp)
    pred=pd.concat(predictions,ignore_index=True) if predictions else pd.DataFrame()
    if pred.empty:return pd.DataFrame(),pred
    ppy=(3*365)/horizon
    for (model,variant),g in pred.groupby(['model','variant']):
        bt=_portfolio_from_scores(g,'score',target,cfg.top_quantile,cfg.one_way_cost_bps,rebalance_every=horizon)
        d=_perf(bt.net_return,ppy); d.update({'model':model,'variant':variant,'horizon_hours':8*horizon,'rebalance_every_bars':horizon,'mean_turnover':float(bt.turnover.mean()) if not bt.empty else np.nan,'timestamps':int(bt.timestamp.nunique()) if not bt.empty else 0})
        ics=[]
        for _,z in g.groupby('timestamp'):
            if z.score.nunique()>1 and z[target].nunique()>1: ics.append(z.score.corr(z[target],method='spearman'))
        d['mean_spearman_ic']=float(np.nanmean(ics)) if ics else np.nan; results.append(d)
    return pd.DataFrame(results),pred
