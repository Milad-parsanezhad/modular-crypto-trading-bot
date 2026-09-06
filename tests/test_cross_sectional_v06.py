import numpy as np
import pandas as pd
import pytest

from research_bot.cross_sectional_v06 import (
    _canonical_utc_timestamp,
    _portfolio_from_scores,
    filter_regime_comparison,
    add_cross_sectional_features,
    run_cross_sectional_experiment,
    CrossSectionConfig,
)


def synthetic_panel(n_times=800,n_symbols=8,seed=11):
    rng=np.random.default_rng(seed); ts=pd.date_range('2022-01-01',periods=n_times,freq='8h',tz='UTC'); rows=[]
    for j in range(n_symbols):
        r=rng.normal(0.0001,0.012,n_times); close=(100+j*10)*np.exp(np.cumsum(r)); funding=0.0001+0.0002*rng.normal(size=n_times); premium=funding*10+rng.normal(0,0.0005,n_times); quote=rng.lognormal(12,0.5,n_times); buy=quote*(0.5+np.clip(rng.normal(0,0.05,n_times),-0.2,0.2))
        for i,t in enumerate(ts): rows.append({'timestamp':t,'symbol':f'S{j}USDT','futures_close':close[i],'futures_volume':quote[i]/close[i],'futures_quote_volume':quote[i],'futures_taker_buy_quote':buy[i],'funding_rate':funding[i],'funding_interval_hours':8,'premium_index_close':premium[i]})
    return pd.DataFrame(rows)


def test_feature_panel_is_cross_sectional_and_point_in_time_shape():
    x=add_cross_sectional_features(synthetic_panel(100,8)); assert 'cs_funding_rate_pct' in x; assert x.groupby('timestamp').size().min()==8; assert x['future_ret_8h'].notna().sum()>500


def test_cross_sectional_experiment_runs():
    panel=synthetic_panel(); cfg=CrossSectionConfig(n_splits=2,top_quantile=.25,one_way_cost_bps=6); summary,pred=run_cross_sectional_experiment(panel,cfg,1); assert not summary.empty; assert set(summary.variant)=={'baseline','funding_premium'}; assert set(summary.model)=={'ridge_logit','hgb'}; assert pred.timestamp.nunique()>100


def test_timestamp_canonicalization_prevents_merge_asof_unit_failure():
    base=pd.date_range('2024-01-01',periods=4,freq='8h',tz='UTC')
    left=pd.DataFrame({'timestamp':pd.Series(base.astype('datetime64[us, UTC]')),'x':[1,2,3,4]})
    right=pd.DataFrame({'timestamp':pd.Series(base.astype('datetime64[ms, UTC]')),'y':[10,20,30,40]})
    assert left.timestamp.dtype != right.timestamp.dtype
    left['timestamp']=_canonical_utc_timestamp(left.timestamp)
    right['timestamp']=_canonical_utc_timestamp(right.timestamp)
    assert left.timestamp.dtype == right.timestamp.dtype
    merged=pd.merge_asof(left.sort_values('timestamp'),right.sort_values('timestamp'),on='timestamp',direction='backward')
    assert merged.y.tolist()==[10,20,30,40]


def test_24h_portfolio_evaluation_is_non_overlapping_on_8h_clock():
    ts=pd.date_range('2024-01-01',periods=9,freq='8h',tz='UTC'); rows=[]
    for t in ts:
        for j in range(8):
            rows.append({'timestamp':t,'symbol':f'S{j}','score':float(j),'future_ret_24h':0.001*(j-3.5)})
    frame=pd.DataFrame(rows)
    every_bar=_portfolio_from_scores(frame,'score','future_ret_24h',.25,6,rebalance_every=1)
    non_overlap=_portfolio_from_scores(frame,'score','future_ret_24h',.25,6,rebalance_every=3)
    assert len(every_bar)==9
    assert len(non_overlap)==3
    assert non_overlap.timestamp.tolist()==ts[::3].tolist()


def test_regime_comparison_contract_uses_horizon_bars():
    comp=pd.DataFrame([
        {'horizon_bars':1,'regime':'bear','variant':'open_interest','incremental_mean_return':0.001},
        {'horizon_bars':1,'regime':'bull','variant':'open_interest','incremental_mean_return':-0.001},
        {'horizon_bars':3,'regime':'bear','variant':'open_interest','incremental_mean_return':0.002},
    ])
    out=filter_regime_comparison(comp,1,'bear','open_interest')
    assert len(out)==1
    assert out.iloc[0]['horizon_bars']==1
    assert out.iloc[0]['regime']=='bear'


def test_regime_comparison_contract_fails_fast_on_schema_drift():
    comp=pd.DataFrame([{'horizon':1,'regime':'bear','variant':'open_interest'}])
    with pytest.raises(RuntimeError,match='schema mismatch'):
        filter_regime_comparison(comp,1,'bear','open_interest')
