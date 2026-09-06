import numpy as np
import pandas as pd
from research_bot.cross_sectional_v06 import add_cross_sectional_features,run_cross_sectional_experiment,CrossSectionConfig

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
