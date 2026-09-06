import numpy as np,pandas as pd
from research_bot.alpha_ablation import AblationConfig,build_point_in_time_features,run_ablation_panel

def synthetic(n=2600,seed=123):
    r=np.random.default_rng(seed); fund=r.normal(0,.00015,n); basis=np.zeros(n); ret=np.zeros(n)
    for i in range(1,n): ret[i]=.0001+.15*fund[i-1]-.05*basis[i-1]+r.normal(0,.006); basis[i]=.8*basis[i-1]+r.normal(0,.00025)
    spot=30000*np.exp(np.cumsum(ret)); fut=spot*(1+basis); ts=pd.date_range('2023-01-01',periods=n,freq='4h',tz='UTC'); return pd.DataFrame({'timestamp':ts,'spot_close':spot,'futures_close':fut,'realized_basis_rate':basis,'actual_funding_rate':fund,'theoretical_funding_rate':fund+r.normal(0,1e-5,n)})

def test_ablation_runs():
    f,m=build_point_in_time_features(synthetic()); cfg=AblationConfig(n_splits=3,min_train_size=1500,purge_bars=2,probability_threshold=.55); v={'baseline':['baseline'],'funding':['baseline','funding'],'basis':['baseline','basis'],'funding_basis':['baseline','funding','basis']}; s,p,st=run_ablation_panel(f,m,v,cfg); assert set(s.variant)==set(v) and len(p)>100 and st['common_rows']>2000
