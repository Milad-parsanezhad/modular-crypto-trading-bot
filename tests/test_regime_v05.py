import numpy as np
import pandas as pd
from research_bot.regime_robustness import add_point_in_time_regimes, moving_block_bootstrap, benjamini_hochberg, RegimeRobustnessConfig

def synthetic(n=1800,seed=11):
    rng=np.random.default_rng(seed); r=rng.normal(0.0001,0.012,n); close=30000*np.exp(np.cumsum(r))
    ts=pd.date_range('2023-01-01',periods=n,freq='4h',tz='UTC')
    return pd.DataFrame({'timestamp':ts,'spot_close':close})

def test_regime_no_future_leakage():
    a=synthetic(); b=a.copy(); b.loc[1200:,'spot_close']*=np.linspace(1,2,len(b)-1200)
    ra=add_point_in_time_regimes(a); rb=add_point_in_time_regimes(b)
    assert (ra.loc[:1100,'regime'].to_numpy()==rb.loc[:1100,'regime'].to_numpy()).all()

def test_regime_labels_and_config():
    r=add_point_in_time_regimes(synthetic())
    assert set(r['regime'].unique()) <= {'unknown','bull','bear','sideways','high_vol','crisis'}
    assert RegimeRobustnessConfig().horizons==(1,3,6)

def test_bootstrap_and_bh():
    x=np.r_[np.ones(150)*0.001,np.ones(150)*0.002]
    b=moving_block_bootstrap(x,block=12,runs=100,seed=1)
    assert b['mean']>0 and b['ci_low']>0
    q=benjamini_hochberg({'a':.01,'b':.04,'c':.5})
    assert 0<=q['a']<=q['b']<=q['c']<=1
