import numpy as np
import pandas as pd
from research_bot.features import add_features
from research_bot.regime import add_regime_features
from research_bot.model import fit_predict_holdout,positions_from_probabilities
from research_bot.backtest import backtest_positions

def synthetic_ohlcv(n=900,seed=7):
    rng=np.random.default_rng(seed); ret=rng.normal(0.0002,0.012,n); close=30000*np.exp(np.cumsum(ret)); open_=np.r_[close[0],close[:-1]]; span=np.maximum(10,close*rng.uniform(0.001,0.012,n)); high=np.maximum(open_,close)+span; low=np.minimum(open_,close)-span; vol=rng.lognormal(10,0.4,n); ts=pd.date_range("2024-01-01",periods=n,freq="4h",tz="UTC")
    return pd.DataFrame({"timestamp":ts,"open":open_,"high":high,"low":low,"close":close,"volume":vol})

def test_end_to_end_synthetic():
    df=add_regime_features(add_features(synthetic_ohlcv())); _,train,test,cols=fit_predict_holdout(df); assert len(train)>300; assert len(test)>100; assert len(cols)>=20
    pos=positions_from_probabilities(test["prob_up"]); _,metrics=backtest_positions(test["future_return"],pos); assert metrics["n"]==len(test); assert 0<=metrics["avg_abs_position"]<=1
