import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
from milad_trader.config import ExperimentConfig
from milad_trader.data import synthetic_candles
from milad_trader.experiment import run_experiment
from milad_trader.paper import paper_step


def test_end_to_end_experiment_and_paper_checkpoint(tmp_path):
    raw=synthetic_candles(500)
    cfg=ExperimentConfig(lookback=8,train_bars=150,validation_bars=40,test_bars=40,
                         folds=1,seeds=[42],models=["random_forest"],ablation=True)
    out=tmp_path/"experiment"
    metrics=run_experiment(raw,cfg,out,"synthetic_software_test")
    assert {"ichi","no_ichi"} <= set(metrics.features)
    assert json.loads((out/"manifest.json").read_text())["status"]=="completed"
    artifact=out/"fold0_seed42_random_forest_ichi.joblib"
    state=tmp_path/"account.json"
    first=raw.iloc[:400]
    now=first.index[-1]+pd.Timedelta("4h")
    assert paper_step(artifact,first,state,now)["status"]=="paper_checkpoint_saved"
    assert paper_step(artifact,first,state,now)["status"]=="no_new_closed_candle"
    after=raw.iloc[:401]
    paper_step(artifact,after,state,after.index[-1]+pd.Timedelta("4h"))
    snap=json.loads(state.read_text())
    assert len(snap["events"])==2
    with pytest.raises(ValueError,match="Missed"):
        paper_step(artifact,raw.iloc[:405],state,raw.index[404]+pd.Timedelta("4h"))
    revised=after.copy();revised.iloc[0,revised.columns.get_loc("volume")]*=2
    with pytest.raises(ValueError,match="Historical"):
        paper_step(artifact,revised,state,after.index[-1]+pd.Timedelta("4h"))
    with pytest.raises(ValueError,match="stale"):
        paper_step(artifact,after,state,after.index[-1]+pd.Timedelta("12h"))


@pytest.mark.parametrize("kind",["lstm","cnn","transformer"])
def test_sequence_models_train_save_and_predict(tmp_path,kind):
    pytest.importorskip("torch")
    from milad_trader.models import Predictor
    rng=np.random.default_rng(42)
    x=rng.normal(size=(100,4))
    y=(x[:,0]>0).astype(float)
    p=Predictor(kind,lookback=8,epochs=2).fit(x,y,np.arange(7,60),np.arange(65,80))
    expected=p.predict(x,np.arange(85,95))
    path=tmp_path/"model.joblib";joblib.dump(p,path)
    actual=joblib.load(path).predict(x,np.arange(85,95))
    np.testing.assert_allclose(actual,expected)
    assert ((actual>=0)&(actual<=1)).all()


def test_gymnasium_contract_and_ppo_smoke():
    pytest.importorskip("stable_baselines3")
    from milad_trader.features import make_features,feature_columns
    from milad_trader.rl import TradingEnv,fit_ppo,evaluate_ppo
    from stable_baselines3.common.env_checker import check_env
    f=make_features(synthetic_candles(250))
    x=f[feature_columns(f)].to_numpy()
    cfg=ExperimentConfig(lookback=8,train_bars=100,ppo_steps=128)
    from sklearn.preprocessing import StandardScaler
    scaled=StandardScaler().fit_transform(x[:100])
    env=TradingEnv(f.iloc[:100],scaled,cfg.risk,8)
    check_env(env,warn=True)
    agent,scaler=fit_ppo(f,x,100,cfg,42)
    curve,fills=evaluate_ppo(f,x,110,140,agent,scaler,cfg)
    assert len(curve)==30
    assert np.isfinite(curve.equity).all()
