from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from milad_trader.config import RiskConfig,ExperimentConfig
from milad_trader.data import synthetic_candles,validate_candles
from milad_trader.features import make_features,feature_columns,labels
from milad_trader.engine import Broker,backtest
from milad_trader.splits import walk_forward
from milad_trader.metrics import performance,paired_block_interval
from milad_trader.models import Predictor


@pytest.fixture
def raw():
    return synthetic_candles(500)


def bar(opening=100, high=101, low=99, close=100):
    return dict(open=opening,high=high,low=low,close=close)


def test_features_do_not_change_when_future_changes(raw):
    before=make_features(raw)
    changed=raw.copy()
    changed.iloc[350:,changed.columns.get_indexer(["open","high","low","close"])] *= 3
    after=make_features(changed)
    pd.testing.assert_frame_equal(before.loc[:raw.index[349]],after.loc[:raw.index[349]])
    pd.testing.assert_frame_equal(before.loc[:raw.index[349]],make_features(raw.iloc[:350]))


def test_ichimoku_cloud_is_displaced_from_known_past(raw):
    f=make_features(raw)
    t=200
    historical=raw.iloc[:t-26+1]
    midpoint=lambda n:(historical.high.iloc[-n:].max()+historical.low.iloc[-n:].min())/2
    a=(midpoint(9)+midpoint(26))/2
    b=midpoint(52)
    assert f.loc[raw.index[t],"cloud_top"] == max(a,b)
    assert not any("ichi" in c for c in feature_columns(f,False))


def test_labels_use_executable_open_to_open_return(raw):
    f=make_features(raw)
    y=labels(f,horizon=2,cost=0)
    assert y.iloc[0] == float(f.open.iloc[3]>f.open.iloc[1])
    assert y.iloc[-3:].isna().all()


@pytest.mark.parametrize("problem",["duplicate","gap","unsorted","negative","range","nan","unfinished"])
def test_bad_market_data_rejected(raw,problem):
    x=raw.copy()
    if problem=="duplicate":x.index=pd.DatetimeIndex([*x.index[:-1],x.index[-2]])
    if problem=="gap":x=x.drop(x.index[10])
    if problem=="unsorted":x=x.iloc[::-1]
    if problem=="negative":x.iloc[10,x.columns.get_loc("volume")]=-1
    if problem=="range":x.iloc[10,x.columns.get_loc("high")]=1
    if problem=="nan":x.iloc[10,0]=np.nan
    kwargs={"now":raw.index[-1]} if problem=="unfinished" else {}
    with pytest.raises(ValueError):validate_candles(x,**kwargs)


def test_forward_label_information_cannot_cross_partition():
    c=ExperimentConfig(train_bars=100,validation_bars=40,test_bars=30,folds=2,horizon=4)
    folds=list(walk_forward(220,c))
    for f in folds:
        assert f.train[1]-1+c.horizon+1 < f.validation[0]
        assert f.validation[1]-1+c.horizon+1 < f.test[0]
    assert folds[0].test[1] == folds[1].test[0]


def test_initial_purchase_is_not_a_cash_drawdown():
    broker=Broker(RiskConfig(fee=0,slippage_bps=0,risk_per_trade=1))
    result=broker.process(pd.Timestamp("2024-01-01",tz="UTC"),bar(),1,10)
    assert broker.cash < 1000
    assert result["equity"] == pytest.approx(10000)
    assert not broker.halted


def test_fee_accounting_on_both_sides():
    broker=Broker(RiskConfig(slippage_bps=0,risk_per_trade=1))
    broker.process(pd.Timestamp("2024-01-01",tz="UTC"),bar(),1,10)
    qty=broker.quantity
    assert broker.cash == pytest.approx(10000-qty*100*1.001)
    broker.process(pd.Timestamp("2024-01-01T04:00Z"),bar(),0,10)
    assert broker.cash == pytest.approx(10000-qty*100*.002)
    assert broker.fills[-1]["pnl"] == pytest.approx(-qty*100*.002)


def test_both_stop_and_take_touch_uses_stop_first():
    broker=Broker(RiskConfig(fee=0,slippage_bps=0))
    broker.process(pd.Timestamp("2024-01-01",tz="UTC"),bar(100,105,97,100),1,1)
    assert broker.quantity==0
    assert broker.fills[-1]["reason"]=="stop_loss"
    assert broker.fills[-1]["price"]==98


def test_gap_stop_uses_worse_open_price():
    broker=Broker(RiskConfig(fee=0,slippage_bps=0))
    broker.process(pd.Timestamp("2024-01-01",tz="UTC"),bar(),1,1)
    broker.process(pd.Timestamp("2024-01-01T04:00Z"),bar(90,92,88,90),1,1)
    assert broker.fills[-1]["price"]==90
    assert broker.fills[-1]["reason"]=="gap_stop"
    assert broker.quantity==0


def test_duplicate_processing_is_rejected_without_extra_fill():
    b=Broker(RiskConfig())
    ts=pd.Timestamp("2024-01-01",tz="UTC")
    b.process(ts,bar(),1,10)
    count=len(b.fills)
    with pytest.raises(ValueError):b.process(ts,bar(),1,10)
    assert len(b.fills)==count


def test_risk_stop_latches_and_cannot_be_overruled():
    risk=RiskConfig(fee=0,slippage_bps=0,risk_per_trade=1,max_drawdown=.02,daily_loss=.5)
    b=Broker(risk)
    b.process(pd.Timestamp("2024-01-01",tz="UTC"),bar(),1,10)
    b.process(pd.Timestamp("2024-01-01T04:00Z"),bar(96,97,95,96),1,10)
    assert b.halted
    assert b.quantity==0
    b.process(pd.Timestamp("2024-01-02",tz="UTC"),bar(),1,10)
    assert b.quantity==0


def test_buy_decision_fills_next_bar_open():
    f=make_features(synthetic_candles(100))
    actions=np.zeros(len(f),int);actions[0]=1
    _,fills=backtest(f,actions,RiskConfig(),0,3)
    first=fills.iloc[0]
    assert pd.Timestamp(first.timestamp)==f.index[1]
    assert first.price==pytest.approx(f.open.iloc[1]*1.0005)


def test_scaler_is_fit_on_training_history_only(raw):
    f=make_features(raw)
    x=f[feature_columns(f)].to_numpy()
    y=labels(f).to_numpy()
    model=Predictor("random_forest",lookback=5)
    model.fit(x,y,np.arange(4,200),np.arange(205,250))
    np.testing.assert_allclose(model.scaler.mean_,x[:200].mean(axis=0))
    assert np.isfinite(model.predict(x,np.arange(255,260))).all()


def test_bootstrap_is_paired_and_reproducible():
    a=np.arange(100)*.0001
    r=paired_block_interval(a,a,seed=42)
    assert r["lower"]==r["upper"]==0
    assert r==paired_block_interval(a,a,seed=42)


def test_cash_baseline_has_zero_fees_and_undefined_sharpe(raw):
    f=make_features(raw)
    c,t=backtest(f,np.zeros(len(f),int),RiskConfig())
    m=performance(c,t,2190)
    assert m["fees"]==0 and m["total_return"]==0
    assert m["sharpe"] is None


def test_restart_preserves_accounting():
    b=Broker(RiskConfig())
    b.process(pd.Timestamp("2024-01-01",tz="UTC"),bar(),1,10)
    restored=Broker.restore(b.snapshot())
    ts=pd.Timestamp("2024-01-01T04:00Z")
    assert b.process(ts,bar(),0,10)==restored.process(ts,bar(),0,10)
