from __future__ import annotations

import numpy as np
import pandas as pd

from research_bot.soft_portfolio_v38 import (
    POLICY_V38,
    TOTAL_EFFECTIVE_TRIALS_V38,
    _waterfill,
    decision_v38,
    portfolio_metrics_v38,
    preregistration_manifest_v38,
    screen_three_venues_v38,
    soft_allocate_v38,
)


def _events() -> pd.DataFrame:
    rows=[]
    base=pd.Timestamp('2025-01-01', tz='UTC')
    for i in range(12):
        for j,s in enumerate(['BTC/USDT','ETH/USDT','SOL/USDT']):
            t=base+pd.Timedelta(days=i*3)
            rows.append({
                'symbol':s,'signal_time':t,'entry_time':t+pd.Timedelta(days=1),
                'exit_time':t+pd.Timedelta(days=2+j),'side':1 if j<2 else -1,
                'base_strategy_v36':'V30_D1_CUSUM_BREAKOUT_3','utility_v36':0.2+j*0.1,
                'selected_v36':True,'r_multiple':1.0 if (i+j)%3 else -1.0,
            })
    return pd.DataFrame(rows)


def test_manifest_is_fail_closed_and_counts_trial():
    m=preregistration_manifest_v38()
    assert m['total_effective_trials']==TOTAL_EFFECTIVE_TRIALS_V38==133
    assert m['reserved_holdout_venue']=='kraken'
    assert m['kraken_touched'] is False
    assert m['live_execution_authorized'] is False
    assert m['min_trades_per_venue']==200
    assert m['min_block_ci_low']==0.0


def test_waterfill_respects_budget_and_cap():
    a=_waterfill(np.array([10.,2.,1.]),0.006,0.0025)
    assert a.sum() <= 0.006 + 1e-12
    assert a.max() <= 0.0025 + 1e-12
    assert (a>=0).all()


def test_soft_allocator_respects_event_and_portfolio_risk():
    out,diag=soft_allocate_v38(_events())
    assert diag['input_selected']==36
    assert (out['risk_fraction_v38']>=0).all()
    assert out['risk_fraction_v38'].max() <= POLICY_V38.per_event_risk_cap + 1e-12
    assert not out.loc[~out['executed_v38'],'account_return_v38'].ne(0).any()


def test_metrics_and_screen_fail_closed():
    out,_=soft_allocate_v38(_events())
    m=portfolio_metrics_v38(out)
    assert m['trades'] > 0
    bad={v:{'trades':199,'profit_factor':2,'expectancy_r':1,'positive_asset_fraction':1,'block_ci_low':1e-4,'max_drawdown':-0.01} for v in ('coinex_consumed','okx_consumed','kucoin_consumed')}
    s=screen_three_venues_v38(bad)
    assert s['development_eligible_v38'] is False
    d=decision_v38(s)
    assert d['winner'] is None
    assert d['kraken_touched'] is False
    assert d['live_execution_authorized'] is False
