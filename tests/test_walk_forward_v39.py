from __future__ import annotations

import pandas as pd

from research_bot.walk_forward_v39 import (
    N_FOLDS_V39,
    REQUIRED_POSITIVE_FOLDS_V39,
    decision_v39,
    preregistration_manifest_v39,
    walk_forward_v39,
)


def _frame():
    rows=[]
    base=pd.Timestamp('2025-01-01',tz='UTC')
    for i in range(250):
        rows.append({'executed_v38':True,'exit_time':base+pd.Timedelta(days=i),'symbol':['BTC','ETH','SOL','ADA'][i%4],
                     'account_return_v38':0.0005 if i%5 else -0.0002,'risk_fraction_v38':0.001})
    return pd.DataFrame(rows)


def test_manifest_is_sealed_and_nonadaptive():
    m=preregistration_manifest_v39()
    assert m['folds']==N_FOLDS_V39==5
    assert m['required_positive_folds']==REQUIRED_POSITIVE_FOLDS_V39==4
    assert m['new_hyperparameter_trials']==0
    assert m['kraken_touched'] is False
    assert m['live_execution_authorized'] is False


def test_walk_forward_returns_five_folds():
    r=walk_forward_v39(_frame())
    assert len(r['folds'])==5
    assert r['positive_folds']>=4
    assert r['stable'] is True


def test_parent_failure_blocks_progress():
    venues={v:{'stable':True} for v in ('coinex_consumed','okx_consumed','kucoin_consumed')}
    d=decision_v39(False,venues)
    assert d['winner'] is None
    assert d['kraken_touched'] is False
    assert d['live_execution_authorized'] is False
