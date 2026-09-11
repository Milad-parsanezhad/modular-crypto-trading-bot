from __future__ import annotations

import pandas as pd

from research_bot.execution_stress_v40 import (
    SCENARIOS_V40,
    apply_execution_scenario_v40,
    decision_v40,
    preregistration_manifest_v40,
)


def _frame():
    return pd.DataFrame([{
        'executed_v38':True,'entry':100.0,'stop':95.0,'r_multiple':1.0,'risk_fraction_v38':0.002,
        'exit_time':pd.Timestamp('2025-01-02',tz='UTC'),'symbol':'BTC/USDT'
    }])


def test_manifest_keeps_live_disabled():
    m=preregistration_manifest_v40()
    assert len(m['scenarios'])==3
    assert m['input_cost_already_embedded_bps']==24.0
    assert m['no_alpha_retraining'] is True
    assert m['kraken_touched'] is False
    assert m['live_execution_authorized'] is False


def test_stress_penalty_is_monotonic():
    base=apply_execution_scenario_v40(_frame(),SCENARIOS_V40[0])['stressed_r_v40'].iloc[0]
    s36=apply_execution_scenario_v40(_frame(),SCENARIOS_V40[1])['stressed_r_v40'].iloc[0]
    s60=apply_execution_scenario_v40(_frame(),SCENARIOS_V40[2])['stressed_r_v40'].iloc[0]
    assert base > s36 > s60


def test_failed_parent_cannot_progress():
    d=decision_v40({'execution_stress_eligible_v40':False})
    assert d['winner'] is None
    assert d['kraken_touched'] is False
    assert d['live_execution_authorized'] is False
