from __future__ import annotations

import pandas as pd

from research_bot.event_family_priority_v43 import assign_event_family_priority_v43


def _row(**overrides):
    base={
        'mother_event_v39':1,
        'ict_recent_sweep_down_v39':0,'ict_recent_sweep_up_v39':0,
        'canonical_bull_displacement':0,'canonical_bear_displacement':0,
        'brooks_failed_breakdown_v39':0,'brooks_failed_breakout_v39':0,
        'smc_bull_ob_retest_v39':0,'smc_bear_ob_retest_v39':0,
        'ichimoku_pullback_long_v39':0,'ichimoku_pullback_short_v39':0,
        'ichimoku_breakout_up_v39':0,'ichimoku_breakout_down_v39':0,
        'brooks_h2_v39':0,'brooks_l2_v39':0,
        'ict_mss_up_v39':0,'ict_mss_down_v39':0,
    }
    base.update(overrides)
    return base


def test_first_match_priority_is_not_overwritten_by_later_family() -> None:
    x=pd.DataFrame([
        _row(ict_recent_sweep_down_v39=1, ict_mss_up_v39=1),
        _row(canonical_bull_displacement=1, brooks_failed_breakout_v39=1, ict_mss_down_v39=1),
        _row(brooks_h2_v39=1, ict_mss_up_v39=1),
    ])
    fam=assign_event_family_priority_v43(x).tolist()
    assert fam == ['LIQUIDITY_SWEEP','SMC_FVG_STRUCTURE','BROOKS_H2L2']


def test_non_mother_event_is_other_even_if_flags_are_set() -> None:
    x=pd.DataFrame([_row(mother_event_v39=0, ict_recent_sweep_down_v39=1, ict_mss_up_v39=1)])
    assert assign_event_family_priority_v43(x).iloc[0] == 'OTHER_MOTHER_EVENT'
