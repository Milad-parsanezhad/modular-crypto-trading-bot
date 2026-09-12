from __future__ import annotations

import pandas as pd

from research_bot.time_cluster_bootstrap_v43 import timestamp_cluster_block_resample_v43


def _events() -> pd.DataFrame:
    rows=[]
    for i, ts in enumerate(pd.date_range('2026-01-01', periods=80, freq='4h', tz='UTC')):
        for venue in ('coinex','okx','kucoin'):
            rows.append({'signal_time':ts,'venue':venue,'x':i})
    return pd.DataFrame(rows)


def test_timestamp_cluster_bootstrap_is_deterministic_and_cluster_safe() -> None:
    x=_events()
    a, da=timestamp_cluster_block_resample_v43(x, seed=314, target_block_events=64)
    b, db=timestamp_cluster_block_resample_v43(x, seed=314, target_block_events=64)
    c, _=timestamp_cluster_block_resample_v43(x, seed=1618, target_block_events=64)
    assert a.equals(b)
    assert da == db
    assert da.circular_wrap_used is False
    assert not a.equals(c)
    assert len(a) >= len(x)
    original=x.groupby('signal_time').size().to_dict()
    sampled=a.groupby('signal_time').size().to_dict()
    for ts,count in sampled.items():
        assert count % original[ts] == 0


def test_timestamp_cluster_bootstrap_never_reads_outcomes() -> None:
    x=_events()
    x['outcome']='STOP'
    a,da=timestamp_cluster_block_resample_v43(x, seed=2718)
    x['outcome']='TARGET'
    b,db=timestamp_cluster_block_resample_v43(x, seed=2718)
    assert da.circular_wrap_used is False and db.circular_wrap_used is False
    assert a[['signal_time','venue','x']].equals(b[['signal_time','venue','x']])
