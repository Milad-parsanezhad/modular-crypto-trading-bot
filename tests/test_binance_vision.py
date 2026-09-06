from io import BytesIO
from zipfile import ZipFile
import pandas as pd
from research_bot.binance_vision import _metrics,merge_archives_point_in_time

def z(text):
    b=BytesIO();
    with ZipFile(b,'w') as f:f.writestr('x.csv',text)
    return b.getvalue()

def test_metrics_and_merge():
    d=_metrics(z('create_time,symbol,sum_open_interest,sum_open_interest_value\n2026-01-01 00:00:00,BTCUSDT,100,200000\n')); assert len(d)==1 and float(d.sum_open_interest.iloc[0])==100
    base=pd.DataFrame({'timestamp':pd.date_range('2026-01-02',periods=3,freq='4h',tz='UTC'),'spot_close':[1,2,3]}); oi=pd.DataFrame({'timestamp':[pd.Timestamp('2026-01-02',tz='UTC')],'binance_open_interest':[100.]}); m=merge_archives_point_in_time(base,oi=oi); assert m.binance_open_interest.notna().all()
