from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timedelta,timezone
from hashlib import sha256
from io import BytesIO
import time
from urllib.error import HTTPError,URLError
from urllib.request import Request,urlopen
from zipfile import ZipFile
import numpy as np
import pandas as pd
BASE_URL='https://data.binance.vision/data'

def _download(url,timeout=30,attempts=5,missing_ok=False):
    last=None
    for i in range(attempts):
        try:
            with urlopen(Request(url,headers={'User-Agent':'modular-crypto-research-bot/0.4'}),timeout=timeout) as r: return r.read()
        except HTTPError as e:
            if e.code==404 and missing_ok:return None
            if e.code not in {408,425,429,500,502,503,504}:raise
            last=e
        except (URLError,TimeoutError,ConnectionError,OSError) as e:last=e
        if i<attempts-1:time.sleep(min(8,.75*(2**i)))
    raise RuntimeError(f'Download failed after {attempts} attempts: {url}: {last}') from last

def _verify(url,payload,enabled):
    if not enabled:return None
    c=_download(url+'.CHECKSUM',missing_ok=True)
    if c is None:return None
    expected=c.decode(errors='ignore').strip().split()[0].lower(); actual=sha256(payload).hexdigest().lower()
    if expected!=actual:raise RuntimeError(f'Checksum mismatch for {url}')
    return True

def _read_zip(payload):
    with ZipFile(BytesIO(payload)) as z:
        names=[n for n in z.namelist() if n.lower().endswith('.csv')]
        if not names:raise RuntimeError('ZIP contains no CSV')
        raw=z.read(names[0])
    first=raw.splitlines()[0].decode(errors='ignore').lower(); has_header=any(k in first for k in ('open_time','create_time','symbol','interest'))
    return pd.read_csv(BytesIO(raw),header=0 if has_header else None)

def _utc(s):
    n=pd.to_numeric(s,errors='coerce')
    if n.notna().mean()>.95:
        med=float(n.dropna().abs().median()); unit='us' if med>1e14 else 'ms' if med>1e11 else 's'; return pd.to_datetime(n,unit=unit,utc=True,errors='coerce')
    return pd.to_datetime(s,utc=True,errors='coerce')

def fetch_um_monthly_klines(symbol='BTCUSDT',interval='4h',start_month='2023-09',end_month=None,verify_checksum=True):
    symbol=symbol.upper(); now=pd.Timestamp.now(tz='UTC'); end_month=end_month or (now.to_period('M')-1).strftime('%Y-%m'); frames=[]; files=[]
    for per in pd.period_range(pd.Period(start_month,'M'),pd.Period(end_month,'M'),freq='M'):
        ym=per.strftime('%Y-%m'); url=f'{BASE_URL}/futures/um/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{ym}.zip'; payload=_download(url,missing_ok=True)
        if payload is None:files.append({'file':ym,'status':'missing'});continue
        verified=_verify(url,payload,verify_checksum); raw=_read_zip(payload)
        if raw.shape[1]<11:raise RuntimeError(f'Unexpected kline schema {ym}: {raw.shape[1]}')
        cols=['open_time','open','high','low','close','volume','close_time','quote_volume','trade_count','taker_buy_base','taker_buy_quote','ignore']; raw=raw.iloc[:,:min(len(cols),raw.shape[1])].copy(); raw.columns=cols[:raw.shape[1]]; raw['timestamp']=_utc(raw.open_time)
        for c in ('quote_volume','taker_buy_quote'):raw[c]=pd.to_numeric(raw[c],errors='coerce')
        sell=(raw.quote_volume-raw.taker_buy_quote).clip(lower=0); raw['binance_taker_buy_quote']=raw.taker_buy_quote; raw['binance_quote_volume']=raw.quote_volume; raw['binance_of_quote_imbalance']=(raw.taker_buy_quote-sell)/raw.quote_volume.replace(0,np.nan); raw['binance_taker_buy_share']=raw.taker_buy_quote/raw.quote_volume.replace(0,np.nan)
        frames.append(raw[['timestamp','binance_taker_buy_quote','binance_quote_volume','binance_of_quote_imbalance','binance_taker_buy_share']]); files.append({'file':ym,'status':'ok','checksum_verified':verified,'rows':len(raw)})
    if not frames:return pd.DataFrame(),{'files':files}
    df=pd.concat(frames,ignore_index=True).dropna(subset=['timestamp']).drop_duplicates('timestamp').sort_values('timestamp').reset_index(drop=True); return df,{'files':files,'rows':len(df),'coverage_start':df.timestamp.min().isoformat(),'coverage_end':df.timestamp.max().isoformat(),'source':'Binance Vision USD-M monthly klines'}

def _metrics(payload):
    raw=_read_zip(payload); raw.columns=[str(c).strip().lower().replace(' ','_') for c in raw.columns]
    if 'create_time' not in raw:
        cols=['create_time','symbol','sum_open_interest','sum_open_interest_value','count_toptrader_long_short_ratio','sum_toptrader_long_short_ratio','count_long_short_ratio','sum_taker_long_short_vol_ratio']; raw.columns=cols[:raw.shape[1]]
    raw['timestamp']=_utc(raw.create_time)
    for c in ('sum_open_interest','sum_open_interest_value'):
        if c in raw:raw[c]=pd.to_numeric(raw[c],errors='coerce')
    return raw[[c for c in ('timestamp','sum_open_interest','sum_open_interest_value') if c in raw]].dropna(subset=['timestamp'])

def fetch_um_daily_metrics(symbol='BTCUSDT',start_date=None,end_date=None,max_workers=12,verify_checksum=False):
    symbol=symbol.upper(); end=pd.Timestamp(end_date or (datetime.now(timezone.utc).date()-timedelta(days=1))).date(); start=pd.Timestamp(start_date or (end-timedelta(days=365))).date(); days=pd.date_range(start,end,freq='D').date
    def one(d):
        ds=d.isoformat(); url=f'{BASE_URL}/futures/um/daily/metrics/{symbol}/{symbol}-metrics-{ds}.zip'; payload=_download(url,missing_ok=True)
        return (ds,None,None) if payload is None else (ds,_metrics(payload),_verify(url,payload,verify_checksum))
    frames=[]; files=[]
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        fs=[ex.submit(one,d) for d in days]
        for f in as_completed(fs):
            ds,fr,v=f.result(); files.append({'file':ds,'status':'missing' if fr is None else 'ok','checksum_verified':v,'rows':0 if fr is None else len(fr)}); frames+=[] if fr is None else [fr]
    if not frames:return pd.DataFrame(),{'files':sorted(files,key=lambda x:x['file'])}
    df=pd.concat(frames,ignore_index=True).dropna(subset=['timestamp']).sort_values('timestamp').drop_duplicates('timestamp',keep='last'); df['timestamp']=df.timestamp+pd.Timedelta(days=1); df=df.rename(columns={'sum_open_interest':'binance_open_interest','sum_open_interest_value':'binance_open_interest_value'}).reset_index(drop=True)
    return df,{'files':sorted(files,key=lambda x:x['file']),'rows':len(df),'coverage_start':df.timestamp.min().isoformat(),'coverage_end':df.timestamp.max().isoformat(),'source':'Binance Vision USD-M daily metrics','availability_shift':'+1 day conservative'}

def merge_archives_point_in_time(base,orderflow=None,oi=None):
    x=base.copy(); x['timestamp']=pd.to_datetime(x.timestamp,utc=True); x=x.sort_values('timestamp')
    if orderflow is not None and not orderflow.empty:
        o=orderflow.copy(); o['timestamp']=pd.to_datetime(o.timestamp,utc=True); x=x.merge(o,on='timestamp',how='left')
    if oi is not None and not oi.empty:
        o=oi.copy(); o['timestamp']=pd.to_datetime(o.timestamp,utc=True); x=pd.merge_asof(x.sort_values('timestamp'),o.sort_values('timestamp'),on='timestamp',direction='backward')
    return x.reset_index(drop=True)
