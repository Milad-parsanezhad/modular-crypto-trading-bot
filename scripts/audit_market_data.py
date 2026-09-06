"""Cross-check 4h candles against independent delivery and official daily gaps."""
import argparse,hashlib,json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd
from milad_trader.archive import BASE,_download,parse_archive
from milad_trader.data import read_csv,validate_candles


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True)
    p.add_argument('--crosscheck',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    raw=read_csv(a.data)
    daily=raw.resample('1D').agg(dict(open='first',high='max',low='min',close='last',volume='sum'))
    external=pd.read_csv(a.crosscheck,skiprows=1)
    external.index=pd.to_datetime(external.Date,utc=True)
    external=external.rename(columns={'Open':'open','High':'high','Low':'low','Close':'close','Volume BTC':'volume'})
    if external.index.has_duplicates:raise ValueError('Duplicate cross-check dates')
    common=daily.index.intersection(external.index)
    missing=daily.index.difference(external.index)
    errors=(daily.loc[common]-external.loc[common,daily.columns]).abs().max()
    cache=Path(a.data).parent/'archive-cache'
    def month(m):
        name=f'BTCUSDT-1d-{m}.zip';url=f'{BASE}/BTCUSDT/1d/{name}'
        fields=_download(url+'.CHECKSUM').decode('ascii').split()
        if len(fields)!=2 or fields[1].lstrip('*')!=name:raise ValueError('Wrong checksum filename')
        dest=cache/name
        payload=dest.read_bytes() if dest.exists() else _download(url)
        frame=validate_candles(parse_archive(payload,fields[0]),'1D')
        dest.write_bytes(payload)
        return frame,dict(url=url,sha256=fields[0],rows=len(frame))
    periods=sorted(set(missing.strftime('%Y-%m')))
    with ThreadPoolExecutor(max_workers=8) as executor:results=list(executor.map(month,periods))
    extra_errors={}
    if results:
        official=pd.concat([r[0] for r in results])
        extra_errors=(daily.loc[missing]-official.loc[missing,daily.columns]).abs().max().to_dict()
    def within(values):
        return all(values[k]<=.01 for k in ['open','high','low','close']) and values['volume']<=.00001
    meta=json.loads(Path(a.data).with_suffix('.manifest.json').read_text())
    result=dict(rows=len(raw),days=len(daily),first_open=str(raw.index[0]),last_open=str(raw.index[-1]),
        data_sha256=hashlib.sha256(Path(a.data).read_bytes()).hexdigest(),archive_count=len(meta['archives']),
        source='https://data.binance.vision/',
        crosscheck_source='https://www.cryptodatadownload.com/cdd/Binance_BTCUSDT_d.csv',
        crosscheck_file_sha256=hashlib.sha256(Path(a.crosscheck).read_bytes()).hexdigest(),
        crosscheck_note='Second delivery of the same exchange, not independent exchange validation.',
        crosscheck_days=len(common),maximum_absolute_differences=errors.to_dict(),
        missing_crosscheck_dates=[str(t) for t in missing],
        supplemental_check='Official Binance daily archives for dates absent from CryptoDataDownload',
        supplemental_archives=[r[1] for r in results],supplemental_maximum_absolute_differences=extra_errors,
        passed=bool(within(errors.to_dict()) and (not extra_errors or within(extra_errors))))
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    Path(a.output).write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ['supplemental_archives','missing_crosscheck_dates']},indent=2))
    if not result['passed']:raise ValueError('Cross-check mismatch; inspect report before training')


if __name__=='__main__':main()
