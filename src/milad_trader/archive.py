"""Verified Binance spot monthly archives; no account or trading API required."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import io
import re
import time
import urllib.request
import zipfile
import pandas as pd
from .data import OHLCV, validate_candles

BASE = "https://data.binance.vision/data/spot/monthly/klines"


def parse_archive(payload, expected_sha256):
    if not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256):
        raise ValueError("Invalid archive checksum")
    if hashlib.sha256(payload).hexdigest() != expected_sha256.lower():
        raise ValueError("Archive checksum mismatch")
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        members = archive.infolist()
        if len(members) != 1 or not members[0].filename.endswith('.csv') or members[0].file_size > 10_000_000:
            raise ValueError("Expected one bounded CSV archive member")
        with archive.open(members[0]) as stream:
            table = pd.read_csv(stream, header=None)
    if table.shape[1] != 12 or table.empty:
        raise ValueError("Expected Binance's 12-column kline schema")
    stamp = pd.to_numeric(table.iloc[:, 0], errors='raise').astype('int64')
    microseconds = stamp >= 100_000_000_000_000
    if microseconds.any() and not microseconds.all():
        raise ValueError("Mixed timestamp units within archive")
    index = pd.to_datetime(stamp, unit='us' if microseconds.all() else 'ms', utc=True)
    result = table.iloc[:, 1:6].copy()
    result.columns = OHLCV
    result.index = pd.DatetimeIndex(index, name='timestamp')
    return result


def _download(url):
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                payload = response.read(10_000_001)
            if len(payload) > 10_000_000:
                raise ValueError("Archive response exceeds size limit")
            return payload
        except (OSError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def fetch_archive(symbol, timeframe, since, until, cache_dir='data/raw/archive-cache', workers=4):
    """Download complete UTC calendar months [since, until); fail on gaps.

    Checksums are fetched from the official HTTPS origin on every invocation.
    Cached archives are reused only after matching those checksums.
    """
    if symbol not in {'BTC/USDT', 'ETH/USDT'} or timeframe not in {'1h', '4h'}:
        raise ValueError("Archive supports BTC/USDT or ETH/USDT at 1h or 4h")
    start, end = pd.Timestamp(since), pd.Timestamp(until)
    if start.tzinfo is None or end.tzinfo is None or start >= end:
        raise ValueError("Require timezone-aware since < until")
    start, end = start.tz_convert('UTC'), end.tz_convert('UTC')
    if any(t.day != 1 or t != t.normalize() for t in (start, end)):
        raise ValueError("Monthly archive bounds must be UTC month starts")
    if end > pd.Timestamp.now(tz='UTC'):
        raise ValueError("Cannot request unfinished months")
    if not 1 <= workers <= 8:
        raise ValueError("workers must be between 1 and 8")
    months = pd.date_range(start, end, freq='MS', inclusive='left')
    if len(months) > 120:
        raise ValueError("Request at most 120 months")
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    pair = symbol.replace('/', '')

    def one(month):
        name = f'{pair}-{timeframe}-{month:%Y-%m}.zip'
        url = f'{BASE}/{pair}/{timeframe}/{name}'
        checksum_text = _download(url + '.CHECKSUM').decode('ascii').strip()
        fields = checksum_text.split()
        if len(fields) != 2 or fields[1].lstrip('*') != name:
            raise ValueError("Checksum manifest names a different archive")
        path = cache / name
        payload = path.read_bytes() if path.exists() else _download(url)
        frame = parse_archive(payload, fields[0])
        frame = validate_candles(frame, timeframe)
        expected_end = month + pd.offsets.MonthBegin(1)
        if frame.index[0] != month or frame.index[-1] + pd.Timedelta(timeframe) != expected_end:
            raise ValueError(f'Incomplete month: {name}')
        if not path.exists():
            temporary = path.with_suffix('.tmp')
            temporary.write_bytes(payload)
            temporary.replace(path)
        return frame, {'url': url, 'checksum_url': url + '.CHECKSUM',
                       'sha256': fields[0].lower(), 'rows': len(frame)}

    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(one, months))
    candles = validate_candles(pd.concat([r[0] for r in results]), timeframe)
    metadata = dict(source='binance_public_archive', kind='historical', market='spot',
                    symbol=symbol, timeframe=timeframe, requested_since=start.isoformat(),
                    requested_until_exclusive=end.isoformat(), archives=[r[1] for r in results],
                    retrieved_at=datetime.now(timezone.utc).isoformat())
    return candles, metadata
