import hashlib
import io
import zipfile
import pandas as pd
import pytest
from milad_trader.archive import parse_archive, fetch_archive


def zipped(text):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as z:
        z.writestr('candles.csv', text)
    payload = buffer.getvalue()
    return payload, hashlib.sha256(payload).hexdigest()


@pytest.mark.parametrize('stamp,expected', [(1704067200000, '2024-01-01'), (1735689600000000, '2025-01-01')])
def test_timestamp_units(stamp, expected):
    frame = parse_archive(*zipped(f'{stamp},100,110,90,105,20,0,0,0,0,0,0\n'))
    assert frame.index[0] == pd.Timestamp(expected, tz='UTC')
    assert frame.iloc[0].to_dict() == dict(open=100, high=110, low=90, close=105, volume=20)


def test_checksum_rejects_tampering():
    payload, checksum = zipped('1704067200000,100,110,90,105,20,0,0,0,0,0,0\n')
    with pytest.raises(ValueError, match='checksum mismatch'):
        parse_archive(payload+b'changed', checksum)


def test_mixed_units_rejected():
    with pytest.raises(ValueError, match='Mixed timestamp'):
        parse_archive(*zipped('1704067200000,100,110,90,105,20,0,0,0,0,0,0\n1735689600000000,100,110,90,105,20,0,0,0,0,0,0\n'))


def test_complete_month_and_cache(tmp_path, monkeypatch):
    import milad_trader.archive as module
    timestamps = pd.date_range('2025-01-01', '2025-02-01', freq='4h', inclusive='left', tz='UTC')
    rows = ''.join(f'{t.value//1000},100,110,90,105,20,0,0,0,0,0,0\n' for t in timestamps)
    payload, digest = zipped(rows)
    calls = []
    def download(url):
        calls.append(url)
        return f'{digest}  BTCUSDT-4h-2025-01.zip'.encode() if url.endswith('.CHECKSUM') else payload
    monkeypatch.setattr(module, '_download', download)
    args = ('BTC/USDT','4h','2025-01-01T00:00:00Z','2025-02-01T00:00:00Z',tmp_path)
    frame, meta = fetch_archive(*args)
    assert len(frame) == 186 and meta['archives'][0]['sha256'] == digest
    fetch_archive(*args)
    assert len(calls) == 3  # second invocation rechecks origin digest, reuses payload
    (tmp_path/'BTCUSDT-4h-2025-01.zip').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='checksum mismatch'):
        fetch_archive(*args)


def test_partial_month_rejected(tmp_path, monkeypatch):
    import milad_trader.archive as module
    payload, digest = zipped('1735689600000000,100,110,90,105,20,0,0,0,0,0,0\n1735704000000000,100,110,90,105,20,0,0,0,0,0,0\n')
    monkeypatch.setattr(module, '_download', lambda url: f'{digest}  BTCUSDT-4h-2025-01.zip'.encode() if url.endswith('.CHECKSUM') else payload)
    with pytest.raises(ValueError, match='Incomplete month'):
        fetch_archive('BTC/USDT','4h','2025-01-01T00:00:00Z','2025-02-01T00:00:00Z',tmp_path)
