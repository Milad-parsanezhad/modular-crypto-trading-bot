from __future__ import annotations

from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pandas as pd


COLUMNS = (
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_base", "taker_quote", "ignore",
)


def _parse_timestamp(values: pd.Series) -> pd.Series:
    raw = pd.to_numeric(values, errors="raise")
    unit = pd.Series("ms", index=raw.index)
    unit.loc[raw.abs() >= 10**14] = "us"
    out = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns, UTC]")
    for name in ("ms", "us"):
        mask = unit.eq(name)
        out.loc[mask] = pd.to_datetime(raw.loc[mask], unit=name, utc=True, errors="raise")
    return out


def load_monthly_spot_archives(
    cache_dir: str | Path,
    symbol: str,
    *,
    timeframe: str = "4h",
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Load official Binance Vision monthly ZIPs already present on disk."""

    cache = Path(cache_dir)
    paths = sorted(cache.glob(f"{symbol.upper()}-{timeframe}-????-??.zip"))
    if not paths:
        raise FileNotFoundError(f"no Binance Vision archives for {symbol} {timeframe} in {cache}")
    frames = []
    for path in paths:
        with ZipFile(path) as archive:
            names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if not names:
                raise ValueError(f"archive has no CSV: {path}")
            raw = pd.read_csv(BytesIO(archive.read(names[0])), header=None)
        if raw.shape[1] < 6:
            raise ValueError(f"unexpected Binance kline schema: {path}")
        raw = raw.iloc[:, : min(len(COLUMNS), raw.shape[1])]
        raw.columns = COLUMNS[: raw.shape[1]]
        raw["timestamp"] = _parse_timestamp(raw["open_time"])
        for col in ("open", "high", "low", "close", "volume"):
            raw[col] = pd.to_numeric(raw[col], errors="raise")
        frames.append(raw[["timestamp", "open", "high", "low", "close", "volume"]])
    out = pd.concat(frames, ignore_index=True).sort_values("timestamp")
    out = out.drop_duplicates("timestamp", keep="last")
    if start:
        out = out[out["timestamp"] >= pd.Timestamp(start, tz="UTC")]
    if end:
        out = out[out["timestamp"] <= pd.Timestamp(end, tz="UTC")]
    return out.reset_index(drop=True)
