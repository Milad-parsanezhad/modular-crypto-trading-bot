from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Iterable

import numpy as np
import pandas as pd

from .binance_vision import BASE_URL, _download, _read_zip, _utc, _verify, fetch_um_daily_metrics


KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trade_count",
    "taker_buy_base", "taker_buy_quote", "ignore",
]


@dataclass(frozen=True)
class MicrostructureForwardConfig:
    interval: str = "1h"
    start_date: str = "2026-09-01"
    end_date: str | None = None
    min_rows_per_symbol: int = 48
    basis_z_window: int = 24
    flow_window: int = 12
    oi_z_window: int = 24
    verify_checksum: bool = True


def _date_range(start_date: str, end_date: str | None) -> list[date]:
    start = pd.Timestamp(start_date).date()
    if end_date is None:
        end = (datetime.now(timezone.utc).date() - timedelta(days=1))
    else:
        end = pd.Timestamp(end_date).date()
    if end < start:
        return []
    return [x.date() for x in pd.date_range(start, end, freq="D")]


def _parse_kline(payload: bytes, *, market: str) -> pd.DataFrame:
    raw = _read_zip(payload)
    if raw.shape[1] < 11:
        raise RuntimeError(f"Unexpected {market} kline schema: {raw.shape}")
    raw = raw.iloc[:, : min(len(KLINE_COLUMNS), raw.shape[1])].copy()
    raw.columns = KLINE_COLUMNS[: raw.shape[1]]
    raw["timestamp_open"] = _utc(raw["open_time"])
    raw["timestamp"] = _utc(raw["close_time"]) + pd.Timedelta(microseconds=1)
    for col in ("open", "high", "low", "close", "volume", "quote_volume", "taker_buy_quote"):
        if col in raw:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw["market_kind"] = market
    return raw.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


def fetch_daily_klines(
    symbol: str,
    *,
    market: str,
    interval: str,
    start_date: str,
    end_date: str | None,
    verify_checksum: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Download point-in-time daily Binance Vision klines.

    This collector is intentionally limited to public archives and never uses
    private credentials. The decision timestamp is the completed candle close,
    not the candle open.
    """
    symbol = symbol.upper()
    if market not in {"spot", "um"}:
        raise ValueError("market must be 'spot' or 'um'")
    frames: list[pd.DataFrame] = []
    files: list[dict] = []
    for day in _date_range(start_date, end_date):
        ds = day.isoformat()
        if market == "spot":
            url = f"{BASE_URL}/spot/daily/klines/{symbol}/{interval}/{symbol}-{interval}-{ds}.zip"
        else:
            url = f"{BASE_URL}/futures/um/daily/klines/{symbol}/{interval}/{symbol}-{interval}-{ds}.zip"
        payload = _download(url, missing_ok=True)
        if payload is None:
            files.append({"date": ds, "status": "missing"})
            continue
        verified = _verify(url, payload, verify_checksum)
        frame = _parse_kline(payload, market=market)
        frames.append(frame)
        files.append({"date": ds, "status": "ok", "rows": int(len(frame)), "checksum_verified": verified})
    if not frames:
        return pd.DataFrame(), {"symbol": symbol, "market": market, "files": files, "rows": 0}
    out = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates("timestamp")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    return out, {
        "symbol": symbol,
        "market": market,
        "files": files,
        "rows": int(len(out)),
        "coverage_start": out["timestamp"].min().isoformat(),
        "coverage_end": out["timestamp"].max().isoformat(),
        "source": "Binance Vision public daily kline archive",
    }


def _flow_features(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    x = frame.copy()
    total = pd.to_numeric(x["quote_volume"], errors="coerce")
    buy = pd.to_numeric(x["taker_buy_quote"], errors="coerce")
    x[f"{prefix}_taker_buy_share"] = buy / total.replace(0, np.nan)
    x[f"{prefix}_flow_imbalance"] = (2.0 * buy - total) / total.replace(0, np.nan)
    return x


def build_forward_microstructure_panel(
    symbol: str,
    cfg: MicrostructureForwardConfig | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Build a forward-only microstructure evidence panel.

    v0.13 does not promote a model. It creates a new post-v0.12 observation
    window with true spot-perpetual basis, finer 1h taker-flow proxies and
    conservatively lagged OI. Liquidation history remains DATA_UNAVAILABLE
    rather than being fabricated.
    """
    cfg = cfg or MicrostructureForwardConfig()
    spot, spot_meta = fetch_daily_klines(
        symbol, market="spot", interval=cfg.interval,
        start_date=cfg.start_date, end_date=cfg.end_date,
        verify_checksum=cfg.verify_checksum,
    )
    fut, fut_meta = fetch_daily_klines(
        symbol, market="um", interval=cfg.interval,
        start_date=cfg.start_date, end_date=cfg.end_date,
        verify_checksum=cfg.verify_checksum,
    )
    if spot.empty or fut.empty:
        return pd.DataFrame(), {
            "symbol": symbol,
            "status": "DATA_UNAVAILABLE",
            "spot": spot_meta,
            "futures": fut_meta,
            "liquidations": "DATA_UNAVAILABLE",
        }

    spot = _flow_features(spot, "spot")
    fut = _flow_features(fut, "futures")
    s = spot[
        ["timestamp", "close", "quote_volume", "taker_buy_quote",
         "spot_taker_buy_share", "spot_flow_imbalance"]
    ].rename(columns={
        "close": "spot_close",
        "quote_volume": "spot_quote_volume",
        "taker_buy_quote": "spot_taker_buy_quote",
    })
    f = fut[
        ["timestamp", "close", "quote_volume", "taker_buy_quote",
         "futures_taker_buy_share", "futures_flow_imbalance"]
    ].rename(columns={
        "close": "futures_close",
        "quote_volume": "futures_quote_volume",
        "taker_buy_quote": "futures_taker_buy_quote",
    })
    x = s.merge(f, on="timestamp", how="inner").sort_values("timestamp").reset_index(drop=True)
    x["symbol"] = symbol.upper()
    x["true_spot_perp_basis"] = x["futures_close"] / x["spot_close"] - 1.0
    x["basis_change_1"] = x["true_spot_perp_basis"].diff()
    bmu = x["true_spot_perp_basis"].rolling(cfg.basis_z_window).mean()
    bsd = x["true_spot_perp_basis"].rolling(cfg.basis_z_window).std()
    x["basis_z"] = (x["true_spot_perp_basis"] - bmu) / bsd.replace(0, np.nan)
    x["flow_spread"] = x["futures_flow_imbalance"] - x["spot_flow_imbalance"]
    x["futures_flow_mean"] = x["futures_flow_imbalance"].rolling(cfg.flow_window).mean()
    x["spot_flow_mean"] = x["spot_flow_imbalance"].rolling(cfg.flow_window).mean()
    x["flow_spread_mean"] = x["flow_spread"].rolling(cfg.flow_window).mean()

    start = pd.Timestamp(cfg.start_date, tz="UTC").date()
    end = pd.Timestamp(cfg.end_date, tz="UTC").date() if cfg.end_date else datetime.now(timezone.utc).date() - timedelta(days=1)
    oi, oi_meta = fetch_um_daily_metrics(symbol.upper(), start, end, max_workers=8, verify_checksum=False)
    if not oi.empty:
        oi = oi.copy()
        oi["timestamp"] = pd.to_datetime(oi["timestamp"], utc=True)
        x = pd.merge_asof(
            x.sort_values("timestamp"),
            oi.sort_values("timestamp"),
            on="timestamp",
            direction="backward",
        )
        raw = pd.to_numeric(x.get("binance_open_interest"), errors="coerce")
        val = pd.to_numeric(x.get("binance_open_interest_value"), errors="coerce")
        x["oi_log"] = np.log(raw.where(raw > 0))
        x["oi_delta_1"] = x["oi_log"].diff()
        x["oi_value_log"] = np.log(val.where(val > 0))
        x["oi_value_delta_1"] = x["oi_value_log"].diff()
        mu, sd = x["oi_log"].rolling(cfg.oi_z_window).mean(), x["oi_log"].rolling(cfg.oi_z_window).std()
        x["oi_z"] = (x["oi_log"] - mu) / sd.replace(0, np.nan)
    else:
        oi_meta = {**(oi_meta or {}), "status": "DATA_UNAVAILABLE"}

    x["future_spot_return_1h"] = x["spot_close"].shift(-1) / x["spot_close"] - 1.0
    x = x.replace([np.inf, -np.inf], np.nan)

    feature_cols = [
        "true_spot_perp_basis", "basis_change_1", "basis_z",
        "spot_flow_imbalance", "futures_flow_imbalance", "flow_spread",
        "spot_flow_mean", "futures_flow_mean", "flow_spread_mean",
        "oi_delta_1", "oi_value_delta_1", "oi_z",
    ]
    available = {c: int(x[c].notna().sum()) if c in x else 0 for c in feature_cols}
    status = "FORWARD_WINDOW_READY" if len(x) >= cfg.min_rows_per_symbol else "FORWARD_WINDOW_ACCUMULATING"
    meta = {
        "symbol": symbol.upper(),
        "status": status,
        "config": asdict(cfg),
        "rows": int(len(x)),
        "coverage_start": x["timestamp"].min().isoformat() if len(x) else None,
        "coverage_end": x["timestamp"].max().isoformat() if len(x) else None,
        "spot": spot_meta,
        "futures": fut_meta,
        "open_interest": oi_meta,
        "feature_available_rows": available,
        "liquidations": "DATA_UNAVAILABLE",
        "order_flow_scope": "1h kline taker-flow proxy; not L2/L3 order-book reconstruction",
        "decision_contract": "DATA_COLLECTION_ONLY_NOT_TRADING_SIGNAL",
    }
    return x, meta


def diagnostic_correlations(panel: pd.DataFrame) -> dict[str, float | None]:
    if panel.empty or "future_spot_return_1h" not in panel:
        return {}
    cols = [
        "true_spot_perp_basis", "basis_change_1", "basis_z",
        "spot_flow_imbalance", "futures_flow_imbalance", "flow_spread",
        "oi_delta_1", "oi_value_delta_1", "oi_z",
    ]
    out: dict[str, float | None] = {}
    target = panel["future_spot_return_1h"]
    for c in cols:
        if c not in panel:
            out[c] = None
            continue
        pair = pd.concat([panel[c], target], axis=1).dropna()
        out[c] = float(pair.iloc[:, 0].corr(pair.iloc[:, 1], method="spearman")) if len(pair) >= 20 else None
    return out
