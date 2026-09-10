from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from hashlib import sha256
import json
from math import isfinite
from statistics import mean
from typing import Iterable

import numpy as np
import pandas as pd


TRADE_SIDE_SEMANTICS = "exchange_reported_side_unverified_aggressor"


@dataclass(frozen=True)
class VenueMicrostructureObservation:
    venue: str
    symbol: str
    observed_at: str
    best_bid: float
    best_ask: float
    bid_depth_notional: float
    ask_depth_notional: float
    trade_buy_notional: float
    trade_sell_notional: float
    trade_unknown_notional: float
    trade_count: int
    source: str
    trade_window_start: str | None = None
    trade_window_end: str | None = None
    trade_first_at: str | None = None
    trade_last_at: str | None = None
    trade_window_seconds: int = 60
    raw_trade_count: int = 0
    future_trade_count_excluded: int = 0
    trade_staleness_seconds: float | None = None
    trade_side_semantics: str = TRADE_SIDE_SEMANTICS

    @property
    def mid(self) -> float:
        return (self.best_bid + self.best_ask) / 2.0

    @property
    def spread_bps(self) -> float:
        m = self.mid
        return (self.best_ask - self.best_bid) / m * 10_000.0 if m > 0 else float("nan")

    @property
    def depth_imbalance(self) -> float:
        d = self.bid_depth_notional + self.ask_depth_notional
        return (self.bid_depth_notional - self.ask_depth_notional) / d if d > 0 else 0.0

    @property
    def trade_imbalance(self) -> float:
        d = self.trade_buy_notional + self.trade_sell_notional
        return (self.trade_buy_notional - self.trade_sell_notional) / d if d > 0 else 0.0

    @property
    def signed_trade_coverage(self) -> float:
        d = self.trade_buy_notional + self.trade_sell_notional + self.trade_unknown_notional
        return (self.trade_buy_notional + self.trade_sell_notional) / d if d > 0 else 0.0


@dataclass(frozen=True)
class V19MicrostructureConfig:
    symbols: tuple[str, ...] = ("BTC/USDT", "ETH/USDT")
    venues: tuple[str, ...] = ("coinex", "okx", "kucoin")
    depth_levels: int = 10
    trades_limit: int = 500
    # Measurement window is intentionally much shorter than the 4h trading horizon.
    # It is a point-in-time microstructure sample, not a new trading timeframe.
    trade_window_seconds: int = 60
    min_recent_trade_count: int = 5
    max_trade_staleness_seconds: float = 30.0
    min_venues_per_symbol: int = 2
    min_signed_trade_coverage: float = 0.80
    max_spread_bps: float = 50.0
    max_mid_dispersion_bps: float = 100.0
    max_venue_clock_skew_seconds: float = 60.0


def _utc_timestamp(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        raise ValueError("invalid timestamp")
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _numeric_timestamp_to_utc(value: float) -> pd.Timestamp:
    """Parse Unix timestamps defensively across seconds/ms/us/ns magnitudes."""
    x = float(value)
    if not np.isfinite(x):
        return pd.NaT
    a = abs(x)
    if a >= 1e17:
        unit = "ns"
    elif a >= 1e14:
        unit = "us"
    elif a >= 1e11:
        unit = "ms"
    elif a >= 1e8:
        unit = "s"
    else:
        return pd.NaT
    return pd.to_datetime(x, unit=unit, utc=True, errors="coerce")


def _coerce_trade_timestamp(series: pd.Series) -> pd.Series:
    """Coerce mixed provider/CCXT timestamps to UTC without unit leakage."""
    out = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns, UTC]")
    for idx, value in series.items():
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        try:
            if isinstance(value, (pd.Timestamp, datetime, np.datetime64)):
                out.loc[idx] = _utc_timestamp(value)
                continue
            if isinstance(value, (int, float, np.integer, np.floating)):
                out.loc[idx] = _numeric_timestamp_to_utc(float(value))
                continue
            text = str(value).strip()
            if not text:
                continue
            try:
                number = float(text)
            except ValueError:
                out.loc[idx] = pd.to_datetime(text, utc=True, errors="coerce")
            else:
                out.loc[idx] = _numeric_timestamp_to_utc(number)
        except Exception:
            out.loc[idx] = pd.NaT
    return out


def summarize_trade_window(
    trades: pd.DataFrame | None,
    *,
    observed_at: str,
    window_seconds: int,
) -> dict:
    """Summarize trades in the closed PIT interval [t-window, t]."""
    if int(window_seconds) <= 0:
        raise ValueError("window_seconds must be positive")
    end = _utc_timestamp(observed_at)
    start = end - pd.Timedelta(seconds=int(window_seconds))
    empty = {
        "trade_buy_notional": 0.0,
        "trade_sell_notional": 0.0,
        "trade_unknown_notional": 0.0,
        "trade_count": 0,
        "raw_trade_count": 0,
        "future_trade_count_excluded": 0,
        "trade_window_start": start.isoformat(),
        "trade_window_end": end.isoformat(),
        "trade_first_at": None,
        "trade_last_at": None,
        "trade_window_seconds": int(window_seconds),
        "trade_staleness_seconds": None,
        "trade_side_semantics": TRADE_SIDE_SEMANTICS,
    }
    if trades is None or trades.empty or "timestamp" not in trades.columns:
        return empty

    x = trades.copy()
    x["_ts"] = _coerce_trade_timestamp(x["timestamp"])
    x = x.dropna(subset=["_ts"]).copy()
    raw_count = int(len(x))
    future_count = int((x["_ts"] > end).sum())
    x = x[(x["_ts"] >= start) & (x["_ts"] <= end)].copy()
    if x.empty:
        return {**empty, "raw_trade_count": raw_count, "future_trade_count_excluded": future_count}

    if "notional" not in x.columns:
        if "price" not in x.columns or "amount" not in x.columns:
            raise ValueError("trades require notional or price+amount")
        x["notional"] = pd.to_numeric(x["price"], errors="coerce") * pd.to_numeric(x["amount"], errors="coerce")
    x["notional"] = pd.to_numeric(x["notional"], errors="coerce")
    x = x[np.isfinite(x["notional"]) & (x["notional"] >= 0.0)].copy()
    if x.empty:
        return {**empty, "raw_trade_count": raw_count, "future_trade_count_excluded": future_count}

    side = x.get("side", pd.Series("", index=x.index)).fillna("").astype(str).str.lower()
    buy = float(x.loc[side.eq("buy"), "notional"].sum())
    sell = float(x.loc[side.eq("sell"), "notional"].sum())
    unknown = float(x.loc[~side.isin(["buy", "sell"]), "notional"].sum())
    first = x["_ts"].min()
    last = x["_ts"].max()
    staleness = float(max(0.0, (end - last).total_seconds()))
    return {
        "trade_buy_notional": buy,
        "trade_sell_notional": sell,
        "trade_unknown_notional": unknown,
        "trade_count": int(len(x)),
        "raw_trade_count": raw_count,
        "future_trade_count_excluded": future_count,
        "trade_window_start": start.isoformat(),
        "trade_window_end": end.isoformat(),
        "trade_first_at": first.isoformat(),
        "trade_last_at": last.isoformat(),
        "trade_window_seconds": int(window_seconds),
        "trade_staleness_seconds": staleness,
        "trade_side_semantics": TRADE_SIDE_SEMANTICS,
    }


def _finite(v: float) -> bool:
    try:
        return isfinite(float(v))
    except (TypeError, ValueError):
        return False


def validate_observation(obs: VenueMicrostructureObservation, cfg: V19MicrostructureConfig | None = None) -> list[str]:
    cfg = cfg or V19MicrostructureConfig()
    problems: list[str] = []
    venue = str(obs.venue or "").lower()
    if not obs.venue or not obs.symbol or not obs.source:
        problems.append("MISSING_IDENTITY")
    if venue and venue not in {v.lower() for v in cfg.venues}:
        problems.append("UNCONFIGURED_VENUE")
    if obs.symbol and obs.symbol not in set(cfg.symbols):
        problems.append("UNCONFIGURED_SYMBOL")

    try:
        observed = _utc_timestamp(obs.observed_at)
    except Exception:
        observed = None
        problems.append("INVALID_OBSERVED_AT")

    if not _finite(obs.best_bid) or not _finite(obs.best_ask) or obs.best_bid <= 0 or obs.best_ask <= 0:
        problems.append("INVALID_TOP_OF_BOOK")
    elif obs.best_ask < obs.best_bid:
        problems.append("CROSSED_BOOK")
    if not _finite(obs.spread_bps) or obs.spread_bps < 0:
        problems.append("INVALID_SPREAD")
    elif obs.spread_bps > cfg.max_spread_bps:
        problems.append("SPREAD_TOO_WIDE")
    if min(obs.bid_depth_notional, obs.ask_depth_notional, obs.trade_buy_notional, obs.trade_sell_notional, obs.trade_unknown_notional) < 0:
        problems.append("NEGATIVE_NOTIONAL")
    if not -1.0000001 <= obs.depth_imbalance <= 1.0000001:
        problems.append("INVALID_DEPTH_IMBALANCE")
    if not -1.0000001 <= obs.trade_imbalance <= 1.0000001:
        problems.append("INVALID_TRADE_IMBALANCE")

    if obs.trade_side_semantics != TRADE_SIDE_SEMANTICS:
        problems.append("UNDECLARED_TRADE_SIDE_SEMANTICS")
    if int(obs.trade_window_seconds) != int(cfg.trade_window_seconds):
        problems.append("TRADE_WINDOW_MISMATCH")

    # The stored window is part of the evidence contract, not merely metadata.
    if observed is not None:
        expected_end = observed
        expected_start = observed - pd.Timedelta(seconds=int(cfg.trade_window_seconds))
        try:
            if obs.trade_window_end is None or _utc_timestamp(obs.trade_window_end) != expected_end:
                problems.append("TRADE_WINDOW_END_MISMATCH")
        except Exception:
            problems.append("TRADE_WINDOW_END_MISMATCH")
        try:
            if obs.trade_window_start is None or _utc_timestamp(obs.trade_window_start) != expected_start:
                problems.append("TRADE_WINDOW_START_MISMATCH")
        except Exception:
            problems.append("TRADE_WINDOW_START_MISMATCH")

    if obs.trade_count < int(cfg.min_recent_trade_count):
        problems.append("INSUFFICIENT_RECENT_TRADES")
    if obs.trade_staleness_seconds is None or not _finite(obs.trade_staleness_seconds):
        problems.append("TRADE_STALENESS_UNKNOWN")
    elif float(obs.trade_staleness_seconds) > float(cfg.max_trade_staleness_seconds):
        problems.append("STALE_RECENT_TRADES")

    if observed is not None:
        for field_name, value in (("TRADE_FIRST_AT", obs.trade_first_at), ("TRADE_LAST_AT", obs.trade_last_at)):
            if value is None:
                continue
            try:
                ts = _utc_timestamp(value)
                start = observed - pd.Timedelta(seconds=int(cfg.trade_window_seconds))
                if ts < start or ts > observed:
                    problems.append(f"{field_name}_OUTSIDE_WINDOW")
            except Exception:
                problems.append(f"INVALID_{field_name}")
        if obs.trade_last_at is not None:
            try:
                if _utc_timestamp(obs.trade_last_at) > observed:
                    problems.append("FUTURE_TRADE_LEAKAGE")
            except Exception:
                problems.append("INVALID_TRADE_LAST_AT")
    return sorted(set(problems))


def observation_from_orderbook_and_trades(
    *,
    venue: str,
    symbol: str,
    observed_at: str,
    bids: Iterable[Iterable[float]],
    asks: Iterable[Iterable[float]],
    trades: pd.DataFrame,
    source: str,
    levels: int = 10,
    trade_window_seconds: int = 60,
) -> VenueMicrostructureObservation:
    bid_rows = list(bids)[:levels]
    ask_rows = list(asks)[:levels]
    if not bid_rows or not ask_rows:
        raise ValueError("empty order book")
    best_bid = float(bid_rows[0][0])
    best_ask = float(ask_rows[0][0])
    bid_depth = float(sum(float(p) * float(q) for p, q, *_ in bid_rows))
    ask_depth = float(sum(float(p) * float(q) for p, q, *_ in ask_rows))
    tw = summarize_trade_window(trades, observed_at=observed_at, window_seconds=trade_window_seconds)

    return VenueMicrostructureObservation(
        venue=str(venue).lower(),
        symbol=symbol,
        observed_at=_utc_timestamp(observed_at).isoformat(),
        best_bid=best_bid,
        best_ask=best_ask,
        bid_depth_notional=bid_depth,
        ask_depth_notional=ask_depth,
        trade_buy_notional=tw["trade_buy_notional"],
        trade_sell_notional=tw["trade_sell_notional"],
        trade_unknown_notional=tw["trade_unknown_notional"],
        trade_count=tw["trade_count"],
        source=source,
        trade_window_start=tw["trade_window_start"],
        trade_window_end=tw["trade_window_end"],
        trade_first_at=tw["trade_first_at"],
        trade_last_at=tw["trade_last_at"],
        trade_window_seconds=tw["trade_window_seconds"],
        raw_trade_count=tw["raw_trade_count"],
        future_trade_count_excluded=tw["future_trade_count_excluded"],
        trade_staleness_seconds=tw["trade_staleness_seconds"],
        trade_side_semantics=tw["trade_side_semantics"],
    )


def _insufficient_row(symbol: str, cfg: V19MicrostructureConfig, accepted: list[tuple[VenueMicrostructureObservation, dict]], rejected: list[dict], flags: list[str]) -> dict:
    names = sorted({x.venue for x, _ in accepted})
    return {
        "symbol": symbol,
        "status": "INSUFFICIENT_VENUE_COVERAGE",
        "accepted_venues": len(names),
        "accepted_venue_names": names,
        "required_venues": cfg.min_venues_per_symbol,
        "observed_at_min": None,
        "observed_at_max": None,
        "median_mid": None,
        "mid_dispersion_bps": None,
        "venue_clock_skew_seconds": None,
        "mean_spread_bps": None,
        "mean_depth_imbalance": None,
        "mean_reported_trade_imbalance": None,
        "mean_trade_imbalance": None,
        "trade_sign_agreement": None,
        "trade_side_semantics": TRADE_SIDE_SEMANTICS,
        "trade_window_seconds": int(cfg.trade_window_seconds),
        "feature_authorized": False,
        "quality_flags": sorted(set(flags + ["INSUFFICIENT_UNIQUE_VENUE_COVERAGE"])),
        "venues": [r for _, r in accepted],
        "rejected": rejected,
    }


def aggregate_symbol(observations: Iterable[VenueMicrostructureObservation], cfg: V19MicrostructureConfig | None = None) -> dict:
    cfg = cfg or V19MicrostructureConfig()
    obs = list(observations)
    if not obs:
        raise ValueError("no observations")
    symbols = {x.symbol for x in obs}
    if len(symbols) != 1:
        raise ValueError("aggregate_symbol requires one symbol")

    valid: list[tuple[VenueMicrostructureObservation, dict]] = []
    rejected: list[dict] = []
    for x in obs:
        issues = validate_observation(x, cfg)
        record = {
            **asdict(x),
            "spread_bps": x.spread_bps,
            "depth_imbalance": x.depth_imbalance,
            "reported_trade_imbalance": x.trade_imbalance,
            "trade_imbalance": x.trade_imbalance,
            "signed_trade_coverage": x.signed_trade_coverage,
        }
        if issues:
            record["quality_issues"] = issues
            rejected.append(record)
        else:
            valid.append((x, record))

    # Cross-venue evidence requires distinct venues. A duplicated observation from
    # one provider must never masquerade as two independent venues. Keep only the
    # latest valid observation per venue and gate the symbol if duplicates exist.
    by_venue: dict[str, list[tuple[VenueMicrostructureObservation, dict]]] = {}
    for item in valid:
        by_venue.setdefault(item[0].venue, []).append(item)
    accepted: list[tuple[VenueMicrostructureObservation, dict]] = []
    aggregate_flags: list[str] = []
    for venue, items in sorted(by_venue.items()):
        items = sorted(items, key=lambda item: _utc_timestamp(item[0].observed_at))
        accepted.append(items[-1])
        if len(items) > 1:
            aggregate_flags.append("DUPLICATE_VENUE_OBSERVATIONS")
            for extra_obs, extra_record in items[:-1]:
                duplicate_record = dict(extra_record)
                duplicate_record["quality_issues"] = ["DUPLICATE_VENUE_OBSERVATION"]
                rejected.append(duplicate_record)

    symbol = next(iter(symbols))
    unique_venue_count = len({x.venue for x, _ in accepted})
    if unique_venue_count < cfg.min_venues_per_symbol:
        return _insufficient_row(symbol, cfg, accepted, rejected, aggregate_flags)

    mids = np.asarray([x.mid for x, _ in accepted], dtype=float)
    median_mid = float(np.median(mids))
    mid_dispersion_bps = float((np.max(mids) - np.min(mids)) / median_mid * 10_000.0) if median_mid > 0 else float("nan")
    trade_imbalances = [x.trade_imbalance for x, _ in accepted if x.signed_trade_coverage >= cfg.min_signed_trade_coverage]
    depth_imbalances = [x.depth_imbalance for x, _ in accepted]
    spreads = [x.spread_bps for x, _ in accepted]
    clocks = pd.to_datetime([x.observed_at for x, _ in accepted], utc=True, errors="coerce")
    if clocks.isna().any():
        clock_skew_seconds = float("inf")
        observed_min = observed_max = None
    else:
        clock_skew_seconds = float((clocks.max() - clocks.min()).total_seconds())
        observed_min = clocks.min().isoformat()
        observed_max = clocks.max().isoformat()

    signs = [np.sign(x) for x in trade_imbalances if x != 0]
    sign_agreement = float(abs(sum(signs)) / len(signs)) if signs else 0.0
    quality_flags: list[str] = list(aggregate_flags)
    if mid_dispersion_bps > cfg.max_mid_dispersion_bps:
        quality_flags.append("CROSS_VENUE_MID_DISPERSION_TOO_LARGE")
    if len(trade_imbalances) < cfg.min_venues_per_symbol:
        quality_flags.append("SIGNED_TRADE_COVERAGE_INSUFFICIENT")
    if clock_skew_seconds > cfg.max_venue_clock_skew_seconds:
        quality_flags.append("CROSS_VENUE_CLOCK_SKEW_TOO_LARGE")

    quality_flags = sorted(set(quality_flags))
    feature_authorized = not quality_flags
    mean_reported = float(mean(trade_imbalances)) if trade_imbalances else None
    accepted_names = sorted({x.venue for x, _ in accepted})
    return {
        "symbol": symbol,
        "status": "OK" if feature_authorized else "QUALITY_GATED",
        "accepted_venues": len(accepted_names),
        "accepted_venue_names": accepted_names,
        "required_venues": cfg.min_venues_per_symbol,
        "observed_at_min": observed_min,
        "observed_at_max": observed_max,
        "median_mid": median_mid,
        "mid_dispersion_bps": mid_dispersion_bps,
        "venue_clock_skew_seconds": clock_skew_seconds,
        "mean_spread_bps": float(mean(spreads)),
        "mean_depth_imbalance": float(mean(depth_imbalances)),
        "mean_reported_trade_imbalance": mean_reported,
        "mean_trade_imbalance": mean_reported,
        "trade_sign_agreement": sign_agreement,
        "trade_side_semantics": TRADE_SIDE_SEMANTICS,
        "trade_window_seconds": int(cfg.trade_window_seconds),
        "feature_authorized": feature_authorized,
        "quality_flags": quality_flags,
        "venues": [r for _, r in accepted],
        "rejected": rejected,
    }


def _missing_symbol_row(symbol: str, cfg: V19MicrostructureConfig) -> dict:
    return _insufficient_row(symbol, cfg, [], [], ["NO_OBSERVATIONS_FOR_CONFIGURED_SYMBOL"])


def build_snapshot(observations: Iterable[VenueMicrostructureObservation], cfg: V19MicrostructureConfig | None = None) -> dict:
    cfg = cfg or V19MicrostructureConfig()
    obs = list(observations)
    configured_symbols = set(cfg.symbols)
    by_symbol: dict[str, list[VenueMicrostructureObservation]] = {s: [] for s in cfg.symbols}
    unexpected_observations = 0
    for x in obs:
        if x.symbol in configured_symbols:
            by_symbol[x.symbol].append(x)
        else:
            unexpected_observations += 1

    symbol_rows = [aggregate_symbol(by_symbol[s], cfg) if by_symbol[s] else _missing_symbol_row(s, cfg) for s in cfg.symbols]
    complete = [x for x in symbol_rows if x.get("feature_authorized")]
    payload = {
        "version": "v0.19",
        "research_status": "PROSPECTIVE_MULTI_VENUE_MICROSTRUCTURE_COLLECTION_ONLY",
        "measurement_contract": {
            "primary_trading_horizon": "4h",
            "snapshot_measurement_frequency_target": "30min",
            "reported_trade_window_seconds": int(cfg.trade_window_seconds),
            "trade_side_semantics": TRADE_SIDE_SEMANTICS,
            "rest_snapshot_not_event_stream": True,
            "venue_coverage_unit": "unique_venue",
        },
        "config": asdict(cfg),
        "symbols": symbol_rows,
        "authorized_symbol_count": len(complete),
        "unexpected_observation_count": unexpected_observations,
        "signal_authorized": False,
        "paper_strategy_replacement_authorized": False,
        "live_execution_authorized": False,
        "claim_scope": "prospective public REST microstructure measurement only; no alpha claim before sufficient forward history",
    }
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    payload["snapshot_sha256"] = sha256(raw).hexdigest()
    return payload
