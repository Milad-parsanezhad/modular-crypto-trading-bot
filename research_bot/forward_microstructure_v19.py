from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import isfinite
from statistics import mean
from typing import Iterable

import numpy as np
import pandas as pd


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
    min_venues_per_symbol: int = 2
    min_signed_trade_coverage: float = 0.80
    max_spread_bps: float = 50.0
    max_mid_dispersion_bps: float = 100.0
    max_venue_clock_skew_seconds: float = 60.0


def _finite(v: float) -> bool:
    try:
        return isfinite(float(v))
    except (TypeError, ValueError):
        return False


def validate_observation(obs: VenueMicrostructureObservation, cfg: V19MicrostructureConfig | None = None) -> list[str]:
    cfg = cfg or V19MicrostructureConfig()
    problems: list[str] = []
    if not obs.venue or not obs.symbol or not obs.source:
        problems.append("MISSING_IDENTITY")
    try:
        pd.Timestamp(obs.observed_at)
    except Exception:
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
    return problems


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
) -> VenueMicrostructureObservation:
    bid_rows = list(bids)[:levels]
    ask_rows = list(asks)[:levels]
    if not bid_rows or not ask_rows:
        raise ValueError("empty order book")
    best_bid = float(bid_rows[0][0])
    best_ask = float(ask_rows[0][0])
    bid_depth = float(sum(float(p) * float(q) for p, q, *_ in bid_rows))
    ask_depth = float(sum(float(p) * float(q) for p, q, *_ in ask_rows))

    buy = sell = unknown = 0.0
    trade_count = 0
    if trades is not None and not trades.empty:
        t = trades.copy()
        if "notional" not in t.columns:
            t["notional"] = pd.to_numeric(t["price"], errors="coerce") * pd.to_numeric(t["amount"], errors="coerce")
        for row in t.itertuples(index=False):
            notion = float(getattr(row, "notional", 0.0) or 0.0)
            if not np.isfinite(notion) or notion < 0:
                continue
            side = str(getattr(row, "side", "") or "").lower()
            trade_count += 1
            if side == "buy":
                buy += notion
            elif side == "sell":
                sell += notion
            else:
                unknown += notion

    return VenueMicrostructureObservation(
        venue=venue,
        symbol=symbol,
        observed_at=observed_at,
        best_bid=best_bid,
        best_ask=best_ask,
        bid_depth_notional=bid_depth,
        ask_depth_notional=ask_depth,
        trade_buy_notional=buy,
        trade_sell_notional=sell,
        trade_unknown_notional=unknown,
        trade_count=trade_count,
        source=source,
    )


def aggregate_symbol(observations: Iterable[VenueMicrostructureObservation], cfg: V19MicrostructureConfig | None = None) -> dict:
    cfg = cfg or V19MicrostructureConfig()
    obs = list(observations)
    if not obs:
        raise ValueError("no observations")
    symbols = {x.symbol for x in obs}
    if len(symbols) != 1:
        raise ValueError("aggregate_symbol requires one symbol")

    accepted = []
    rejected = []
    for x in obs:
        issues = validate_observation(x, cfg)
        record = {**asdict(x), "spread_bps": x.spread_bps, "depth_imbalance": x.depth_imbalance, "trade_imbalance": x.trade_imbalance, "signed_trade_coverage": x.signed_trade_coverage}
        if issues:
            record["quality_issues"] = issues
            rejected.append(record)
        else:
            accepted.append((x, record))

    symbol = next(iter(symbols))
    if len(accepted) < cfg.min_venues_per_symbol:
        return {
            "symbol": symbol,
            "status": "INSUFFICIENT_VENUE_COVERAGE",
            "accepted_venues": len(accepted),
            "required_venues": cfg.min_venues_per_symbol,
            "rejected": rejected,
            "feature_authorized": False,
        }

    mids = np.asarray([x.mid for x, _ in accepted], dtype=float)
    median_mid = float(np.median(mids))
    mid_dispersion_bps = float((np.max(mids) - np.min(mids)) / median_mid * 10_000.0) if median_mid > 0 else float("nan")
    trade_imbalances = [x.trade_imbalance for x, _ in accepted if x.signed_trade_coverage >= cfg.min_signed_trade_coverage]
    depth_imbalances = [x.depth_imbalance for x, _ in accepted]
    spreads = [x.spread_bps for x, _ in accepted]
    clocks = pd.to_datetime([x.observed_at for x, _ in accepted], utc=True, errors="coerce")
    if clocks.isna().any():
        clock_skew_seconds = float("inf")
    else:
        clock_skew_seconds = float((clocks.max() - clocks.min()).total_seconds())

    signs = [np.sign(x) for x in trade_imbalances if x != 0]
    sign_agreement = float(abs(sum(signs)) / len(signs)) if signs else 0.0
    quality_flags: list[str] = []
    if mid_dispersion_bps > cfg.max_mid_dispersion_bps:
        quality_flags.append("CROSS_VENUE_MID_DISPERSION_TOO_LARGE")
    if len(trade_imbalances) < cfg.min_venues_per_symbol:
        quality_flags.append("SIGNED_TRADE_COVERAGE_INSUFFICIENT")
    if clock_skew_seconds > cfg.max_venue_clock_skew_seconds:
        quality_flags.append("CROSS_VENUE_CLOCK_SKEW_TOO_LARGE")

    feature_authorized = not quality_flags
    return {
        "symbol": symbol,
        "status": "OK" if feature_authorized else "QUALITY_GATED",
        "accepted_venues": len(accepted),
        "required_venues": cfg.min_venues_per_symbol,
        "median_mid": median_mid,
        "mid_dispersion_bps": mid_dispersion_bps,
        "venue_clock_skew_seconds": clock_skew_seconds,
        "mean_spread_bps": float(mean(spreads)),
        "mean_depth_imbalance": float(mean(depth_imbalances)),
        "mean_trade_imbalance": float(mean(trade_imbalances)) if trade_imbalances else None,
        "trade_sign_agreement": sign_agreement,
        "feature_authorized": feature_authorized,
        "quality_flags": quality_flags,
        "venues": [r for _, r in accepted],
        "rejected": rejected,
    }


def build_snapshot(observations: Iterable[VenueMicrostructureObservation], cfg: V19MicrostructureConfig | None = None) -> dict:
    cfg = cfg or V19MicrostructureConfig()
    obs = list(observations)
    by_symbol: dict[str, list[VenueMicrostructureObservation]] = {}
    for x in obs:
        by_symbol.setdefault(x.symbol, []).append(x)
    symbol_rows = [aggregate_symbol(by_symbol[s], cfg) for s in sorted(by_symbol)]
    complete = [x for x in symbol_rows if x.get("feature_authorized")]
    payload = {
        "version": "v0.19",
        "research_status": "PROSPECTIVE_MULTI_VENUE_MICROSTRUCTURE_COLLECTION_ONLY",
        "config": asdict(cfg),
        "symbols": symbol_rows,
        "authorized_symbol_count": len(complete),
        "signal_authorized": False,
        "paper_strategy_replacement_authorized": False,
        "live_execution_authorized": False,
        "claim_scope": "prospective public-data feature collection; no alpha claim before sufficient forward history",
    }
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    payload["snapshot_sha256"] = sha256(raw).hexdigest()
    return payload
