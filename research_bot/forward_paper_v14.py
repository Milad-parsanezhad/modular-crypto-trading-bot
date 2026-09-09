from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Protocol

import numpy as np
import pandas as pd

from .coinex_depth import DepthSnapshot, fetch_coinex_depth
from .coinex_public import PERIOD_MS, fetch_coinex_klines
from .contracts import ExecutionMode
from .execution import ExecutionPolicy, ExecutionRequest, OrderSide, PaperExecutionEngine
from .ichimoku_advanced import detect_kumo_triangle_breakout
from .persistence import PaperAccount, PaperPosition
from .risk import RiskEngine, RiskLimits, RiskSnapshot


STRATEGY_VERSION = "ICHIMOKU_SHADOW_V14"


@dataclass(frozen=True)
class ForwardPaperConfig:
    symbols: tuple[str, ...] = ("BTC/USDT", "ETH/USDT")
    period: str = "4hour"
    bars: int = 280
    initial_cash: float = 10_000.0
    risk_fraction: float = 0.005
    max_order_notional: float = 2_000.0
    entry_rule_score: float = 0.80
    fee_bps: float = 10.0
    slippage_bps: float = 2.0
    max_spread_bps: float = 35.0
    max_drawdown: float = 0.12
    max_asset_weight: float = 0.25
    min_stop_fraction: float = 0.01
    atr_stop_multiple: float = 2.0


@dataclass(frozen=True)
class ShadowSignal:
    action: str
    rule_score: float
    reasons: tuple[str, ...]
    atr_pct: float
    reference_price: float
    features: dict


class ForwardMarketClient(Protocol):
    def klines(self, symbol: str, period: str, bars: int) -> pd.DataFrame: ...
    def depth(self, symbol: str) -> DepthSnapshot: ...


class CoinExForwardMarketClient:
    def klines(self, symbol: str, period: str, bars: int) -> pd.DataFrame:
        return fetch_coinex_klines(symbol, period=period, market_type="spot", bars=bars)

    def depth(self, symbol: str) -> DepthSnapshot:
        return fetch_coinex_depth(symbol)


def _last_closed_bars(df: pd.DataFrame, period: str, now: datetime | None = None) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    if period not in PERIOD_MS:
        raise ValueError(f"unsupported period: {period}")
    now = now or datetime.now(timezone.utc)
    x = df.copy()
    x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True)
    close_time = x["timestamp"] + pd.Timedelta(milliseconds=PERIOD_MS[period])
    return x[close_time <= pd.Timestamp(now)].sort_values("timestamp").reset_index(drop=True)


def _atr_pct(df: pd.DataFrame, window: int = 14) -> float:
    if len(df) < window + 1:
        return float("nan")
    prev = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev).abs(),
            (df["low"] - prev).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1.0 / window, adjust=False).mean()
    last = float(df["close"].iloc[-1])
    return float(atr.iloc[-1] / last) if last > 0 else float("nan")


def frozen_shadow_signal(df: pd.DataFrame, currently_long: bool, entry_rule_score: float = 0.80) -> ShadowSignal:
    """Frozen transparent paper-only Ichimoku rule.

    The rule is deliberately not labelled as validated alpha. It exists to
    collect forward execution/risk evidence after v0.12 produced no promoted
    learned model.
    """
    if len(df) < 120:
        raise ValueError("at least 120 closed bars are required")
    x = detect_kumo_triangle_breakout(df)
    r = x.iloc[-1]
    mom6 = float(df["close"].iloc[-1] / df["close"].iloc[-7] - 1.0)
    checks = {
        "price_above_cloud": bool(float(r["ichi_price_above_cloud"]) >= 0.5),
        "tk_bullish": bool(float(r["ichi_tk_bullish"]) >= 0.5),
        "cloud_bullish": bool(float(r["ichi_cloud_bullish"]) >= 0.5),
        "price_above_kijun": bool(float(r["ichi_price_kijun_pct"]) > 0.0),
        "momentum_6_positive": bool(mom6 > 0.0),
    }
    score = float(sum(checks.values()) / len(checks))
    triangle = bool(float(r.get("triangle_candidate", 0.0)) >= 0.5)
    if triangle:
        score = 1.0

    below_cloud = bool(float(r["ichi_price_below_cloud"]) >= 0.5)
    bearish_tk_below_kijun = not checks["tk_bullish"] and float(r["ichi_price_kijun_pct"]) < 0.0
    reasons = [k.upper() for k, v in checks.items() if v]
    if triangle:
        reasons.append("TRIANGLE_KUMO_CANDIDATE")

    if currently_long and (below_cloud or bearish_tk_below_kijun):
        action = "EXIT"
        reasons.append("FROZEN_ICHIMOKU_EXIT")
    elif currently_long:
        action = "HOLD"
    elif score >= float(entry_rule_score):
        action = "BUY_CANDIDATE"
        reasons.append("FROZEN_ICHIMOKU_ENTRY_SCORE")
    else:
        action = "NO_TRADE"

    atr = _atr_pct(df)
    features = {
        "price_above_cloud": checks["price_above_cloud"],
        "tk_bullish": checks["tk_bullish"],
        "cloud_bullish": checks["cloud_bullish"],
        "price_above_kijun": checks["price_above_kijun"],
        "momentum_6": mom6,
        "triangle_candidate": triangle,
        "ichi_tk_distance_pct": float(r["ichi_tk_distance_pct"]),
        "ichi_price_kijun_pct": float(r["ichi_price_kijun_pct"]),
        "ichi_cloud_width_pct": float(r["ichi_cloud_width_pct"]),
    }
    return ShadowSignal(action=action, rule_score=score, reasons=tuple(reasons), atr_pct=float(atr), reference_price=float(df["close"].iloc[-1]), features=features)


class ForwardPaperRunner:
    """Forward-only CoinEx paper observer/executor with restart deduplication."""

    def __init__(self, store, *, config: ForwardPaperConfig | None = None, market_client: ForwardMarketClient | None = None, paper_execution_enabled: bool = True):
        self.store = store
        self.config = config or ForwardPaperConfig()
        self.market = market_client or CoinExForwardMarketClient()
        self.paper_execution_enabled = bool(paper_execution_enabled)
        self.risk_engine = RiskEngine(RiskLimits(max_drawdown=self.config.max_drawdown, max_gross_exposure=1.0, max_asset_weight=self.config.max_asset_weight, max_turnover_per_step=0.50, max_spread_bps=self.config.max_spread_bps, max_slippage_bps=25.0, max_cvar_95=0.035, min_cash_buffer=0.05))
        self.execution = PaperExecutionEngine(ExecutionPolicy(mode=ExecutionMode.PAPER, fee_bps=self.config.fee_bps, slippage_bps=self.config.slippage_bps, max_order_notional=self.config.max_order_notional, live_execution_enabled=False))
        self._last_marks: dict[str, float] = {}

    def _mark_account(self, current_symbol: str, current_price: float) -> tuple[PaperAccount, float]:
        account = self.store.get_account()
        self._last_marks[current_symbol] = float(current_price)
        gross = 0.0
        position_value = 0.0
        for pos in self.store.list_positions():
            mark = self._last_marks.get(pos.symbol, pos.avg_price)
            value = float(pos.quantity * mark)
            gross += abs(value)
            position_value += value
        equity = float(account.cash + position_value)
        peak = max(float(account.peak_equity), equity)
        updated = PaperAccount(cash=float(account.cash), equity=equity, peak_equity=peak)
        self.store.set_account(updated)
        return updated, (gross / equity if equity > 0 else 0.0)

    def _apply_fill(self, symbol: str, fill) -> PaperAccount:
        account = self.store.get_account()
        pos = self.store.get_position(symbol)
        q = float(fill.filled_quantity)
        if fill.side is OrderSide.BUY:
            cost = q * float(fill.fill_price) + float(fill.fee_paid)
            cash = float(account.cash - cost)
            old_cost = float(pos.quantity * pos.avg_price)
            new_qty = float(pos.quantity + q)
            avg = (old_cost + q * float(fill.fill_price)) / new_qty if new_qty > 0 else 0.0
            self.store.set_position(PaperPosition(symbol, new_qty, float(avg)))
        else:
            proceeds = q * float(fill.fill_price) - float(fill.fee_paid)
            cash = float(account.cash + proceeds)
            new_qty = max(0.0, float(pos.quantity - q))
            self.store.set_position(PaperPosition(symbol, new_qty, pos.avg_price if new_qty > 0 else 0.0))
        account = PaperAccount(cash=cash, equity=account.equity, peak_equity=account.peak_equity)
        self.store.set_account(account)
        return account

    def run_symbol(self, symbol: str, now: datetime | None = None) -> dict:
        now = now or datetime.now(timezone.utc)
        raw = self.market.klines(symbol, self.config.period, self.config.bars)
        bars = _last_closed_bars(raw, self.config.period, now=now)
        if len(bars) < 120:
            return {"symbol": symbol, "status": "INSUFFICIENT_CLOSED_BARS", "rows": int(len(bars))}

        position = self.store.get_position(symbol)
        signal = frozen_shadow_signal(bars, currently_long=position.quantity > 0, entry_rule_score=self.config.entry_rule_score)
        depth = self.market.depth(symbol)
        bar_ts = pd.Timestamp(bars["timestamp"].iloc[-1]).to_pydatetime()
        observation = {
            "observed_at": now.isoformat(), "bar_timestamp": bar_ts.isoformat(), "symbol": symbol,
            "strategy_version": STRATEGY_VERSION, "action": signal.action, "rule_score": signal.rule_score,
            "reference_price": signal.reference_price, "spread_bps": depth.spread_bps,
            "atr_pct": signal.atr_pct if np.isfinite(signal.atr_pct) else None,
            "features": {**signal.features, "depth_imbalance": depth.imbalance, "best_bid": depth.best_bid, "best_ask": depth.best_ask, "evidence_label": "FORWARD_PAPER_HYPOTHESIS"},
            "reasons": list(signal.reasons),
        }
        if not self.store.record_observation(observation):
            return {"symbol": symbol, "status": "ALREADY_OBSERVED_BAR", "bar_timestamp": bar_ts.isoformat(), "action": signal.action}

        account, gross_exposure = self._mark_account(symbol, signal.reference_price)
        actionable = signal.action in {"BUY_CANDIDATE", "EXIT"}
        if not actionable or not self.paper_execution_enabled:
            self.store.record_equity({"timestamp": now.isoformat(), "equity": account.equity, "cash": account.cash, "gross_exposure": gross_exposure, "metadata": {"symbol": symbol, "action": signal.action, "paper_execution_enabled": self.paper_execution_enabled}})
            return {"symbol": symbol, "status": "OBSERVED_ONLY" if not self.paper_execution_enabled else "ABSTAINED", "signal": asdict(signal), "depth": asdict(depth), "account": asdict(account)}

        if signal.action == "BUY_CANDIDATE":
            stop_fraction = max(self.config.min_stop_fraction, self.config.atr_stop_multiple * (signal.atr_pct if np.isfinite(signal.atr_pct) else self.config.min_stop_fraction))
            risk_notional = self.risk_engine.position_size_from_risk(equity=account.equity, stop_distance_fraction=stop_fraction, risk_fraction=self.config.risk_fraction, volatility_scale=1.0)
            notional = min(risk_notional, self.config.max_order_notional, max(0.0, account.cash * 0.90))
            quantity = notional / signal.reference_price if signal.reference_price > 0 else 0.0
            proposed_weight = notional / account.equity if account.equity > 0 else 1.0
            turnover = proposed_weight
        else:
            quantity = float(position.quantity)
            notional = quantity * signal.reference_price
            proposed_weight = notional / account.equity if account.equity > 0 else 1.0
            turnover = proposed_weight

        if quantity <= 0:
            return {"symbol": symbol, "status": "NO_POSITION_OR_SIZE", "signal": asdict(signal)}

        risk = self.risk_engine.evaluate(RiskSnapshot(equity=account.equity, peak_equity=account.peak_equity, gross_exposure=gross_exposure + (proposed_weight if signal.action == "BUY_CANDIDATE" else 0.0), asset_weight=proposed_weight, turnover=turnover, spread_bps=float(depth.spread_bps), slippage_bps=float(self.config.slippage_bps + max(0.0, depth.spread_bps / 2.0)), recent_returns=tuple(self.store.recent_equity_returns(limit=100))))
        if not risk.approved:
            self.store.record_equity({"timestamp": now.isoformat(), "equity": account.equity, "cash": account.cash, "gross_exposure": gross_exposure, "metadata": {"symbol": symbol, "risk_rejected": list(risk.reasons)}})
            return {"symbol": symbol, "status": "RISK_REJECTED", "signal": asdict(signal), "risk": asdict(risk)}

        side = OrderSide.BUY if signal.action == "BUY_CANDIDATE" else OrderSide.SELL
        visible_qty = depth.ask_depth if side is OrderSide.BUY else depth.bid_depth
        fill_fraction = float(min(1.0, max(0.05, visible_qty / quantity))) if quantity > 0 else 0.0
        extra_slippage = float(min(20.0, max(0.0, depth.spread_bps / 2.0)))
        client_order_id = f"{STRATEGY_VERSION}:{symbol.replace('/','')}:{int(pd.Timestamp(bar_ts).timestamp())}:{side.value}"
        fill = self.execution.execute(ExecutionRequest(client_order_id=client_order_id, symbol=symbol, side=side, quantity=quantity, reference_price=signal.reference_price, created_at=now), fill_fraction=fill_fraction, extra_slippage_bps=extra_slippage)
        self._apply_fill(symbol, fill)
        self.store.record_fill({"timestamp": fill.timestamp.isoformat(), "client_order_id": fill.client_order_id, "symbol": fill.symbol, "side": fill.side.value, "requested_quantity": fill.requested_quantity, "filled_quantity": fill.filled_quantity, "fill_price": fill.fill_price, "fee_paid": fill.fee_paid, "slippage_paid": fill.slippage_paid, "status": fill.status, "strategy_version": STRATEGY_VERSION, "metadata": {"rule_score": signal.rule_score, "risk_reasons": list(risk.reasons), "spread_bps": depth.spread_bps, "paper_only": True}})
        account, gross_exposure = self._mark_account(symbol, signal.reference_price)
        self.store.record_equity({"timestamp": now.isoformat(), "equity": account.equity, "cash": account.cash, "gross_exposure": gross_exposure, "metadata": {"symbol": symbol, "fill_status": fill.status}})
        return {"symbol": symbol, "status": "EXECUTED_PAPER", "signal": asdict(signal), "risk": asdict(risk), "fill": {**asdict(fill), "side": fill.side.value, "mode": fill.mode.value, "timestamp": fill.timestamp.isoformat()}, "account": asdict(account)}

    def run_cycle(self, now: datetime | None = None) -> dict:
        now = now or datetime.now(timezone.utc)
        rows = []
        for symbol in self.config.symbols:
            try:
                rows.append(self.run_symbol(symbol, now=now))
            except Exception as exc:
                rows.append({"symbol": symbol, "status": "ERROR", "error": f"{type(exc).__name__}: {exc}"})
        return {"research_status": "V14_FORWARD_PAPER_RUNNING_NOT_LIVE", "strategy_version": STRATEGY_VERSION, "timestamp": now.isoformat(), "paper_execution_enabled": self.paper_execution_enabled, "live_execution": False, "results": rows, "store": self.store.summary()}
