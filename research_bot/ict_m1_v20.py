"""Causal ICT/M1 strategy extraction and research-only trade simulator.

The implementation converts the M1Trades teaching sequence into falsifiable
rules without claiming that the source material proves profitability:

HTF location -> closed-bar liquidity sweep -> close-confirmed structure break
-> return to an origin candidate -> structural stop -> fixed-R target.

Every signal field is observable at its row close.  Trade outcome columns are
future audit data and are never fed back into signal construction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .ichimoku_advanced import add_ichimoku_state


OriginVariant = Literal["sweep_origin", "opposing_candle", "fvg"]
IchimokuGate = Literal["off", "trend"]
ORIGIN_VARIANTS: tuple[OriginVariant, ...] = ("sweep_origin", "opposing_candle", "fvg")
REPORTING_PERIODS = (
    ("development", "2020-01-01 00:00:00+00:00", "2023-12-31 23:59:59+00:00"),
    ("validation", "2024-01-01 00:00:00+00:00", "2024-12-31 23:59:59+00:00"),
    ("final_test", "2025-01-01 00:00:00+00:00", "2025-12-31 23:59:59+00:00"),
)


@dataclass(frozen=True)
class IctM1Config:
    """Frozen v0.20 hypothesis contract; rates are fractions unless named bps."""

    timeframe: str = "4h"
    atr_window: int = 14
    min_displacement_atr: float = 0.50
    max_confirmation_bars: int = 12
    pending_bars: int = 12
    max_holding_bars: int = 36
    origin_entry_fraction: float = 0.50
    stop_buffer_atr: float = 0.05
    target_rr: float = 3.0
    breakeven_trigger_rr: float = 2.0
    risk_per_trade: float = 0.0025
    fee_bps: float = 10.0
    slippage_bps: float = 2.0
    ichimoku_gate: IchimokuGate = "trend"

    def __post_init__(self) -> None:
        if self.atr_window < 2:
            raise ValueError("atr_window must be >= 2")
        if self.max_confirmation_bars < 1 or self.pending_bars < 1 or self.max_holding_bars < 1:
            raise ValueError("bar horizons must be >= 1")
        if not 0.0 <= self.origin_entry_fraction <= 1.0:
            raise ValueError("origin_entry_fraction must be in [0, 1]")
        if self.target_rr <= 0 or self.breakeven_trigger_rr <= 0:
            raise ValueError("R multiples must be positive")
        if not 0 < self.risk_per_trade <= 0.01:
            raise ValueError("risk_per_trade must be in (0, 0.01]")
        if self.fee_bps < 0 or self.slippage_bps < 0:
            raise ValueError("costs cannot be negative")
        if self.ichimoku_gate not in {"off", "trend"}:
            raise ValueError("ichimoku_gate must be 'off' or 'trend'")


def _validate_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    if missing := required - set(frame.columns):
        raise ValueError(f"missing OHLCV columns: {sorted(missing)}")
    x = frame.copy()
    x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True, errors="coerce")
    for column in ("open", "high", "low", "close", "volume"):
        x[column] = pd.to_numeric(x[column], errors="coerce")
    x = x.dropna(subset=list(required)).sort_values("timestamp")
    x = x.drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    invalid = (x[["open", "high", "low", "close"]] <= 0).any(axis=1)
    invalid |= x["high"] < x[["open", "close", "low"]].max(axis=1)
    invalid |= x["low"] > x[["open", "close", "high"]].min(axis=1)
    invalid |= x["volume"] < 0
    if invalid.any():
        raise ValueError(f"invalid OHLC rows: {int(invalid.sum())}")
    if x.empty:
        raise ValueError("no valid OHLCV rows")
    return x


def _atr(frame: pd.DataFrame, window: int) -> pd.Series:
    previous_close = frame["close"].shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1.0 / window, adjust=False).mean()


def build_ict_m1_features(
    frame: pd.DataFrame,
    config: IctM1Config | None = None,
) -> pd.DataFrame:
    """Build closed-bar features with delayed, causal swing availability.

    A pivot at ``t-1`` is first confirmed by the close of ``t``.  It becomes a
    sweep reference starting at ``t+1`` so the sweep bar cannot create the
    pivot it claims to raid.
    """

    cfg = config or IctM1Config()
    x = add_ichimoku_state(_validate_ohlcv(frame))
    x["atr"] = _atr(x, cfg.atr_window)
    x["kijun_slope_3"] = x["ichi_kijun"].diff(3)

    confirmed_high = x["high"].shift(1).where(
        (x["high"].shift(1) > x["high"].shift(2))
        & (x["high"].shift(1) >= x["high"])
    )
    confirmed_low = x["low"].shift(1).where(
        (x["low"].shift(1) < x["low"].shift(2))
        & (x["low"].shift(1) <= x["low"])
    )
    x["confirmed_swing_high"] = confirmed_high
    x["confirmed_swing_low"] = confirmed_low
    x["last_confirmed_swing_high"] = confirmed_high.ffill().shift(1)
    x["last_confirmed_swing_low"] = confirmed_low.ffill().shift(1)

    x["bullish_sweep"] = (
        (x["low"] < x["last_confirmed_swing_low"])
        & (x["close"] > x["last_confirmed_swing_low"])
    ).fillna(False)
    x["bearish_sweep"] = (
        (x["high"] > x["last_confirmed_swing_high"])
        & (x["close"] < x["last_confirmed_swing_high"])
    ).fillna(False)
    x["bullish_fvg_low"] = x["high"].shift(2).where(x["low"] > x["high"].shift(2))
    x["bullish_fvg_high"] = x["low"].where(x["low"] > x["high"].shift(2))
    x["bearish_fvg_low"] = x["high"].where(x["high"] < x["low"].shift(2))
    x["bearish_fvg_high"] = x["low"].shift(2).where(x["high"] < x["low"].shift(2))
    return x.replace([np.inf, -np.inf], np.nan)


def _trend_gate(features: pd.DataFrame, i: int, direction: str, cfg: IctM1Config) -> bool:
    if cfg.ichimoku_gate == "off":
        return True
    row = features.iloc[i]
    if direction == "LONG":
        return bool(
            row["close"] > row["ichi_cloud_top_now"]
            and row["ichi_tenkan"] > row["ichi_kijun"]
            and row["kijun_slope_3"] >= 0
        )
    return bool(
        row["close"] < row["ichi_cloud_bottom_now"]
        and row["ichi_tenkan"] < row["ichi_kijun"]
        and row["kijun_slope_3"] <= 0
    )


def _body_zone(row: pd.Series) -> tuple[float, float]:
    return float(min(row["open"], row["close"])), float(max(row["open"], row["close"]))


def _opposing_zone(features: pd.DataFrame, start: int, end: int, direction: str) -> tuple[float, float]:
    window = features.iloc[start:end]
    if direction == "LONG":
        candidates = window[window["close"] < window["open"]]
    else:
        candidates = window[window["close"] > window["open"]]
    if candidates.empty:
        return np.nan, np.nan
    return _body_zone(candidates.iloc[-1])


def _latest_fvg_zone(features: pd.DataFrame, start: int, end: int, direction: str) -> tuple[float, float]:
    window = features.iloc[start : end + 1]
    prefix = "bullish" if direction == "LONG" else "bearish"
    low_column, high_column = f"{prefix}_fvg_low", f"{prefix}_fvg_high"
    candidates = window.dropna(subset=[low_column, high_column])
    if candidates.empty:
        return np.nan, np.nan
    row = candidates.iloc[-1]
    return float(row[low_column]), float(row[high_column])


def _setup_row(
    features: pd.DataFrame,
    candidate: dict,
    confirm_i: int,
    direction: str,
    cfg: IctM1Config,
) -> dict:
    sweep_i = int(candidate["sweep_index"])
    sweep = features.iloc[sweep_i]
    sweep_zone = _body_zone(sweep)
    opposing_zone = _opposing_zone(features, sweep_i, confirm_i, direction)
    fvg_zone = _latest_fvg_zone(features, sweep_i, confirm_i, direction)
    structural_stop = (
        float(sweep["low"] - cfg.stop_buffer_atr * sweep["atr"])
        if direction == "LONG"
        else float(sweep["high"] + cfg.stop_buffer_atr * sweep["atr"])
    )
    return {
        "setup_id": (
            f"{direction}|{pd.Timestamp(sweep['timestamp']).isoformat()}|"
            f"{pd.Timestamp(features.iloc[confirm_i]['timestamp']).isoformat()}"
        ),
        "direction": direction,
        "sweep_index": sweep_i,
        "confirm_index": int(confirm_i),
        "sweep_time": sweep["timestamp"],
        "confirm_time": features.iloc[confirm_i]["timestamp"],
        "swept_level": float(candidate["swept_level"]),
        "break_level": float(candidate["break_level"]),
        "structural_stop": structural_stop,
        "sweep_origin_low": sweep_zone[0],
        "sweep_origin_high": sweep_zone[1],
        "opposing_candle_low": opposing_zone[0],
        "opposing_candle_high": opposing_zone[1],
        "fvg_low": fvg_zone[0],
        "fvg_high": fvg_zone[1],
        "confirmation_displacement_atr": float(
            abs(features.iloc[confirm_i]["close"] - features.iloc[confirm_i]["open"])
            / features.iloc[confirm_i]["atr"]
        ),
        "ichimoku_gate": cfg.ichimoku_gate,
    }


def extract_ict_m1_setups(
    features: pd.DataFrame,
    config: IctM1Config | None = None,
) -> pd.DataFrame:
    """Extract sweep-to-MSB setups using a two-sided causal state machine."""

    cfg = config or IctM1Config()
    required = {
        "timestamp", "open", "high", "low", "close", "atr",
        "last_confirmed_swing_high", "last_confirmed_swing_low",
        "bullish_sweep", "bearish_sweep",
    }
    if missing := required - set(features.columns):
        raise ValueError(f"missing ICT/M1 feature columns: {sorted(missing)}")

    active: dict[str, dict | None] = {"LONG": None, "SHORT": None}
    rows: list[dict] = []
    for i in range(len(features)):
        row = features.iloc[i]
        for direction in ("LONG", "SHORT"):
            candidate = active[direction]
            if candidate is None:
                continue
            age = i - int(candidate["sweep_index"])
            invalidated = (
                direction == "LONG" and row["close"] <= candidate["sweep_extreme"]
            ) or (
                direction == "SHORT" and row["close"] >= candidate["sweep_extreme"]
            )
            if invalidated or age > cfg.max_confirmation_bars:
                active[direction] = None
                continue
            if age < 1:
                continue
            displacement_atr = abs(float(row["close"] - row["open"])) / float(row["atr"])
            broke = (
                direction == "LONG" and row["close"] > candidate["break_level"] and row["close"] > row["open"]
            ) or (
                direction == "SHORT" and row["close"] < candidate["break_level"] and row["close"] < row["open"]
            )
            if broke and displacement_atr >= cfg.min_displacement_atr and _trend_gate(features, i, direction, cfg):
                rows.append(_setup_row(features, candidate, i, direction, cfg))
                active[direction] = None

        # New sweeps are recorded only after older candidates have been tested,
        # so one outside bar cannot be both the sweep and its confirmation.
        if bool(row["bullish_sweep"]) and np.isfinite(row["last_confirmed_swing_high"]):
            active["LONG"] = {
                "sweep_index": i,
                "swept_level": float(row["last_confirmed_swing_low"]),
                "break_level": float(row["last_confirmed_swing_high"]),
                "sweep_extreme": float(row["low"]),
            }
        if bool(row["bearish_sweep"]) and np.isfinite(row["last_confirmed_swing_low"]):
            active["SHORT"] = {
                "sweep_index": i,
                "swept_level": float(row["last_confirmed_swing_high"]),
                "break_level": float(row["last_confirmed_swing_low"]),
                "sweep_extreme": float(row["high"]),
            }

    columns = [
        "setup_id", "direction", "sweep_index", "confirm_index", "sweep_time", "confirm_time",
        "swept_level", "break_level", "structural_stop",
        "sweep_origin_low", "sweep_origin_high",
        "opposing_candle_low", "opposing_candle_high", "fvg_low", "fvg_high",
        "confirmation_displacement_atr", "ichimoku_gate",
    ]
    return pd.DataFrame(rows, columns=columns)


def _variant_zone(setup: pd.Series, variant: OriginVariant) -> tuple[float, float]:
    if variant == "sweep_origin":
        return float(setup["sweep_origin_low"]), float(setup["sweep_origin_high"])
    if variant == "opposing_candle":
        return float(setup["opposing_candle_low"]), float(setup["opposing_candle_high"])
    if variant == "fvg":
        return float(setup["fvg_low"]), float(setup["fvg_high"])
    raise ValueError(f"unsupported origin variant: {variant}")


def simulate_ict_m1_trades(
    features: pd.DataFrame,
    setups: pd.DataFrame,
    config: IctM1Config | None = None,
    *,
    variant: OriginVariant = "sweep_origin",
) -> pd.DataFrame:
    """Simulate non-overlapping pending-limit trades with conservative OHLC ordering."""

    cfg = config or IctM1Config()
    if variant not in ORIGIN_VARIANTS:
        raise ValueError(f"unsupported origin variant: {variant}")
    if setups.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    busy_until = -1
    fee_rate = cfg.fee_bps / 10_000.0
    slip_rate = cfg.slippage_bps / 10_000.0
    for _, setup in setups.sort_values(["confirm_index", "setup_id"]).iterrows():
        confirm_i = int(setup["confirm_index"])
        if confirm_i <= busy_until:
            continue
        zone_low, zone_high = _variant_zone(setup, variant)
        if not (np.isfinite(zone_low) and np.isfinite(zone_high) and zone_high >= zone_low):
            continue
        raw_entry = zone_low + cfg.origin_entry_fraction * (zone_high - zone_low)
        direction = str(setup["direction"])
        fill_i = None
        pending_end = min(len(features) - 1, confirm_i + cfg.pending_bars)
        for i in range(confirm_i + 1, pending_end + 1):
            bar = features.iloc[i]
            if float(bar["low"]) <= raw_entry <= float(bar["high"]):
                fill_i = i
                break
        if fill_i is None:
            continue

        execution_entry = raw_entry * (1.0 + slip_rate if direction == "LONG" else 1.0 - slip_rate)
        structural_stop = float(setup["structural_stop"])
        risk_distance = execution_entry - structural_stop if direction == "LONG" else structural_stop - execution_entry
        if not np.isfinite(risk_distance) or risk_distance <= 0:
            continue
        target = (
            execution_entry + cfg.target_rr * risk_distance
            if direction == "LONG"
            else execution_entry - cfg.target_rr * risk_distance
        )
        breakeven_trigger = (
            execution_entry + cfg.breakeven_trigger_rr * risk_distance
            if direction == "LONG"
            else execution_entry - cfg.breakeven_trigger_rr * risk_distance
        )
        stop = structural_stop
        breakeven_active = False
        outcome = "TIME"
        raw_exit = float(features.iloc[min(len(features) - 1, fill_i + cfg.max_holding_bars)]["close"])
        exit_i = min(len(features) - 1, fill_i + cfg.max_holding_bars)
        for i in range(fill_i, exit_i + 1):
            bar = features.iloc[i]
            hit_stop = float(bar["low"]) <= stop if direction == "LONG" else float(bar["high"]) >= stop
            hit_target = float(bar["high"]) >= target if direction == "LONG" else float(bar["low"]) <= target
            if hit_stop:  # stop-first resolves unknown same-bar path conservatively
                raw_exit, exit_i = stop, i
                outcome = "BREAKEVEN" if breakeven_active else "STOP"
                break
            if hit_target:
                raw_exit, exit_i, outcome = target, i, "TARGET"
                break
            reached_breakeven_trigger = (
                float(bar["high"]) >= breakeven_trigger
                if direction == "LONG"
                else float(bar["low"]) <= breakeven_trigger
            )
            if reached_breakeven_trigger:
                # Activated for the next bar; same-bar path remains unknowable.
                stop = execution_entry
                breakeven_active = True

        execution_exit = raw_exit * (1.0 - slip_rate if direction == "LONG" else 1.0 + slip_rate)
        gross_pnl = execution_exit - execution_entry if direction == "LONG" else execution_entry - execution_exit
        fees = fee_rate * (execution_entry + execution_exit)
        net_pnl = gross_pnl - fees
        net_r = net_pnl / risk_distance
        rows.append(
            {
                "setup_id": setup["setup_id"],
                "variant": variant,
                "direction": direction,
                "confirm_index": confirm_i,
                "fill_index": int(fill_i),
                "exit_index": int(exit_i),
                "confirm_time": setup["confirm_time"],
                "fill_time": features.iloc[fill_i]["timestamp"],
                "exit_time": features.iloc[exit_i]["timestamp"],
                "entry_price": float(execution_entry),
                "initial_stop": structural_stop,
                "target_price": float(target),
                "exit_price": float(execution_exit),
                "outcome": outcome,
                "net_r": float(net_r),
                "account_return": float(cfg.risk_per_trade * net_r),
                "risk_fraction": cfg.risk_per_trade,
                "fee_bps": cfg.fee_bps,
                "slippage_bps": cfg.slippage_bps,
                "paper_only": True,
                "live_execution": False,
            }
        )
        busy_until = int(exit_i)
    return pd.DataFrame(rows)


def summarize_ict_m1_trades(variant: str, setups: pd.DataFrame, trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {
            "variant": variant,
            "status": "NO_TRADES",
            "setups": int(len(setups)),
            "trades": 0,
            "win_rate": np.nan,
            "mean_net_r": np.nan,
            "profit_factor_r": np.nan,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "live_execution": False,
        }
    returns = trades["account_return"].astype(float)
    equity = (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    positive = trades.loc[trades["net_r"] > 0, "net_r"].sum()
    negative = -trades.loc[trades["net_r"] < 0, "net_r"].sum()
    return {
        "variant": variant,
        "status": "EVALUATED",
        "setups": int(len(setups)),
        "trades": int(len(trades)),
        "win_rate": float((trades["net_r"] > 0).mean()),
        "mean_net_r": float(trades["net_r"].mean()),
        "profit_factor_r": float(positive / negative) if negative > 0 else (np.inf if positive > 0 else np.nan),
        "total_return": float(equity.iloc[-1] - 1.0),
        "max_drawdown": float(drawdown.min()),
        "live_execution": False,
    }


def evaluate_ict_m1_variants(
    frame: pd.DataFrame,
    config: IctM1Config | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the frozen origin-definition ablation under identical assumptions."""

    cfg = config or IctM1Config()
    features = build_ict_m1_features(frame, cfg)
    setups = extract_ict_m1_setups(features, cfg)
    summaries: list[dict] = []
    trade_tables: list[pd.DataFrame] = []
    for variant in ORIGIN_VARIANTS:
        trades = simulate_ict_m1_trades(features, setups, cfg, variant=variant)
        summaries.append(summarize_ict_m1_trades(variant, setups, trades))
        if not trades.empty:
            trade_tables.append(trades)
    all_trades = pd.concat(trade_tables, ignore_index=True) if trade_tables else pd.DataFrame()
    summary = pd.DataFrame(summaries)
    summary.attrs["config"] = asdict(cfg)
    return summary, all_trades, setups
