from __future__ import annotations

"""v0.30 preregistered regime/event-aware alpha candidates.

Motivation
----------
v0.27 found no candidate from the frozen 42-strategy registry that survived the
CoinEx + OKX development robustness screen. v0.30 therefore introduces new
hypotheses rather than retuning the rejected registry.

Scientific contract
-------------------
- Development evidence is limited to already-consumed CoinEx + OKX.
- KuCoin remains the untouched final venue holdout and must never participate
  in feature construction, parameter choice or winner selection.
- Candidate count and parameters are fixed before the tournament: 12 total,
  restricted to 4h and 1d to target slower, state-dependent structure.
- v0.25 portfolio allocation and the 24 bps round-trip cost assumption remain
  frozen. No live execution is authorized by this module.
"""

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from research_bot.multitimeframe_strategies_v19 import StrategySpec, build_features


DEVELOPMENT_VENUES_V30 = ("coinex_consumed", "okx_consumed")
FINAL_HOLDOUT_VENUE_V30 = "kucoin"
ROUND_TRIP_COST_FRACTION_V30 = 0.0024
PRIOR_EFFECTIVE_TRIALS = 84
NEW_TRIALS_V30 = 12
TOTAL_EFFECTIVE_TRIALS_V30 = PRIOR_EFFECTIVE_TRIALS + NEW_TRIALS_V30

REGIME_WINDOW = {"4h": 42, "1d": 28}
DISPERSION_LOOKBACK = {"4h": 126, "1d": 90}


@dataclass(frozen=True)
class V30Candidate:
    name: str
    timeframe: str
    family: str
    source_basis: str
    rr: float
    stop_atr: float
    max_hold_bars: int
    fast_horizon: int | None = None
    slow_horizon: int | None = None
    cusum_threshold: float | None = None
    breakout_lookback: int = 20
    volume_multiplier: float = 1.0

    def strategy_spec(self) -> StrategySpec:
        return StrategySpec(
            self.name,
            self.timeframe,
            f"v30_{self.family}",
            self.source_basis,
            rr=self.rr,
            stop_atr=self.stop_atr,
            max_hold_bars=self.max_hold_bars,
            long_short=True,
        )


V30_CANDIDATES: tuple[V30Candidate, ...] = (
    V30Candidate("V30_H4_REGIME_MOM_20_60", "4h", "regime_momentum", "State-dependent crypto momentum + persistent market regime", 2.5, 1.5, 30, fast_horizon=20, slow_horizon=60),
    V30Candidate("V30_H4_REGIME_MOM_30_90", "4h", "regime_momentum", "State-dependent crypto momentum + persistent market regime", 2.5, 1.5, 30, fast_horizon=30, slow_horizon=90),
    V30Candidate("V30_D1_REGIME_MOM_20_90", "1d", "regime_momentum", "State-dependent crypto momentum + persistent market regime", 2.5, 1.7, 30, fast_horizon=20, slow_horizon=90),
    V30Candidate("V30_D1_REGIME_MOM_30_120", "1d", "regime_momentum", "State-dependent crypto momentum + persistent market regime", 2.5, 1.7, 35, fast_horizon=30, slow_horizon=120),
    V30Candidate("V30_H4_CUSUM_BREAKOUT_3", "4h", "cusum_breakout", "CUSUM information event + breakout + volume confirmation", 3.0, 1.5, 30, cusum_threshold=3.0, breakout_lookback=20, volume_multiplier=1.10),
    V30Candidate("V30_H4_CUSUM_BREAKOUT_4", "4h", "cusum_breakout", "CUSUM information event + breakout + volume confirmation", 3.0, 1.5, 30, cusum_threshold=4.0, breakout_lookback=40, volume_multiplier=1.20),
    V30Candidate("V30_D1_CUSUM_BREAKOUT_3", "1d", "cusum_breakout", "CUSUM information event + breakout + volume confirmation", 3.0, 1.7, 25, cusum_threshold=3.0, breakout_lookback=20, volume_multiplier=1.10),
    V30Candidate("V30_D1_CUSUM_BREAKOUT_4", "1d", "cusum_breakout", "CUSUM information event + breakout + volume confirmation", 3.0, 1.7, 30, cusum_threshold=4.0, breakout_lookback=40, volume_multiplier=1.20),
    V30Candidate("V30_H4_ICHIMOKU_REGIME_100", "4h", "ichimoku_regime", "Causal Ichimoku breakout gated by persistent regime and dispersion", 3.0, 1.5, 30, breakout_lookback=20, volume_multiplier=1.00),
    V30Candidate("V30_H4_ICHIMOKU_REGIME_125", "4h", "ichimoku_regime", "Causal Ichimoku breakout gated by persistent regime and dispersion", 3.0, 1.5, 30, breakout_lookback=20, volume_multiplier=1.25),
    V30Candidate("V30_D1_ICHIMOKU_REGIME_100", "1d", "ichimoku_regime", "Causal Ichimoku breakout gated by persistent regime and dispersion", 3.0, 1.7, 25, breakout_lookback=20, volume_multiplier=1.00),
    V30Candidate("V30_D1_ICHIMOKU_REGIME_125", "1d", "ichimoku_regime", "Causal Ichimoku breakout gated by persistent regime and dispersion", 3.0, 1.7, 25, breakout_lookback=20, volume_multiplier=1.25),
)


def registry_frame_v30() -> pd.DataFrame:
    return pd.DataFrame([asdict(x) for x in V30_CANDIDATES])


def preregistration_manifest_v30() -> dict[str, Any]:
    return {
        "version": "v0.30",
        "experiment": "PREREGISTERED_REGIME_EVENT_ALPHA_GENERATION",
        "candidate_count": len(V30_CANDIDATES),
        "candidate_names": [x.name for x in V30_CANDIDATES],
        "timeframes": ["4h", "1d"],
        "families": ["regime_momentum", "cusum_breakout", "ichimoku_regime"],
        "development_venues": list(DEVELOPMENT_VENUES_V30),
        "reserved_holdout_venue": FINAL_HOLDOUT_VENUE_V30,
        "prior_effective_trials": PRIOR_EFFECTIVE_TRIALS,
        "new_trials": NEW_TRIALS_V30,
        "total_effective_trials": TOTAL_EFFECTIVE_TRIALS_V30,
        "round_trip_cost_fraction": ROUND_TRIP_COST_FRACTION_V30,
        "aggregate_open_risk_cap": 0.02,
        "directional_open_risk_cap": 0.015,
        "max_drawdown": 0.05,
        "min_profit_factor": 1.05,
        "min_positive_asset_fraction": 0.60,
        "min_development_trades_per_venue": 200,
        "threshold_relaxation": False,
        "strategy_parameter_retuning": False,
        "holdout_inspection_before_lock": False,
        "live_execution_authorized": False,
    }


def persistent_market_regime(market_frame: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Classify persistent UP-UP / DOWN-DOWN market state causally."""
    if timeframe not in REGIME_WINDOW:
        raise ValueError(f"unsupported v0.30 timeframe: {timeframe}")
    f = build_features(market_frame)[["timestamp", "close"]].copy()
    w = REGIME_WINDOW[timeframe]
    recent = f["close"] / f["close"].shift(w) - 1.0
    prior = f["close"].shift(w) / f["close"].shift(2 * w) - 1.0
    state = pd.Series(0, index=f.index, dtype="int8")
    state.loc[(recent > 0) & (prior > 0)] = 1
    state.loc[(recent < 0) & (prior < 0)] = -1
    return pd.DataFrame({"timestamp": f["timestamp"], "market_regime_v30": state})


def cross_sectional_dispersion(frames: dict[str, pd.DataFrame], timeframe: str) -> pd.DataFrame:
    """Compute causal cross-sectional return dispersion and a lagged stress cap."""
    if timeframe not in DISPERSION_LOOKBACK:
        raise ValueError(f"unsupported v0.30 timeframe: {timeframe}")
    pieces: list[pd.DataFrame] = []
    for symbol, frame in sorted(frames.items()):
        f = build_features(frame)[["timestamp", "ret1"]].copy()
        f["symbol"] = symbol
        pieces.append(f)
    if not pieces:
        return pd.DataFrame(columns=["timestamp", "dispersion_v30", "dispersion_cap_v30"])
    long = pd.concat(pieces, ignore_index=True)
    wide = long.pivot_table(index="timestamp", columns="symbol", values="ret1", aggfunc="last").sort_index()
    dispersion = wide.std(axis=1, ddof=0)
    lookback = DISPERSION_LOOKBACK[timeframe]
    cap = dispersion.rolling(lookback, min_periods=max(20, lookback // 3)).quantile(0.80).shift(1)
    return pd.DataFrame({
        "timestamp": dispersion.index,
        "dispersion_v30": dispersion.to_numpy(),
        "dispersion_cap_v30": cap.to_numpy(),
    }).reset_index(drop=True)


def _attach_context(
    frame: pd.DataFrame,
    *,
    timeframe: str,
    market_frame: pd.DataFrame,
    dispersion: pd.DataFrame | None,
) -> pd.DataFrame:
    f = build_features(frame).sort_values("timestamp").reset_index(drop=True)
    regime = persistent_market_regime(market_frame, timeframe).sort_values("timestamp")
    f = pd.merge_asof(f, regime, on="timestamp", direction="backward", allow_exact_matches=True)
    if dispersion is not None and not dispersion.empty:
        d = dispersion.copy().sort_values("timestamp")
        d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
        f = pd.merge_asof(f.sort_values("timestamp"), d, on="timestamp", direction="backward", allow_exact_matches=True)
    else:
        f["dispersion_v30"] = np.nan
        f["dispersion_cap_v30"] = np.nan
    f["low_dispersion_v30"] = (
        f["dispersion_cap_v30"].isna() | (f["dispersion_v30"] <= f["dispersion_cap_v30"])
    )
    return f


def _volume_confirm(f: pd.DataFrame, multiplier: float) -> pd.Series:
    baseline = f["volume"].rolling(50, min_periods=20).median().shift(1)
    return (f["volume"] >= multiplier * baseline).fillna(False)


def _prior_extreme(series: pd.Series, lookback: int, *, kind: str) -> pd.Series:
    shifted = series.shift(1)
    if kind == "max":
        return shifted.rolling(lookback, min_periods=lookback).max()
    return shifted.rolling(lookback, min_periods=lookback).min()


def _standardized_cusum(f: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    ret = f["close"].pct_change()
    scale = ret.rolling(100, min_periods=30).std(ddof=0).shift(1).replace(0, np.nan)
    z = (ret / scale).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-8.0, 8.0)
    pos = np.zeros(len(f), dtype=float)
    neg = np.zeros(len(f), dtype=float)
    for i in range(1, len(f)):
        value = float(z.iloc[i])
        pos[i] = max(0.0, pos[i - 1] + value)
        neg[i] = min(0.0, neg[i - 1] + value)
    return pd.Series(pos, index=f.index), pd.Series(neg, index=f.index)


def generate_direction_v30(
    candidate: V30Candidate,
    frame: pd.DataFrame,
    *,
    market_frame: pd.DataFrame,
    dispersion: pd.DataFrame | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """Generate one preregistered v0.30 signal without future information."""
    f = _attach_context(
        frame,
        timeframe=candidate.timeframe,
        market_frame=market_frame,
        dispersion=dispersion,
    )
    regime_long = f["market_regime_v30"].fillna(0).eq(1)
    regime_short = f["market_regime_v30"].fillna(0).eq(-1)
    dispersion_ok = f["low_dispersion_v30"].fillna(False)
    volume_ok = _volume_confirm(f, candidate.volume_multiplier)
    high_break = _prior_extreme(f["high"], candidate.breakout_lookback, kind="max")
    low_break = _prior_extreme(f["low"], candidate.breakout_lookback, kind="min")

    if candidate.family == "regime_momentum":
        if candidate.fast_horizon is None or candidate.slow_horizon is None:
            raise ValueError("regime_momentum requires fast and slow horizons")
        fast = f["close"].pct_change(candidate.fast_horizon)
        slow = f["close"].pct_change(candidate.slow_horizon)
        long = (fast > 0) & (slow > 0) & regime_long & dispersion_ok
        short = (fast < 0) & (slow < 0) & regime_short & dispersion_ok
    elif candidate.family == "cusum_breakout":
        if candidate.cusum_threshold is None:
            raise ValueError("cusum_breakout requires cusum threshold")
        pos, neg = _standardized_cusum(f)
        long = (pos >= candidate.cusum_threshold) & (f["close"] > high_break) & volume_ok & regime_long & dispersion_ok
        short = (neg <= -candidate.cusum_threshold) & (f["close"] < low_break) & volume_ok & regime_short & dispersion_ok
        f["cusum_positive_v30"] = pos
        f["cusum_negative_v30"] = neg
    elif candidate.family == "ichimoku_regime":
        long = (
            (f["close"] > f["cloud_top"])
            & (f["tenkan"] > f["kijun"])
            & (f["close"] > high_break)
            & volume_ok
            & regime_long
            & dispersion_ok
        )
        short = (
            (f["close"] < f["cloud_bottom"])
            & (f["tenkan"] < f["kijun"])
            & (f["close"] < low_break)
            & volume_ok
            & regime_short
            & dispersion_ok
        )
    else:
        raise ValueError(f"unsupported v0.30 family: {candidate.family}")

    direction = pd.Series(0, index=f.index, dtype="int8")
    direction.loc[long.fillna(False)] = 1
    direction.loc[short.fillna(False)] = -1
    direction.loc[long.fillna(False) & short.fillna(False)] = 0
    return direction, f
