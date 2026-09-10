from __future__ import annotations

from dataclasses import asdict, dataclass
from math import erf, sqrt
from typing import Iterable

import numpy as np
import pandas as pd

from research_bot.multitimeframe_strategies_v19 import (
    STRATEGY_REGISTRY as V19_REGISTRY,
    StrategySpec,
    TournamentConfig,
    build_features,
    generate_direction as generate_direction_v19,
    moving_block_mean_ci,
    simulate_bracket_trades,
    summarize_trades,
)


@dataclass(frozen=True)
class RiskPsychologyPolicy:
    """Frozen capital-preservation/anti-discretion rules for v0.20."""
    base_risk_per_trade: float = 0.0025
    max_risk_per_trade: float = 0.0050
    vol_floor_scale: float = 0.35
    drawdown_warn_1: float = 0.020
    drawdown_warn_2: float = 0.035
    hard_drawdown_kill: float = 0.050
    drawdown_scale_1: float = 0.75
    drawdown_scale_2: float = 0.50
    max_consecutive_losses: int = 3
    max_daily_trades_intraday: int = 4


@dataclass(frozen=True)
class V20ValidationConfig:
    min_pretest_trades: int = 1000
    min_validation_trades: int = 200
    min_external_trades: int = 200
    min_profit_factor: float = 1.05
    min_positive_asset_fraction: float = 0.60
    max_drawdown: float = 0.05
    max_multiplicity_p: float = 0.10
    bootstrap_samples: int = 750
    bootstrap_block: int = 20
    random_seed: int = 314


HTF_MAP = {
    "1m": ("15min", pd.Timedelta(minutes=15)),
    "5m": ("1h", pd.Timedelta(hours=1)),
    "15m": ("4h", pd.Timedelta(hours=4)),
    "1h": ("1d", pd.Timedelta(days=1)),
    "4h": ("1d", pd.Timedelta(days=1)),
    "1d": ("7d", pd.Timedelta(days=7)),
}
COOLDOWN = {
    "1m": pd.Timedelta(hours=1), "5m": pd.Timedelta(hours=3),
    "15m": pd.Timedelta(hours=6), "1h": pd.Timedelta(hours=12),
    "4h": pd.Timedelta(days=2), "1d": pd.Timedelta(days=7),
}

# Two genuinely new, completed-HTF/risk-aware variants per timeframe. The 30
# v0.19 rules remain frozen baselines so the search history stays auditable.
NEW_SPECS: tuple[StrategySpec, ...] = (
    StrategySpec("M1_MTF_ATM_RISK", "1m", "v20_mtf_atm", "ICT/TTrades + M1Trades: HTF bias -> sweep/MSB -> Origin RTO", rr=3.0, stop_atr=1.1, max_hold_bars=60),
    StrategySpec("M1_MTF_UNICORN_RISK", "1m", "v20_mtf_unicorn", "ICT Unicorn: HTF bias + liquidity sweep + MSS/BOS + FVG/Breaker overlap", rr=3.0, stop_atr=1.1, max_hold_bars=60),
    StrategySpec("M5_MTF_ATM_RISK", "5m", "v20_mtf_atm", "ICT/TTrades fractal mapping: 1H context -> 5m execution", rr=3.0, stop_atr=1.1, max_hold_bars=48),
    StrategySpec("M5_OSOK_CISD_RISK", "5m", "v20_osok_cisd", "TTrades OSOK: HTF swing/narrative -> LTF CISD", rr=2.5, stop_atr=1.1, max_hold_bars=48),
    StrategySpec("M15_MTF_SILVER_BULLET_RISK", "15m", "v20_mtf_silver", "ICT/TTrades Silver Bullet: timed liquidity event + FVG with HTF context", rr=3.0, stop_atr=1.2, max_hold_bars=40),
    StrategySpec("M15_MTF_SUPPLY_DEMAND_RISK", "15m", "v20_mtf_supply_demand", "Supply/Demand: 4H/1H analysis -> 15m entry", rr=3.0, stop_atr=1.2, max_hold_bars=40),
    StrategySpec("H1_D1_OB_BOS_RISK", "1h", "v20_mtf_ob", "Order Blocks + ICT: Daily context -> 1H BOS/OB retest", rr=3.0, stop_atr=1.3, max_hold_bars=36),
    StrategySpec("H1_D1_ICHIMOKU_RISK", "1h", "v20_mtf_ichimoku", "Causal Ichimoku pullback aligned to completed Daily context", rr=3.0, stop_atr=1.3, max_hold_bars=36),
    StrategySpec("H4_D1_S6_VOL_RISK", "4h", "v20_mtf_s6", "Project S6 trend breakout + completed Daily bias + volatility-managed sizing", rr=3.0, stop_atr=1.5, max_hold_bars=30),
    StrategySpec("H4_D1_OB_BOS_RISK", "4h", "v20_mtf_ob", "HTF Order Block/BOS retest + completed Daily bias", rr=3.0, stop_atr=1.5, max_hold_bars=30),
    StrategySpec("D1_WEEKLY_ICT_SWING_RISK", "1d", "v20_weekly_ict", "ICT swing PD arrays: completed Weekly context -> Daily entry", rr=3.0, stop_atr=1.7, max_hold_bars=20),
    StrategySpec("D1_WEEKLY_TSMOM_VOL_RISK", "1d", "v20_tsmom", "Time-series momentum + completed Weekly regime + volatility-managed sizing", rr=2.5, stop_atr=1.7, max_hold_bars=30),
)
STRATEGY_REGISTRY_V20 = tuple(V19_REGISTRY) + NEW_SPECS
V19_NAMES = {s.name for s in V19_REGISTRY}


def registry_frame_v20() -> pd.DataFrame:
    rows = []
    for s in STRATEGY_REGISTRY_V20:
        row = asdict(s)
        row["generation"] = "v0.19_baseline" if s.name in V19_NAMES else "v0.20_new"
        rows.append(row)
    return pd.DataFrame(rows)


def _validate(frame: pd.DataFrame) -> pd.DataFrame:
    cols = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = set(cols) - set(frame.columns)
    if missing:
        raise ValueError(f"missing OHLCV columns: {sorted(missing)}")
    x = frame[cols].copy()
    x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True, errors="coerce")
    for c in cols[1:]:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    return x.dropna().drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def _resample_completed(frame: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    x = _validate(frame).set_index("timestamp")
    rule, delay = HTF_MAP[timeframe]
    h = x.resample(rule, label="left", closed="left", origin="epoch").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum"),
    ).dropna().reset_index()
    h["available_time"] = h["timestamp"] + delay
    return h


def attach_completed_htf_context(frame: pd.DataFrame, timeframe: str, peer: pd.DataFrame | None = None) -> pd.DataFrame:
    """Merge higher-timeframe features only after the source HTF candle closed."""
    base = build_features(frame, peer=peer).sort_values("timestamp").reset_index(drop=True)
    raw = _resample_completed(frame, timeframe)
    if raw.empty:
        return base
    h = build_features(raw[["timestamp", "open", "high", "low", "close", "volume"]])
    h["available_time"] = raw["available_time"].to_numpy()
    wanted = ["available_time", "close", "ema50", "ema200", "ema200_slope", "atr_pct", "cloud_top", "cloud_bottom", "tenkan", "kijun", "bull_retracement", "bear_retracement", "sweep_down", "sweep_up", "bos_up", "bos_down"]
    right = h[wanted].rename(columns={c: f"htf_{c}" for c in wanted if c != "available_time"})
    # CoinEx currently materializes millisecond timestamps while pandas resample
    # arithmetic can promote HTF availability to microseconds. Force a common
    # nanosecond UTC dtype before merge_asof so real-data behavior matches tests.
    base["timestamp"] = pd.to_datetime(base["timestamp"], utc=True).astype("datetime64[ns, UTC]")
    right["available_time"] = pd.to_datetime(right["available_time"], utc=True).astype("datetime64[ns, UTC]")
    return pd.merge_asof(base.sort_values("timestamp"), right.sort_values("available_time"), left_on="timestamp", right_on="available_time", direction="backward", allow_exact_matches=True)


def _recent(flag: pd.Series, bars: int) -> pd.Series:
    return flag.fillna(False).astype(float).rolling(bars, min_periods=1).max().gt(0)


def _htf_long(f: pd.DataFrame) -> pd.Series:
    return (f["htf_close"] > f["htf_ema50"]) & (f["htf_ema200_slope"].fillna(0) >= 0)


def _htf_short(f: pd.DataFrame) -> pd.Series:
    return (f["htf_close"] < f["htf_ema50"]) & (f["htf_ema200_slope"].fillna(0) <= 0)


def _no_trade(f: pd.DataFrame) -> pd.Series:
    both = _recent(f["sweep_down"], 12) & _recent(f["sweep_up"], 12)
    neutral = (f["close"] - f["ema50"]).abs() <= 0.60 * f["atr"]
    return both & neutral


def _proxy(spec: StrategySpec, family: str) -> StrategySpec:
    return StrategySpec(spec.name, spec.timeframe, family, spec.source_basis, rr=spec.rr, stop_atr=spec.stop_atr, max_hold_bars=spec.max_hold_bars, long_short=spec.long_short)


def generate_direction_v20(spec: StrategySpec, frame: pd.DataFrame, peer: pd.DataFrame | None = None) -> tuple[pd.Series, pd.DataFrame]:
    if spec.name in V19_NAMES:
        return generate_direction_v19(spec, frame, peer=peer)
    family_map = {
        "v20_mtf_atm": "atm_origin", "v20_mtf_unicorn": "unicorn",
        "v20_osok_cisd": "fractal_cisd", "v20_mtf_silver": "silver_bullet",
        "v20_mtf_supply_demand": "supply_demand", "v20_mtf_ob": "confirmed_ob",
        "v20_mtf_ichimoku": "ichimoku_pullback", "v20_mtf_s6": "s6_breakout",
        "v20_weekly_ict": "ict_swing_pd",
    }
    f = attach_completed_htf_context(frame, spec.timeframe, peer=peer)
    if spec.family == "v20_tsmom":
        fast, slow = f["close"].pct_change(20), f["close"].pct_change(90)
        long = (fast > 0) & (slow > 0) & _htf_long(f) & ~_no_trade(f)
        short = (fast < 0) & (slow < 0) & _htf_short(f) & ~_no_trade(f)
    else:
        d0, _ = generate_direction_v19(_proxy(spec, family_map[spec.family]), frame, peer=peer)
        d0 = pd.Series(d0, index=f.index).fillna(0)
        long = (d0 > 0) & _htf_long(f) & ~_no_trade(f)
        short = (d0 < 0) & _htf_short(f) & ~_no_trade(f)
        if spec.family in {"v20_osok_cisd", "v20_weekly_ict"}:
            long &= f["htf_bull_retracement"].between(0.35, 0.85) | f["htf_bull_retracement"].isna()
            short &= f["htf_bear_retracement"].between(0.35, 0.85) | f["htf_bear_retracement"].isna()
    d = pd.Series(0, index=f.index, dtype="int8")
    d.loc[long.fillna(False)] = 1
    d.loc[short.fillna(False)] = -1
    d.loc[long.fillna(False) & short.fillna(False)] = 0
    return d, f


def add_causal_volatility_scale(ledger: pd.DataFrame, features: pd.DataFrame, policy: RiskPsychologyPolicy | None = None) -> pd.DataFrame:
    p = policy or RiskPsychologyPolicy()
    if ledger.empty:
        return ledger.copy()
    f = features[["timestamp", "atr_pct"]].copy()
    f["vol_target"] = f["atr_pct"].shift(1).rolling(100, min_periods=30).median()
    f["risk_scale_volatility"] = (f["vol_target"] / f["atr_pct"]).clip(p.vol_floor_scale, 1.0).fillna(1.0)
    out = ledger.copy()
    out["signal_time"] = pd.to_datetime(out["signal_time"], utc=True)
    return out.merge(f[["timestamp", "risk_scale_volatility"]], left_on="signal_time", right_on="timestamp", how="left").drop(columns="timestamp")


def apply_risk_psychology_overlay(ledger: pd.DataFrame, spec: StrategySpec, policy: RiskPsychologyPolicy | None = None) -> pd.DataFrame:
    """Translate psychology into deterministic risk governance; rejected attempts stay auditable."""
    p = policy or RiskPsychologyPolicy()
    if ledger.empty:
        return ledger.copy()
    x = ledger.sort_values(["entry_time", "symbol"]).reset_index(drop=True).copy()
    x["entry_time"] = pd.to_datetime(x["entry_time"], utc=True)
    x["exit_time"] = pd.to_datetime(x["exit_time"], utc=True)
    for c, v in {"executed_v20": False, "reject_reason_v20": "", "risk_scale_drawdown": 0.0, "risk_fraction_v20": 0.0, "account_return_v20": 0.0, "equity_v20": np.nan, "drawdown_before_v20": np.nan, "loss_streak_before_v20": 0}.items():
        x[c] = v
    equity = peak = 1.0
    streak, cooldown, hard_killed = 0, None, False
    day_counts: dict[tuple[str, object], int] = {}
    for i, row in x.iterrows():
        t = row["entry_time"]
        dd = equity / peak - 1.0
        x.at[i, "drawdown_before_v20"] = dd
        x.at[i, "loss_streak_before_v20"] = streak
        key = (str(row["symbol"]), t.date())
        reason = ""
        if hard_killed or dd <= -p.hard_drawdown_kill:
            hard_killed, reason = True, "HARD_DRAWDOWN_KILL"
        elif cooldown is not None and t < cooldown:
            reason = "LOSS_STREAK_COOLDOWN"
        elif spec.timeframe in {"1m", "5m", "15m", "1h"} and day_counts.get(key, 0) >= p.max_daily_trades_intraday:
            reason = "OVERTRADING_DAILY_CAP"
        if reason:
            x.at[i, "reject_reason_v20"], x.at[i, "equity_v20"] = reason, equity
            continue
        dd_scale = p.drawdown_scale_2 if dd <= -p.drawdown_warn_2 else (p.drawdown_scale_1 if dd <= -p.drawdown_warn_1 else 1.0)
        vol_scale = float(np.clip(row.get("risk_scale_volatility", 1.0), p.vol_floor_scale, 1.0))
        risk = min(p.max_risk_per_trade, p.base_risk_per_trade * dd_scale * vol_scale)
        ret = risk * float(row["r_multiple"])
        x.at[i, "executed_v20"] = True
        x.at[i, "risk_scale_drawdown"], x.at[i, "risk_fraction_v20"], x.at[i, "account_return_v20"] = dd_scale, risk, ret
        equity *= 1.0 + ret
        peak = max(peak, equity)
        x.at[i, "equity_v20"] = equity
        day_counts[key] = day_counts.get(key, 0) + 1
        if ret < 0:
            streak += 1
            if streak >= p.max_consecutive_losses:
                cooldown, streak = row["exit_time"] + COOLDOWN[spec.timeframe], 0
        elif ret > 0:
            streak = 0
    return x


def simulate_v20_trades(spec: StrategySpec, frame: pd.DataFrame, direction: pd.Series, features: pd.DataFrame, symbol: str, tournament: TournamentConfig | None = None, policy: RiskPsychologyPolicy | None = None) -> pd.DataFrame:
    p = policy or RiskPsychologyPolicy()
    cfg = tournament or TournamentConfig(risk_per_trade=p.base_risk_per_trade)
    return add_causal_volatility_scale(simulate_bracket_trades(spec, frame, direction, features, symbol, cfg), features, p)


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def _multiplicity_adjusted_p(values: Iterable[float], trials: int) -> float:
    a = np.asarray(list(values), dtype=float)
    a = a[np.isfinite(a)]
    if len(a) < 30 or np.std(a, ddof=1) <= 0:
        return 1.0
    z = float(np.mean(a) / (np.std(a, ddof=1) / np.sqrt(len(a))))
    return float(min(1.0, max(0.0, 1.0 - _normal_cdf(z)) * max(1, trials)))


def evaluate_v20_candidate(attempts: pd.DataFrame, *, total_trials: int, validation: V20ValidationConfig | None = None) -> dict:
    c = validation or V20ValidationConfig()
    executed = attempts[attempts["executed_v20"] == True].copy() if (not attempts.empty and "executed_v20" in attempts) else attempts.copy()  # noqa: E712
    if not executed.empty:
        executed["account_return"] = executed["account_return_v20"]
    out: dict[str, object] = {"attempted_trades": int(len(attempts)), "executed_trades": int(len(executed)), "risk_rejected_trades": int(len(attempts) - len(executed))}
    for seg in ("development", "validation", "test"):
        m = summarize_trades(executed[executed["segment"] == seg]) if not executed.empty else summarize_trades(executed)
        out.update({f"{seg}_{k}": v for k, v in m.items()})
    pre_attempts = attempts[attempts["segment"].isin(["development", "validation"])] if (not attempts.empty and "segment" in attempts) else attempts
    pre_exec = executed[executed["segment"].isin(["development", "validation"])] if not executed.empty else executed
    val = executed[executed["segment"] == "validation"] if not executed.empty else executed
    out["pretest_signal_trades"], out["pretest_executed_trades"], out["pretest_trades"] = int(len(pre_attempts)), int(len(pre_exec)), int(len(pre_attempts))
    assets = val.groupby("symbol")["account_return"].sum() if len(val) else pd.Series(dtype=float)
    out["validation_positive_asset_fraction"] = float((assets > 0).mean()) if len(assets) else 0.0
    lo, hi = moving_block_mean_ci(val["account_return"] if len(val) else [], samples=c.bootstrap_samples, block=c.bootstrap_block, seed=c.random_seed)
    out["validation_block_ci_low"], out["validation_block_ci_high"] = lo, hi
    out["validation_multiplicity_adjusted_p"] = _multiplicity_adjusted_p(val["account_return"] if len(val) else [], total_trials)
    pf, exp, dd, n = float(out.get("validation_profit_factor", np.nan)), float(out.get("validation_expectancy_r", np.nan)), abs(float(out.get("validation_max_drawdown", np.nan))), int(out.get("validation_trades", 0))
    out["validation_score_v20"] = float(exp * np.sqrt(max(n, 1)) + 0.10 * (min(pf, 5.0) - 1.0) - 0.75 * dd) if np.isfinite(exp) and np.isfinite(pf) and np.isfinite(dd) else -np.inf
    out["internal_eligible"] = bool(len(pre_attempts) >= c.min_pretest_trades and n >= c.min_validation_trades and np.isfinite(pf) and pf >= c.min_profit_factor and np.isfinite(exp) and exp > 0 and out["validation_positive_asset_fraction"] >= c.min_positive_asset_fraction and np.isfinite(dd) and dd <= c.max_drawdown and np.isfinite(lo) and lo > 0 and out["validation_multiplicity_adjusted_p"] <= c.max_multiplicity_p)
    return out


def evaluate_external_replication(attempts: pd.DataFrame, validation: V20ValidationConfig | None = None) -> dict:
    c = validation or V20ValidationConfig()
    executed = attempts[attempts["executed_v20"] == True].copy() if (not attempts.empty and "executed_v20" in attempts) else attempts.copy()  # noqa: E712
    if not executed.empty:
        executed["account_return"] = executed["account_return_v20"]
    m = summarize_trades(executed)
    assets = executed.groupby("symbol")["account_return"].sum() if len(executed) else pd.Series(dtype=float)
    lo, hi = moving_block_mean_ci(executed["account_return"] if len(executed) else [], samples=c.bootstrap_samples, block=c.bootstrap_block, seed=c.random_seed + 1)
    breadth = float((assets > 0).mean()) if len(assets) else 0.0
    passed = bool(m["trades"] >= c.min_external_trades and np.isfinite(float(m["profit_factor"])) and m["profit_factor"] >= c.min_profit_factor and np.isfinite(float(m["expectancy_r"])) and m["expectancy_r"] > 0 and breadth >= c.min_positive_asset_fraction and np.isfinite(float(m["max_drawdown"])) and abs(m["max_drawdown"]) <= c.max_drawdown and np.isfinite(lo) and lo > 0)
    return {**{f"external_{k}": v for k, v in m.items()}, "external_positive_asset_fraction": breadth, "external_block_ci_low": lo, "external_block_ci_high": hi, "external_pass": passed}


def apply_candidate_level_policy(attempts: pd.DataFrame, spec: StrategySpec, policy: RiskPsychologyPolicy | None = None) -> pd.DataFrame:
    """Apply risk state independently inside each evaluation segment.

    This prevents a development-period kill switch from mechanically suppressing
    all later validation/test observations while retaining causal, sequential
    state within each segment. Segment boundaries are part of the frozen
    evaluation protocol, not inferred from future outcomes.
    """
    if attempts.empty or "segment" not in attempts:
        return apply_risk_psychology_overlay(attempts, spec, policy)
    parts: list[pd.DataFrame] = []
    for segment in ("development", "validation", "test"):
        part = attempts[attempts["segment"] == segment].copy()
        if not part.empty:
            parts.append(apply_risk_psychology_overlay(part, spec, policy))
    if not parts:
        return attempts.copy()
    return pd.concat(parts, ignore_index=True).sort_values(["entry_time", "symbol"]).reset_index(drop=True)
