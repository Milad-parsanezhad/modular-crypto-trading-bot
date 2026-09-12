from __future__ import annotations

"""v0.39 mother-strategy orchestration.

ICT, SMC, Ichimoku, Brooks and completed higher-timeframe context are independent
engines. Their causal outputs become ML features; no hard k-of-n conjunction is
used. Capital is admitted only from conservative post-cost model outputs through
the v0.39 risk engine. Research only; PAPER/LIVE disabled; Kraken sealed.
"""

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from .brooks_engine_v39 import BrooksPolicyV39, add_brooks_features_v39
from .canonical_strategy_v39r import CanonicalConfig, build_canonical_features
from .financial_system_v39 import (
    FinancialRiskPolicyV39,
    allocate_portfolio_risk,
    causal_robust_normalize,
    tensor_ready_matrix,
)
from .multitimeframe_strategies_v20 import attach_completed_htf_context


@dataclass(frozen=True)
class MotherStrategyPolicyV39:
    timeframe: str = "4h"
    recent_structure_bars: int = 3
    stop_atr: float = 1.50
    min_stop_fraction: float = 0.003
    target_r: float = 3.0
    max_hold_bars: int = 30


@dataclass(frozen=True)
class PsychologyGovernanceV39:
    """Trading psychology expressed as deterministic governance, not discretion."""

    max_consecutive_losses: int = 3
    max_daily_entries: int = 4
    revenge_trading_allowed: bool = False
    martingale_allowed: bool = False
    averaging_down_allowed: bool = False
    discretionary_override_allowed: bool = False
    increase_risk_after_loss_allowed: bool = False


@dataclass(frozen=True)
class AccountStateV39:
    equity: float
    peak_equity: float
    consecutive_losses: int = 0
    entries_today: int = 0
    cooldown_active: bool = False
    open_total_risk_fraction: float = 0.0
    open_long_risk_fraction: float = 0.0
    open_short_risk_fraction: float = 0.0


MOTHER_ENGINE_NAMES_V39 = ("ict", "smc", "ichimoku", "brooks", "mtf")

NEURAL_FEATURES_V39 = (
    "atr_pct", "ret1", "ret12", "ema20_gap_atr_v39", "ema50_gap_atr_v39",
    "ema200_gap_atr_v39", "volume_ratio_v39", "trend_efficiency_v39",
    "ict_score_v39", "smc_score_v39", "ichimoku_score_v39",
    "brooks_score_v39", "mtf_score_v39", "engine_agreement_v39",
    "engine_dispersion_v39", "directional_prior_v39",
    "ict_recent_sweep_down_v39", "ict_recent_sweep_up_v39",
    "ict_mss_up_v39", "ict_mss_down_v39",
    "canonical_bull_displacement", "canonical_bear_displacement",
    "canonical_discount", "canonical_premium",
    "smc_bull_ob_retest_v39", "smc_bear_ob_retest_v39",
    "smc_demand_retest_v39", "smc_supply_retest_v39",
    "ichimoku_breakout_up_v39", "ichimoku_breakout_down_v39",
    "ichimoku_pullback_long_v39", "ichimoku_pullback_short_v39",
    "brooks_always_in_long_v39", "brooks_always_in_short_v39",
    "brooks_trend_regime_v39", "brooks_range_regime_v39",
    "brooks_breakout_up_v39", "brooks_breakout_down_v39",
    "brooks_failed_breakdown_v39", "brooks_failed_breakout_v39",
    "brooks_h1_v39", "brooks_h2_v39", "brooks_l1_v39", "brooks_l2_v39",
    "brooks_wedge_bottom_v39", "brooks_wedge_top_v39",
    "brooks_micro_double_bottom_v39", "brooks_micro_double_top_v39",
    "brooks_signal_bull_v39", "brooks_signal_bear_v39",
    "brooks_follow_through_bull_v39", "brooks_follow_through_bear_v39",
    "brooks_measured_move_progress_long_v39",
    "brooks_measured_move_progress_short_v39",
)


def mother_strategy_manifest_v39() -> dict:
    return {
        "version": "v0.39",
        "architecture": "INDEPENDENT_ENGINES_TO_REGULARIZED_FUSION",
        "engines": list(MOTHER_ENGINE_NAMES_V39),
        "hard_k_of_n_gate": False,
        "learning_target": "post_cost_R_distribution_and_duration",
        "normalization": "causal_rolling_median_MAD",
        "financial_risk": asdict(FinancialRiskPolicyV39()),
        "psychology": asdict(PsychologyGovernanceV39()),
        "exit_policy": asdict(MotherStrategyPolicyV39()),
        "paper_execution": False,
        "live_execution": False,
        "kraken_holdout": "SEALED",
    }


def _recent(flag: pd.Series, bars: int) -> pd.Series:
    return flag.fillna(False).astype(float).rolling(bars, min_periods=1).max().gt(0)


def _b(flag: pd.Series) -> pd.Series:
    return flag.fillna(False).astype(bool)


def build_mother_features_v39(
    frame: pd.DataFrame,
    config: MotherStrategyPolicyV39 | None = None,
) -> pd.DataFrame:
    """Build causal, independent strategy-engine outputs for one symbol."""

    cfg = config or MotherStrategyPolicyV39()
    canonical_cfg = CanonicalConfig(timeframe=cfg.timeframe, stop_atr=cfg.stop_atr)
    x = build_canonical_features(frame, canonical_cfg)
    if x.empty:
        return x

    htf = attach_completed_htf_context(frame, cfg.timeframe)
    htf_cols = ["timestamp"] + [c for c in htf.columns if c.startswith("htf_")]
    x = x.merge(htf[htf_cols], on="timestamp", how="left", validate="one_to_one")

    x["ema20_gap_atr_v39"] = (x["close"] - x["ema20"]) / x["atr"].replace(0.0, np.nan)
    x["ema50_gap_atr_v39"] = (x["close"] - x["ema50"]) / x["atr"].replace(0.0, np.nan)
    x["ema200_gap_atr_v39"] = (x["close"] - x["ema200"]) / x["atr"].replace(0.0, np.nan)
    x["volume_ratio_v39"] = x["volume"] / x["volume"].shift(1).rolling(20, min_periods=10).median().replace(0.0, np.nan)

    # ICT: liquidity raid -> MSS/structure -> displacement -> premium/discount.
    x["ict_recent_sweep_down_v39"] = _recent(x["sweep_down"], cfg.recent_structure_bars).astype("int8")
    x["ict_recent_sweep_up_v39"] = _recent(x["sweep_up"], cfg.recent_structure_bars).astype("int8")
    x["ict_mss_up_v39"] = (
        _recent(x["sweep_down"], cfg.recent_structure_bars)
        & _recent(x["canonical_choch_up"].eq(1) | x["bos_up"], cfg.recent_structure_bars)
    ).astype("int8")
    x["ict_mss_down_v39"] = (
        _recent(x["sweep_up"], cfg.recent_structure_bars)
        & _recent(x["canonical_choch_down"].eq(1) | x["bos_down"], cfg.recent_structure_bars)
    ).astype("int8")
    ict_long = pd.DataFrame({
        "sweep": x["ict_recent_sweep_down_v39"].eq(1),
        "mss": x["ict_mss_up_v39"].eq(1),
        "displacement": x["canonical_bull_displacement"].eq(1),
        "pd": x["canonical_discount"].eq(1),
    })
    ict_short = pd.DataFrame({
        "sweep": x["ict_recent_sweep_up_v39"].eq(1),
        "mss": x["ict_mss_down_v39"].eq(1),
        "displacement": x["canonical_bear_displacement"].eq(1),
        "pd": x["canonical_premium"].eq(1),
    })
    x["ict_score_v39"] = (ict_long.sum(axis=1) - ict_short.sum(axis=1)) / 4.0

    # SMC: BOS/CHoCH + imbalance + order-block and supply-demand mitigation.
    x["smc_bull_ob_retest_v39"] = x["canonical_bull_mitigation"].astype("int8")
    x["smc_bear_ob_retest_v39"] = x["canonical_bear_mitigation"].astype("int8")
    x["smc_demand_retest_v39"] = (
        x["demand_mid"].notna()
        & (x["low"] <= x["demand_mid"] + 0.50 * x["atr"])
        & (x["close"] > x["demand_mid"])
    ).astype("int8")
    x["smc_supply_retest_v39"] = (
        x["supply_mid"].notna()
        & (x["high"] >= x["supply_mid"] - 0.50 * x["atr"])
        & (x["close"] < x["supply_mid"])
    ).astype("int8")
    smc_long = pd.DataFrame({
        "structure": _b(x["bos_up"]) | x["canonical_choch_up"].eq(1),
        "fvg": _b(x["bull_fvg"]),
        "ob": x["smc_bull_ob_retest_v39"].eq(1),
        "sd": x["smc_demand_retest_v39"].eq(1),
    })
    smc_short = pd.DataFrame({
        "structure": _b(x["bos_down"]) | x["canonical_choch_down"].eq(1),
        "fvg": _b(x["bear_fvg"]),
        "ob": x["smc_bear_ob_retest_v39"].eq(1),
        "sd": x["smc_supply_retest_v39"].eq(1),
    })
    x["smc_score_v39"] = (smc_long.sum(axis=1) - smc_short.sum(axis=1)) / 4.0

    # Ichimoku remains independent from ICT/SMC.
    kijun_slope = x["kijun"].diff(3)
    x["ichimoku_breakout_up_v39"] = (
        (x["close"] > x["cloud_top"]) & (x["close"].shift(1) <= x["cloud_top"].shift(1))
    ).astype("int8")
    x["ichimoku_breakout_down_v39"] = (
        (x["close"] < x["cloud_bottom"]) & (x["close"].shift(1) >= x["cloud_bottom"].shift(1))
    ).astype("int8")
    x["ichimoku_pullback_long_v39"] = (
        x["canonical_kumo_bull"].eq(1)
        & (x["low"] <= x["kijun"] + 0.25 * x["atr"])
        & (x["close"] > x["kijun"])
    ).astype("int8")
    x["ichimoku_pullback_short_v39"] = (
        x["canonical_kumo_bear"].eq(1)
        & (x["high"] >= x["kijun"] - 0.25 * x["atr"])
        & (x["close"] < x["kijun"])
    ).astype("int8")
    ichi_long = pd.DataFrame({
        "kumo": x["canonical_kumo_bull"].eq(1),
        "tk": x["tenkan"] > x["kijun"],
        "slope": kijun_slope > 0,
        "trigger": x["ichimoku_breakout_up_v39"].eq(1) | x["ichimoku_pullback_long_v39"].eq(1),
    })
    ichi_short = pd.DataFrame({
        "kumo": x["canonical_kumo_bear"].eq(1),
        "tk": x["tenkan"] < x["kijun"],
        "slope": kijun_slope < 0,
        "trigger": x["ichimoku_breakout_down_v39"].eq(1) | x["ichimoku_pullback_short_v39"].eq(1),
    })
    x["ichimoku_score_v39"] = (ichi_long.sum(axis=1) - ichi_short.sum(axis=1)) / 4.0

    # Brooks engine is independent and exposes its full pattern vector.
    x = add_brooks_features_v39(x, BrooksPolicyV39())

    # Completed HTF only: 4H uses completed Daily context.
    htf_long = (x["htf_close"] > x["htf_ema50"]) & (x["htf_ema200_slope"].fillna(0.0) >= 0)
    htf_short = (x["htf_close"] < x["htf_ema50"]) & (x["htf_ema200_slope"].fillna(0.0) <= 0)
    x["mtf_score_v39"] = np.where(htf_long & ~htf_short, 1.0, np.where(htf_short & ~htf_long, -1.0, 0.0))

    engine_cols = [f"{name}_score_v39" for name in MOTHER_ENGINE_NAMES_V39]
    engines = x[engine_cols].astype(float)
    x["directional_prior_v39"] = engines.mean(axis=1)
    signs = np.sign(engines.to_numpy(dtype=float))
    nonzero = np.maximum(1, np.sum(signs != 0, axis=1))
    x["engine_agreement_v39"] = np.abs(np.sum(signs, axis=1)) / nonzero
    x["engine_dispersion_v39"] = engines.std(axis=1, ddof=0)
    x["engine_conflict_v39"] = ((engines.max(axis=1) > 0) & (engines.min(axis=1) < 0)).astype("int8")

    # Broad candidate pool. No 6/7 or all-engine requirement.
    event = (
        x["canonical_cusum_event"].eq(1) | _b(x["bos_up"]) | _b(x["bos_down"])
        | x["ict_recent_sweep_down_v39"].eq(1) | x["ict_recent_sweep_up_v39"].eq(1)
        | x["brooks_breakout_up_v39"].eq(1) | x["brooks_breakout_down_v39"].eq(1)
        | x["brooks_failed_breakdown_v39"].eq(1) | x["brooks_failed_breakout_v39"].eq(1)
    )
    x["mother_event_v39"] = event.astype("int8")
    x["research_candidate_side_v39"] = np.sign(x["directional_prior_v39"]).astype("int8")
    x.loc[~event, "research_candidate_side_v39"] = 0
    return x.replace([np.inf, -np.inf], np.nan)


def build_neural_panel_v39(
    frames: Mapping[str, pd.DataFrame],
    config: MotherStrategyPolicyV39 | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Return raw causal panel and finite float32 normalized NN matrix."""

    rows: list[pd.DataFrame] = []
    for symbol, frame in frames.items():
        z = build_mother_features_v39(frame, config)
        if z.empty:
            continue
        z = z.copy()
        z["symbol"] = str(symbol)
        rows.append(z)
    if not rows:
        return pd.DataFrame(), np.empty((0, len(NEURAL_FEATURES_V39) * 2), dtype=np.float32)

    panel = pd.concat(rows, ignore_index=True, sort=False)
    normalized = causal_robust_normalize(panel, NEURAL_FEATURES_V39, timestamp_col="timestamp", group_col="symbol")
    matrix = tensor_ready_matrix(normalized, NEURAL_FEATURES_V39, include_missing_indicators=True)
    return panel, matrix


def psychology_allows_new_trade_v39(
    state: AccountStateV39,
    policy: PsychologyGovernanceV39 | None = None,
) -> tuple[bool, str]:
    p = policy or PsychologyGovernanceV39()
    if state.cooldown_active:
        return False, "COOLDOWN_ACTIVE"
    if state.consecutive_losses >= p.max_consecutive_losses:
        return False, "LOSS_STREAK_LIMIT"
    if state.entries_today >= p.max_daily_entries:
        return False, "DAILY_ENTRY_LIMIT"
    if not np.isfinite(state.equity) or not np.isfinite(state.peak_equity) or state.equity <= 0 or state.peak_equity <= 0:
        return False, "INVALID_EQUITY_STATE"
    return True, ""


def _structural_stop_fraction(row: pd.Series, side: int, cfg: MotherStrategyPolicyV39) -> float:
    close, atr = float(row["close"]), float(row["atr"])
    if not np.isfinite(close) or not np.isfinite(atr) or close <= 0 or atr <= 0:
        return np.nan
    atr_distance = cfg.stop_atr * atr
    anchors: list[float] = []
    if side > 0:
        for col in ("last_swing_low", "bull_ob_mid", "demand_mid"):
            v = row.get(col, np.nan)
            if np.isfinite(v) and float(v) < close:
                anchors.append(float(v))
        structural = close - min(anchors) if anchors else atr_distance
    else:
        for col in ("last_swing_high", "bear_ob_mid", "supply_mid"):
            v = row.get(col, np.nan)
            if np.isfinite(v) and float(v) > close:
                anchors.append(float(v))
        structural = max(anchors) - close if anchors else atr_distance
    return float(max(cfg.min_stop_fraction, max(atr_distance, structural) / close))


def build_trade_proposals_v39(
    feature_rows: pd.DataFrame,
    predictions: pd.DataFrame,
    state: AccountStateV39,
    *,
    config: MotherStrategyPolicyV39 | None = None,
    risk_policy: FinancialRiskPolicyV39 | None = None,
    psychology_policy: PsychologyGovernanceV39 | None = None,
) -> pd.DataFrame:
    """Convert conservative post-cost ML outputs into bounded research proposals.

    Required predictions: timestamp, symbol, expected_r, lower_expected_r,
    upper_expected_r. Positive mean alone is never enough; lower_expected_r must
    also be >0. The function sizes research proposals but never places orders.
    """

    cfg = config or MotherStrategyPolicyV39()
    allowed, reason = psychology_allows_new_trade_v39(state, psychology_policy)
    output_cols = [
        "timestamp", "symbol", "side", "expected_r", "lower_expected_r",
        "upper_expected_r", "uncertainty_width_r", "stop_fraction", "target_r",
        "max_hold_bars", "allocated_risk_fraction", "position_weight", "reject_reason",
    ]
    if not allowed:
        out = pd.DataFrame(columns=output_cols)
        if not feature_rows.empty:
            out.loc[0, "reject_reason"] = reason
        return out

    pred_req = {"timestamp", "symbol", "expected_r", "lower_expected_r", "upper_expected_r"}
    feat_req = {"timestamp", "symbol", "research_candidate_side_v39", "mother_event_v39", "close", "atr"}
    if pred_req - set(predictions.columns):
        raise ValueError(f"missing prediction columns: {sorted(pred_req - set(predictions.columns))}")
    if feat_req - set(feature_rows.columns):
        raise ValueError(f"missing feature columns: {sorted(feat_req - set(feature_rows.columns))}")

    f, p = feature_rows.copy(), predictions.copy()
    f["timestamp"] = pd.to_datetime(f["timestamp"], utc=True, errors="raise")
    p["timestamp"] = pd.to_datetime(p["timestamp"], utc=True, errors="raise")
    z = f.merge(p, on=["timestamp", "symbol"], how="inner", validate="one_to_one")
    z = z.loc[z["mother_event_v39"].eq(1) & z["research_candidate_side_v39"].isin([-1, 1])].copy()
    if z.empty:
        return pd.DataFrame(columns=output_cols)

    z["side"] = z["research_candidate_side_v39"].astype(int)
    z["uncertainty_width_r"] = (
        pd.to_numeric(z["upper_expected_r"], errors="coerce")
        - pd.to_numeric(z["lower_expected_r"], errors="coerce")
    ).clip(lower=0.0)
    z["stop_fraction"] = [_structural_stop_fraction(row, int(row["side"]), cfg) for _, row in z.iterrows()]
    z["target_r"] = cfg.target_r
    z["max_hold_bars"] = cfg.max_hold_bars

    z = z.loc[
        pd.to_numeric(z["expected_r"], errors="coerce").gt(0.0)
        & pd.to_numeric(z["lower_expected_r"], errors="coerce").gt(0.0)
        & pd.to_numeric(z["stop_fraction"], errors="coerce").gt(0.0)
    ].copy()
    if z.empty:
        return pd.DataFrame(columns=output_cols)

    allocated = allocate_portfolio_risk(
        z[["symbol", "side", "lower_expected_r", "uncertainty_width_r", "stop_fraction"]],
        equity=state.equity,
        peak=state.peak_equity,
        open_total_risk_fraction=state.open_total_risk_fraction,
        open_long_risk_fraction=state.open_long_risk_fraction,
        open_short_risk_fraction=state.open_short_risk_fraction,
        policy=risk_policy,
    )
    z["allocated_risk_fraction"] = allocated["allocated_risk_fraction"].to_numpy()
    z["position_weight"] = allocated["position_weight"].to_numpy()
    z["reject_reason"] = np.where(z["allocated_risk_fraction"] > 0, "", "RISK_BUDGET_ZERO")
    return z[output_cols].reset_index(drop=True)
