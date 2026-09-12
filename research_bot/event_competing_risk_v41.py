from __future__ import annotations

"""v0.41 event-specific discrete-time competing-risk engine.

This stage follows the negative v0.40 result.  It does not increase neural-model
capacity.  Instead it changes the representation and target: mother-strategy
setups are assigned to frozen semantic event families and modeled as two
competing discrete-time hazards (TARGET vs STOP), with TIME treated as right
censoring up to the frozen 30-bar horizon.  A timeout-value head supplies the
conditional R value if neither absorbing cause occurs by the horizon.

Research only.  CoinEx/OKX/KuCoin are consumed development venues.  Kraken is
sealed.  PAPER/LIVE execution remain disabled.
"""

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import HuberRegressor, LogisticRegression


V41_CANDIDATES: tuple[str, ...] = (
    "V41_LOGIT_CAUSE_SPECIFIC",
    "V41_HISTGB_CAUSE_SPECIFIC",
)
V41_SEEDS: tuple[int, ...] = (314, 1618, 2718)
V41_EVENT_FAMILIES: tuple[str, ...] = (
    "ICT_MSS",
    "BROOKS_H2L2",
    "ICHIMOKU_BREAKOUT",
    "ICHIMOKU_PULLBACK",
    "SMC_OB_RETEST",
    "BROOKS_FAILED_BREAKOUT",
    "SMC_FVG_STRUCTURE",
    "LIQUIDITY_SWEEP",
    "OTHER_MOTHER_EVENT",
)
V41_REGIMES: tuple[str, ...] = ("trend", "range", "transition")
DEVELOPMENT_VENUES_V41 = ("coinex", "okx", "kucoin")
RESERVED_HOLDOUT_V41 = "kraken"


# Deliberately compact semantic representation.  These columns already exist in
# the causally normalized v0.39 event panel.
V41_FEATURES: tuple[str, ...] = (
    "atr_pct",
    "ret1",
    "ret12",
    "ema20_gap_atr_v39",
    "ema50_gap_atr_v39",
    "ema200_gap_atr_v39",
    "volume_ratio_v39",
    "trend_efficiency_v39",
    "ict_score_v39",
    "smc_score_v39",
    "ichimoku_score_v39",
    "brooks_score_v39",
    "mtf_score_v39",
    "engine_agreement_v39",
    "engine_dispersion_v39",
    "directional_prior_v39",
    "ict_recent_sweep_down_v39",
    "ict_recent_sweep_up_v39",
    "ict_mss_up_v39",
    "ict_mss_down_v39",
    "canonical_bull_displacement",
    "canonical_bear_displacement",
    "canonical_discount",
    "canonical_premium",
    "smc_bull_ob_retest_v39",
    "smc_bear_ob_retest_v39",
    "smc_demand_retest_v39",
    "smc_supply_retest_v39",
    "ichimoku_breakout_up_v39",
    "ichimoku_breakout_down_v39",
    "ichimoku_pullback_long_v39",
    "ichimoku_pullback_short_v39",
    "brooks_h1_v39",
    "brooks_h2_v39",
    "brooks_l1_v39",
    "brooks_l2_v39",
    "brooks_failed_breakdown_v39",
    "brooks_failed_breakout_v39",
    "brooks_follow_through_bull_v39",
    "brooks_follow_through_bear_v39",
)


@dataclass(frozen=True)
class CompetingRiskPolicyV41:
    max_hold_bars: int = 30
    conformal_alpha: float = 0.20
    minimum_expert_events: int = 300
    minimum_expert_cause_events: int = 40
    minimum_family_conformal_events: int = 30
    hazard_cap: float = 0.95
    require_target_probability_gt_stop: bool = True

    def __post_init__(self) -> None:
        if self.max_hold_bars < 2:
            raise ValueError("max_hold_bars must be >=2")
        if not (0.0 < self.conformal_alpha < 0.5):
            raise ValueError("conformal_alpha must be in (0,0.5)")
        if not (0.5 <= self.hazard_cap < 1.0):
            raise ValueError("hazard_cap must be in [0.5,1)")


def preregistration_manifest_v41() -> dict:
    return {
        "version": "v0.41",
        "experiment": "EVENT_SPECIFIC_DISCRETE_TIME_COMPETING_RISK",
        "candidate_order": list(V41_CANDIDATES),
        "event_families": list(V41_EVENT_FAMILIES),
        "causes": ["TARGET", "STOP"],
        "censoring": "TIME/right-censored at observed or 30-bar horizon",
        "selection_rule": "first/simplest candidate passing every frozen gate",
        "representation": "frozen compact semantic causal v0.39 features + family + side + regime + time",
        "expert_rule": "family x side expert only if frozen sample/cause minima are met; otherwise pooled hierarchical fallback",
        "uncertainty": "family-conditional split conformal residual lower bound with global fallback",
        "admission": "lower_expected_r > 0 AND P(target before horizon) > P(stop before horizon)",
        "development_venues": list(DEVELOPMENT_VENUES_V41),
        "reserved_holdout": RESERVED_HOLDOUT_V41,
        "seeds": list(V41_SEEDS),
        "seed_rule": "row-wise median; never choose best seed",
        "policy": asdict(CompetingRiskPolicyV41()),
        "paper_execution": False,
        "live_execution": False,
        "kraken_touched": False,
        "post_result_threshold_relaxation": False,
    }


def assign_event_family_v41(features: pd.DataFrame) -> pd.Series:
    """Assign exactly one frozen semantic family using causal signal-time flags."""
    x = features
    family = pd.Series("OTHER_MOTHER_EVENT", index=x.index, dtype="object")
    # Apply reverse priority so the first conceptual family below wins.
    masks: list[tuple[str, pd.Series]] = [
        ("LIQUIDITY_SWEEP", x["ict_recent_sweep_down_v39"].eq(1) | x["ict_recent_sweep_up_v39"].eq(1)),
        ("SMC_FVG_STRUCTURE", (x["canonical_bull_displacement"].eq(1) | x["canonical_bear_displacement"].eq(1))),
        ("BROOKS_FAILED_BREAKOUT", x["brooks_failed_breakdown_v39"].eq(1) | x["brooks_failed_breakout_v39"].eq(1)),
        ("SMC_OB_RETEST", x["smc_bull_ob_retest_v39"].eq(1) | x["smc_bear_ob_retest_v39"].eq(1)),
        ("ICHIMOKU_PULLBACK", x["ichimoku_pullback_long_v39"].eq(1) | x["ichimoku_pullback_short_v39"].eq(1)),
        ("ICHIMOKU_BREAKOUT", x["ichimoku_breakout_up_v39"].eq(1) | x["ichimoku_breakout_down_v39"].eq(1)),
        ("BROOKS_H2L2", x["brooks_h2_v39"].eq(1) | x["brooks_l2_v39"].eq(1)),
        ("ICT_MSS", x["ict_mss_up_v39"].eq(1) | x["ict_mss_down_v39"].eq(1)),
    ]
    for name, mask in masks:
        family.loc[mask.fillna(False)] = name
    family.loc[~x["mother_event_v39"].eq(1)] = "OTHER_MOTHER_EVENT"
    return family.astype(str)


def attach_event_family_v41(panel: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Attach signal-time family labels from raw mother features to event rows."""
    p = panel.copy()
    p["event_family_v41"] = assign_event_family_v41(p)
    lookup = p[["series_id", "timestamp", "event_family_v41"]].rename(columns={"timestamp": "signal_time"})
    out = events.merge(lookup, on=["series_id", "signal_time"], how="left", validate="many_to_one")
    out["event_family_v41"] = out["event_family_v41"].fillna("OTHER_MOTHER_EVENT")
    return out


def _duration_bars(events: pd.DataFrame, max_hold: int) -> np.ndarray:
    entry = pd.to_datetime(events["entry_time"], utc=True, errors="raise")
    exit_ = pd.to_datetime(events["exit_time"], utc=True, errors="raise")
    bars = ((exit_ - entry).dt.total_seconds() / (4.0 * 3600.0)).round().astype(int) + 1
    return bars.clip(lower=1, upper=max_hold).to_numpy(dtype=int)


def _category_matrix(events: pd.DataFrame) -> np.ndarray:
    family = events["event_family_v41"].astype(str).to_numpy()
    regime = events["regime"].astype(str).to_numpy()
    side = pd.to_numeric(events["side"], errors="coerce").fillna(0).to_numpy(dtype=int)
    cols: list[np.ndarray] = []
    for name in V41_EVENT_FAMILIES:
        cols.append((family == name).astype(np.float32))
    cols.append((side > 0).astype(np.float32))
    cols.append((side < 0).astype(np.float32))
    for name in V41_REGIMES:
        cols.append((regime == name).astype(np.float32))
    return np.stack(cols, axis=1) if cols else np.empty((len(events), 0), dtype=np.float32)


def event_design_matrix_v41(events: pd.DataFrame, feature_columns: Sequence[str]) -> np.ndarray:
    base = events.loc[:, feature_columns].to_numpy(dtype=np.float32)
    return np.concatenate([base, _category_matrix(events)], axis=1).astype(np.float32, copy=False)


def hazard_design_matrix_v41(
    events: pd.DataFrame,
    feature_columns: Sequence[str],
    time_bar: np.ndarray,
    max_hold: int,
) -> np.ndarray:
    base = event_design_matrix_v41(events, feature_columns)
    t = np.asarray(time_bar, dtype=np.float32)
    time_features = np.stack(
        [t / float(max_hold), np.log1p(t), np.sqrt(t / float(max_hold))], axis=1
    ).astype(np.float32)
    return np.concatenate([base, time_features], axis=1).astype(np.float32, copy=False)


def build_hazard_training_table_v41(
    events: pd.DataFrame,
    feature_columns: Sequence[str],
    policy: CompetingRiskPolicyV41 | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Expand event rows into at-risk discrete-time rows without future covariates."""
    p = policy or CompetingRiskPolicyV41()
    durations = _duration_bars(events, p.max_hold_bars)
    X_parts: list[np.ndarray] = []
    y_target: list[int] = []
    y_stop: list[int] = []
    owner: list[int] = []
    outcomes = events["outcome"].astype(str).to_numpy()
    for i, duration in enumerate(durations):
        times = np.arange(1, int(duration) + 1, dtype=np.int16)
        repeated = events.iloc[[i] * len(times)]
        X_parts.append(hazard_design_matrix_v41(repeated, feature_columns, times, p.max_hold_bars))
        for t in times:
            terminal = int(t) == int(duration)
            y_target.append(int(terminal and outcomes[i] == "TARGET"))
            y_stop.append(int(terminal and outcomes[i] == "STOP"))
            owner.append(i)
    X = np.concatenate(X_parts, axis=0) if X_parts else np.empty((0, len(feature_columns) + 17), dtype=np.float32)
    return X, np.asarray(y_target, dtype=np.int8), np.asarray(y_stop, dtype=np.int8), np.asarray(owner, dtype=np.int32)


class _ConstantBinary:
    def __init__(self, probability: float):
        self.probability = float(np.clip(probability, 1e-6, 1.0 - 1e-6))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = np.full(len(X), self.probability, dtype=float)
        return np.stack([1.0 - p, p], axis=1)


class _ConstantRegressor:
    def __init__(self, value: float):
        self.value = float(value)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self.value, dtype=float)


def _fit_binary(candidate: str, X: np.ndarray, y: np.ndarray, seed: int):
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) < 2:
        return _ConstantBinary(float(np.mean(y)) if len(y) else 0.0)
    if candidate == "V41_LOGIT_CAUSE_SPECIFIC":
        model = LogisticRegression(
            C=0.10,
            penalty="l2",
            solver="lbfgs",
            class_weight="balanced",
            max_iter=1000,
            random_state=seed,
        )
    elif candidate == "V41_HISTGB_CAUSE_SPECIFIC":
        model = HistGradientBoostingClassifier(
            learning_rate=0.04,
            max_iter=120,
            max_leaf_nodes=15,
            l2_regularization=2.0,
            min_samples_leaf=50,
            early_stopping=False,
            random_state=seed,
        )
    else:
        raise ValueError(candidate)
    model.fit(X, y)
    return model


def _fit_timeout_regressor(candidate: str, events: pd.DataFrame, X_event: np.ndarray, seed: int):
    mask = events["outcome"].astype(str).eq("TIME").to_numpy()
    y = pd.to_numeric(events.loc[mask, "net_r"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(y)
    if int(valid.sum()) < 50:
        fallback = float(np.nanmedian(pd.to_numeric(events["net_r"], errors="coerce")))
        return _ConstantRegressor(fallback if np.isfinite(fallback) else 0.0)
    X = X_event[mask][valid]
    y = y[valid]
    if candidate == "V41_LOGIT_CAUSE_SPECIFIC":
        model = HuberRegressor(epsilon=1.35, alpha=0.01, max_iter=500)
    else:
        model = HistGradientBoostingRegressor(
            learning_rate=0.04,
            max_iter=120,
            max_leaf_nodes=15,
            l2_regularization=2.0,
            min_samples_leaf=40,
            early_stopping=False,
            random_state=seed + 20000,
        )
    model.fit(X, y)
    return model


@dataclass
class CauseSpecificBundleV41:
    candidate: str
    seed: int
    target_pooled: object
    stop_pooled: object
    timeout_pooled: object
    experts: dict[tuple[str, int], tuple[object, object]]
    feature_columns: tuple[str, ...]
    policy: CompetingRiskPolicyV41


def fit_competing_risk_bundle_v41(
    candidate: str,
    events: pd.DataFrame,
    feature_columns: Sequence[str],
    seed: int,
    policy: CompetingRiskPolicyV41 | None = None,
) -> CauseSpecificBundleV41:
    p = policy or CompetingRiskPolicyV41()
    Xh, yt, ys, owner = build_hazard_training_table_v41(events, feature_columns, p)
    target_pooled = _fit_binary(candidate, Xh, yt, seed)
    stop_pooled = _fit_binary(candidate, Xh, ys, seed + 10000)
    Xe = event_design_matrix_v41(events, feature_columns)
    timeout_pooled = _fit_timeout_regressor(candidate, events, Xe, seed)

    experts: dict[tuple[str, int], tuple[object, object]] = {}
    family = events["event_family_v41"].astype(str).to_numpy()
    side = pd.to_numeric(events["side"], errors="coerce").fillna(0).to_numpy(dtype=int)
    outcome = events["outcome"].astype(str).to_numpy()
    for fam in V41_EVENT_FAMILIES:
        for s in (-1, 1):
            emask = (family == fam) & (side == s)
            n_events = int(emask.sum())
            if n_events < p.minimum_expert_events:
                continue
            if int(np.sum(emask & (outcome == "TARGET"))) < p.minimum_expert_cause_events:
                continue
            if int(np.sum(emask & (outcome == "STOP"))) < p.minimum_expert_cause_events:
                continue
            hmask = emask[owner]
            experts[(fam, s)] = (
                _fit_binary(candidate, Xh[hmask], yt[hmask], seed),
                _fit_binary(candidate, Xh[hmask], ys[hmask], seed + 10000),
            )
    return CauseSpecificBundleV41(
        candidate=candidate,
        seed=int(seed),
        target_pooled=target_pooled,
        stop_pooled=stop_pooled,
        timeout_pooled=timeout_pooled,
        experts=experts,
        feature_columns=tuple(feature_columns),
        policy=p,
    )


def _hazard_pair(bundle: CauseSpecificBundleV41, row: pd.DataFrame, time_bar: int) -> tuple[float, float]:
    p = bundle.policy
    X = hazard_design_matrix_v41(
        row,
        bundle.feature_columns,
        np.asarray([time_bar], dtype=np.int16),
        p.max_hold_bars,
    )
    fam = str(row["event_family_v41"].iloc[0])
    side = int(row["side"].iloc[0])
    target_model, stop_model = bundle.experts.get((fam, side), (bundle.target_pooled, bundle.stop_pooled))
    ht = float(target_model.predict_proba(X)[0, 1])
    hs = float(stop_model.predict_proba(X)[0, 1])
    ht = float(np.clip(ht, 0.0, p.hazard_cap))
    hs = float(np.clip(hs, 0.0, p.hazard_cap))
    total = ht + hs
    if total > p.hazard_cap:
        scale = p.hazard_cap / total
        ht *= scale
        hs *= scale
    return ht, hs


def predict_competing_risks_v41(bundle: CauseSpecificBundleV41, target: pd.DataFrame) -> pd.DataFrame:
    """Return CIF target/stop, timeout survival, duration and expected post-cost R."""
    p = bundle.policy
    out = target.copy().reset_index(drop=True)
    p_target = np.zeros(len(out), dtype=float)
    p_stop = np.zeros(len(out), dtype=float)
    p_timeout = np.zeros(len(out), dtype=float)
    expected_duration = np.zeros(len(out), dtype=float)

    for i in range(len(out)):
        row = out.iloc[[i]]
        survival = 1.0
        duration = 0.0
        for t in range(1, p.max_hold_bars + 1):
            ht, hs = _hazard_pair(bundle, row, t)
            p_target[i] += survival * ht
            p_stop[i] += survival * hs
            duration += survival
            survival *= max(0.0, 1.0 - ht - hs)
        p_timeout[i] = survival
        expected_duration[i] = float(np.clip(duration, 1.0, p.max_hold_bars))

    Xe = event_design_matrix_v41(out, bundle.feature_columns)
    timeout_r = np.asarray(bundle.timeout_pooled.predict(Xe), dtype=float)
    base_cost_r = pd.to_numeric(out["base_cost_r"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    target_r = 3.0 - base_cost_r
    stop_r = -1.0 - base_cost_r
    expected_r = p_target * target_r + p_stop * stop_r + p_timeout * timeout_r

    out["candidate"] = bundle.candidate
    out["seed"] = int(bundle.seed)
    out["p_target_v41"] = p_target
    out["p_stop_v41"] = p_stop
    out["p_timeout_v41"] = p_timeout
    out["predicted_timeout_r_v41"] = timeout_r
    out["expected_duration_bars_v41"] = expected_duration
    out["expected_r_v41"] = expected_r
    out["expert_used_v41"] = [
        int((str(f), int(s)) in bundle.experts)
        for f, s in zip(out["event_family_v41"], out["side"])
    ]
    return out


def _higher_quantile(values: np.ndarray, q: float) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        raise ValueError("empty conformal residuals")
    return float(np.quantile(x, q, method="higher"))


def calibrate_expected_r_bounds_v41(
    calibration: pd.DataFrame,
    calibration_prediction: pd.DataFrame,
    target_prediction: pd.DataFrame,
    policy: CompetingRiskPolicyV41 | None = None,
) -> pd.DataFrame:
    p = policy or CompetingRiskPolicyV41()
    y = pd.to_numeric(calibration["net_r"], errors="coerce").to_numpy(dtype=float)
    pred = pd.to_numeric(calibration_prediction["expected_r_v41"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(y) & np.isfinite(pred)
    if int(valid.sum()) < 50:
        raise ValueError("insufficient v0.41 conformal calibration events")
    q = 1.0 - p.conformal_alpha
    global_lo = _higher_quantile(pred[valid] - y[valid], q)
    global_hi = _higher_quantile(y[valid] - pred[valid], q)

    family_buffers: dict[str, tuple[float, float]] = {}
    fam = calibration["event_family_v41"].astype(str).to_numpy()
    for name in V41_EVENT_FAMILIES:
        mask = valid & (fam == name)
        if int(mask.sum()) < p.minimum_family_conformal_events:
            continue
        family_buffers[name] = (
            _higher_quantile(pred[mask] - y[mask], q),
            _higher_quantile(y[mask] - pred[mask], q),
        )

    out = target_prediction.copy()
    lower = np.empty(len(out), dtype=float)
    upper = np.empty(len(out), dtype=float)
    for i, name in enumerate(out["event_family_v41"].astype(str).tolist()):
        lo, hi = family_buffers.get(name, (global_lo, global_hi))
        expected = float(out["expected_r_v41"].iloc[i])
        lower[i] = expected - lo
        upper[i] = expected + hi
    out["lower_expected_r_v41"] = lower
    out["upper_expected_r_v41"] = upper
    out["selected_v41"] = out["lower_expected_r_v41"].gt(0.0)
    if p.require_target_probability_gt_stop:
        out["selected_v41"] &= out["p_target_v41"].gt(out["p_stop_v41"])
    out["utility_v41"] = out["lower_expected_r_v41"] / np.sqrt(1.0 + out["expected_duration_bars_v41"])
    return out


def median_seed_prediction_v41(seed_outputs: Sequence[pd.DataFrame], policy: CompetingRiskPolicyV41 | None = None) -> pd.DataFrame:
    p = policy or CompetingRiskPolicyV41()
    if len(seed_outputs) != len(V41_SEEDS):
        raise ValueError("v0.41 requires exactly three frozen seed outputs")
    base = seed_outputs[0].copy().reset_index(drop=True)
    keys = ["series_id", "signal_time", "entry_time", "exit_time", "event_family_v41"]
    for other in seed_outputs[1:]:
        if len(other) != len(base):
            raise ValueError("seed output length mismatch")
        for key in keys:
            if not np.array_equal(base[key].astype(str).to_numpy(), other[key].astype(str).to_numpy()):
                raise ValueError("seed output row identity mismatch")
    cols = (
        "p_target_v41",
        "p_stop_v41",
        "p_timeout_v41",
        "predicted_timeout_r_v41",
        "expected_duration_bars_v41",
        "expected_r_v41",
        "lower_expected_r_v41",
        "upper_expected_r_v41",
        "utility_v41",
    )
    for col in cols:
        stack = np.stack([pd.to_numeric(x[col], errors="coerce").to_numpy(dtype=float) for x in seed_outputs], axis=0)
        base[col] = np.nanmedian(stack, axis=0)
    base["seed"] = "MEDIAN_3"
    base["selected_v41"] = base["lower_expected_r_v41"].gt(0.0)
    if p.require_target_probability_gt_stop:
        base["selected_v41"] &= base["p_target_v41"].gt(base["p_stop_v41"])
    return base
