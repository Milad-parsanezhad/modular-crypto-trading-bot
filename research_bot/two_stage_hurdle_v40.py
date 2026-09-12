from __future__ import annotations

"""v0.40 two-stage hurdle model for the robust mother strategy.

The model deliberately changes the learning decomposition after v0.39 showed a
bias/variance-admission frontier.  Stage 1 estimates the probability that a
causal mother-strategy event has positive post-cost R.  Stage 2 separately
estimates positive and non-positive R magnitudes plus holding duration.  The two
stages are recombined into an expected net-R distribution and a regime-aware,
one-sided conformal lower bound.

This module is research-only.  CoinEx/OKX/KuCoin are consumed development
venues.  Kraken remains sealed.  PAPER/LIVE are disabled.
"""

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import HuberRegressor, LogisticRegression


V40_CANDIDATES: tuple[str, ...] = (
    "V40_LOGIT_HUBER_HURDLE",
    "V40_HISTGB_HURDLE",
)
V40_SEEDS: tuple[int, ...] = (314, 1618, 2718)
DEVELOPMENT_VENUES_V40 = ("coinex", "okx", "kucoin")
RESERVED_HOLDOUT_VENUE_V40 = "kraken"


@dataclass(frozen=True)
class HurdlePolicyV40:
    stage1_probability_floor: float = 0.55
    conformal_alpha: float = 0.20
    probability_calibration_fraction: float = 0.50
    minimum_regime_conformal_events: int = 30
    minimum_class_events: int = 50
    duration_clip_bars: tuple[float, float] = (1.0, 30.0)

    def __post_init__(self) -> None:
        if not (0.5 <= self.stage1_probability_floor < 1.0):
            raise ValueError("stage1_probability_floor must be in [0.5,1)")
        if not (0.0 < self.conformal_alpha < 0.5):
            raise ValueError("conformal_alpha must be in (0,0.5)")
        if not (0.2 <= self.probability_calibration_fraction <= 0.8):
            raise ValueError("invalid calibration split")


def preregistration_manifest_v40() -> dict:
    return {
        "version": "v0.40",
        "experiment": "TWO_STAGE_META_LABEL_CONDITIONAL_NET_R_HURDLE",
        "candidate_order": list(V40_CANDIDATES),
        "selection_rule": "first/simplest candidate passing all frozen development gates",
        "development_venues": list(DEVELOPMENT_VENUES_V40),
        "reserved_holdout": RESERVED_HOLDOUT_VENUE_V40,
        "stage1_target": "1(net_r > 0)",
        "stage1_calibration": "chronological Platt scaling on first half of fold calibration segment",
        "stage2_positive_head": "E[positive net R | event features]",
        "stage2_loss_head": "E[absolute non-positive net R | event features]",
        "duration_head": "log1p holding bars",
        "expected_r_formula": "p_win * positive_magnitude - (1-p_win) * loss_magnitude",
        "uncertainty": "one-sided split conformal on second half of calibration segment, regime conditional with global fallback",
        "admission": "calibrated p_win >= 0.55 AND expected_r > 0 AND lower_expected_r > 0",
        "seeds": list(V40_SEEDS),
        "seed_rule": "median prediction; never choose best seed",
        "policy": asdict(HurdlePolicyV40()),
        "paper_execution": False,
        "live_execution": False,
        "kraken_touched": False,
        "post_result_threshold_relaxation": False,
    }


def _higher_quantile(values: np.ndarray, q: float) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        raise ValueError("cannot calibrate on empty residuals")
    return float(np.quantile(x, q, method="higher"))


def split_probability_and_conformal_calibration(
    calibration: pd.DataFrame,
    policy: HurdlePolicyV40 | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronologically split a fold calibration segment into disjoint halves."""
    p = policy or HurdlePolicyV40()
    x = calibration.copy()
    x["signal_time"] = pd.to_datetime(x["signal_time"], utc=True, errors="raise")
    x = x.sort_values("signal_time", kind="mergesort").reset_index(drop=True)
    cut = int(np.floor(len(x) * p.probability_calibration_fraction))
    cut = max(1, min(len(x) - 1, cut))
    return x.iloc[:cut].copy(), x.iloc[cut:].copy()


def _classifier(candidate: str, seed: int):
    if candidate == "V40_LOGIT_HUBER_HURDLE":
        return LogisticRegression(
            C=0.25,
            penalty="l2",
            solver="lbfgs",
            max_iter=1000,
            class_weight="balanced",
            random_state=seed,
        )
    if candidate == "V40_HISTGB_HURDLE":
        return HistGradientBoostingClassifier(
            learning_rate=0.04,
            max_iter=160,
            max_leaf_nodes=15,
            l2_regularization=2.0,
            min_samples_leaf=30,
            early_stopping=False,
            random_state=seed,
        )
    raise ValueError(candidate)


def _regressor(candidate: str, seed: int):
    if candidate == "V40_LOGIT_HUBER_HURDLE":
        return HuberRegressor(epsilon=1.35, alpha=0.01, max_iter=500)
    if candidate == "V40_HISTGB_HURDLE":
        return HistGradientBoostingRegressor(
            learning_rate=0.04,
            max_iter=160,
            max_leaf_nodes=15,
            l2_regularization=2.0,
            min_samples_leaf=30,
            early_stopping=False,
            random_state=seed,
        )
    raise ValueError(candidate)


def _safe_logit(probability: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=float), 1e-5, 1.0 - 1e-5)
    return np.log(p / (1.0 - p)).reshape(-1, 1)


def _fit_platt(raw_probability: np.ndarray, y: np.ndarray) -> LogisticRegression:
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) < 2:
        raise ValueError("Platt calibration requires both classes")
    model = LogisticRegression(C=10.0, solver="lbfgs", max_iter=500)
    model.fit(_safe_logit(raw_probability), y)
    return model


def _calibrated_probability(classifier, platt, X: np.ndarray) -> np.ndarray:
    raw = np.asarray(classifier.predict_proba(X)[:, 1], dtype=float)
    return np.asarray(platt.predict_proba(_safe_logit(raw))[:, 1], dtype=float)


def _regime_buffers(
    calibration: pd.DataFrame,
    prediction: np.ndarray,
    policy: HurdlePolicyV40,
) -> tuple[dict[str, tuple[float, float]], tuple[float, float]]:
    y = pd.to_numeric(calibration["net_r"], errors="coerce").to_numpy(dtype=float)
    pred = np.asarray(prediction, dtype=float)
    valid = np.isfinite(y) & np.isfinite(pred)
    if int(valid.sum()) < 20:
        raise ValueError("insufficient conformal calibration observations")
    lower_residual = pred[valid] - y[valid]
    upper_residual = y[valid] - pred[valid]
    q = 1.0 - policy.conformal_alpha
    global_buffers = (
        _higher_quantile(lower_residual, q),
        _higher_quantile(upper_residual, q),
    )
    regime_buffers: dict[str, tuple[float, float]] = {}
    regimes = calibration["regime"].astype(str).to_numpy()
    for regime in sorted(set(regimes.tolist())):
        mask = valid & (regimes == regime)
        if int(mask.sum()) < policy.minimum_regime_conformal_events:
            continue
        regime_buffers[regime] = (
            _higher_quantile(pred[mask] - y[mask], q),
            _higher_quantile(y[mask] - pred[mask], q),
        )
    return regime_buffers, global_buffers


def _apply_buffers(
    frame: pd.DataFrame,
    expected_r: np.ndarray,
    regime_buffers: dict[str, tuple[float, float]],
    global_buffers: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.empty(len(frame), dtype=float)
    upper = np.empty(len(frame), dtype=float)
    for i, regime in enumerate(frame["regime"].astype(str).tolist()):
        lo_q, hi_q = regime_buffers.get(regime, global_buffers)
        lower[i] = float(expected_r[i] - lo_q)
        upper[i] = float(expected_r[i] + hi_q)
    return lower, upper


def fit_predict_seed_v40(
    candidate: str,
    fit: pd.DataFrame,
    probability_calibration: pd.DataFrame,
    conformal_calibration: pd.DataFrame,
    target: pd.DataFrame,
    feature_columns: Sequence[str],
    seed: int,
    policy: HurdlePolicyV40 | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Fit one seed without using target labels for fitting or calibration."""
    p = policy or HurdlePolicyV40()
    Xfit = fit.loc[:, feature_columns].to_numpy(dtype=np.float32)
    Xprob = probability_calibration.loc[:, feature_columns].to_numpy(dtype=np.float32)
    Xconf = conformal_calibration.loc[:, feature_columns].to_numpy(dtype=np.float32)
    Xtarget = target.loc[:, feature_columns].to_numpy(dtype=np.float32)

    yfit = pd.to_numeric(fit["net_r"], errors="coerce").to_numpy(dtype=float)
    yclass = (yfit > 0.0).astype(int)
    if min(int((yclass == 0).sum()), int((yclass == 1).sum())) < p.minimum_class_events:
        raise ValueError("insufficient positive/negative events for v0.40 hurdle fit")

    classifier = _classifier(candidate, seed)
    classifier.fit(Xfit, yclass)
    raw_prob_cal = np.asarray(classifier.predict_proba(Xprob)[:, 1], dtype=float)
    yprob = (pd.to_numeric(probability_calibration["net_r"], errors="coerce").to_numpy(dtype=float) > 0.0).astype(int)
    platt = _fit_platt(raw_prob_cal, yprob)

    positive_mask = yfit > 0.0
    loss_mask = ~positive_mask
    positive_model = _regressor(candidate, seed)
    loss_model = _regressor(candidate, seed + 10000)
    duration_model = _regressor(candidate, seed + 20000)
    positive_model.fit(Xfit[positive_mask], yfit[positive_mask])
    loss_model.fit(Xfit[loss_mask], -yfit[loss_mask])

    entry = pd.to_datetime(fit["entry_time"], utc=True, errors="raise")
    exit_ = pd.to_datetime(fit["exit_time"], utc=True, errors="raise")
    duration_bars = (exit_ - entry).dt.total_seconds().div(4.0 * 3600.0).clip(lower=1.0)
    duration_model.fit(Xfit, np.log1p(duration_bars.to_numpy(dtype=float)))

    def components(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        prob = _calibrated_probability(classifier, platt, X)
        pos = np.clip(np.asarray(positive_model.predict(X), dtype=float), 0.0, 5.0)
        loss = np.clip(np.asarray(loss_model.predict(X), dtype=float), 0.0, 5.0)
        expected = prob * pos - (1.0 - prob) * loss
        duration = np.expm1(np.asarray(duration_model.predict(X), dtype=float))
        duration = np.clip(duration, p.duration_clip_bars[0], p.duration_clip_bars[1])
        return prob, expected, duration, pos

    prob_conf, expected_conf, duration_conf, _ = components(Xconf)
    regime_buffers, global_buffers = _regime_buffers(
        conformal_calibration, expected_conf, p
    )
    prob_target, expected_target, duration_target, positive_target = components(Xtarget)
    lower_target, upper_target = _apply_buffers(
        target, expected_target, regime_buffers, global_buffers
    )

    out = target.copy()
    out["candidate"] = candidate
    out["seed"] = int(seed)
    out["p_win_v40"] = prob_target
    out["expected_r_v40"] = expected_target
    out["lower_expected_r_v40"] = lower_target
    out["upper_expected_r_v40"] = upper_target
    out["predicted_duration_bars_v40"] = duration_target
    out["predicted_positive_magnitude_v40"] = positive_target
    out["utility_v40"] = lower_target / np.sqrt(1.0 + duration_target)
    out["selected_v40"] = (
        (out["p_win_v40"] >= p.stage1_probability_floor)
        & (out["expected_r_v40"] > 0.0)
        & (out["lower_expected_r_v40"] > 0.0)
    )

    yconf = pd.to_numeric(conformal_calibration["net_r"], errors="coerce").to_numpy(dtype=float)
    brier_prob = (yconf > 0.0).astype(float)
    brier = float(np.mean((prob_conf - brier_prob) ** 2))
    baseline_p = float(np.mean(brier_prob))
    baseline_brier = float(np.mean((baseline_p - brier_prob) ** 2))
    diag = {
        "candidate": candidate,
        "seed": int(seed),
        "fit_events": int(len(fit)),
        "probability_calibration_events": int(len(probability_calibration)),
        "conformal_calibration_events": int(len(conformal_calibration)),
        "target_events": int(len(target)),
        "selected_events": int(out["selected_v40"].sum()),
        "brier": brier,
        "baseline_brier": baseline_brier,
        "brier_skill_positive": bool(brier < baseline_brier),
        "regime_specific_buffers": sorted(regime_buffers.keys()),
        "global_lower_buffer_r": float(global_buffers[0]),
        "global_upper_buffer_r": float(global_buffers[1]),
        "median_predicted_duration_bars": float(np.nanmedian(duration_target)),
    }
    return out, diag


def median_seed_prediction_v40(seed_outputs: Sequence[pd.DataFrame], policy: HurdlePolicyV40 | None = None) -> pd.DataFrame:
    """Combine exactly the three frozen seeds by row-wise median."""
    p = policy or HurdlePolicyV40()
    if len(seed_outputs) != len(V40_SEEDS):
        raise ValueError("v0.40 requires exactly three seed outputs")
    base = seed_outputs[0].copy().reset_index(drop=True)
    keys = ["series_id", "signal_time", "entry_time", "exit_time"]
    for other in seed_outputs[1:]:
        if len(other) != len(base):
            raise ValueError("seed outputs have different lengths")
        for key in keys:
            if not np.array_equal(base[key].astype(str).to_numpy(), other[key].astype(str).to_numpy()):
                raise ValueError("seed output row identity mismatch")
    cols = [
        "p_win_v40",
        "expected_r_v40",
        "lower_expected_r_v40",
        "upper_expected_r_v40",
        "predicted_duration_bars_v40",
        "predicted_positive_magnitude_v40",
        "utility_v40",
    ]
    for col in cols:
        stack = np.stack([pd.to_numeric(x[col], errors="coerce").to_numpy(dtype=float) for x in seed_outputs], axis=0)
        base[col] = np.nanmedian(stack, axis=0)
    base["seed"] = "MEDIAN_3"
    base["selected_v40"] = (
        (base["p_win_v40"] >= p.stage1_probability_floor)
        & (base["expected_r_v40"] > 0.0)
        & (base["lower_expected_r_v40"] > 0.0)
    )
    return base
