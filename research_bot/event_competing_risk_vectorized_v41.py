from __future__ import annotations

"""Vectorized, mathematically equivalent inference for v0.41 competing risks."""

import numpy as np
import pandas as pd

from .event_competing_risk_v41 import (
    CauseSpecificBundleV41,
    event_design_matrix_v41,
    hazard_design_matrix_v41,
)


def predict_competing_risks_vectorized_v41(
    bundle: CauseSpecificBundleV41,
    target: pd.DataFrame,
) -> pd.DataFrame:
    p = bundle.policy
    out = target.copy().reset_index(drop=True)
    n = len(out)
    p_target = np.zeros(n, dtype=float)
    p_stop = np.zeros(n, dtype=float)
    survival = np.ones(n, dtype=float)
    expected_duration = np.zeros(n, dtype=float)
    families = out["event_family_v41"].astype(str).to_numpy()
    sides = pd.to_numeric(out["side"], errors="coerce").fillna(0).to_numpy(dtype=int)

    expert_masks = {
        key: np.flatnonzero((families == key[0]) & (sides == key[1]))
        for key in bundle.experts
    }

    for t in range(1, p.max_hold_bars + 1):
        times = np.full(n, t, dtype=np.int16)
        X = hazard_design_matrix_v41(out, bundle.feature_columns, times, p.max_hold_bars)
        ht = np.asarray(bundle.target_pooled.predict_proba(X)[:, 1], dtype=float)
        hs = np.asarray(bundle.stop_pooled.predict_proba(X)[:, 1], dtype=float)

        for key, idx in expert_masks.items():
            if len(idx) == 0:
                continue
            target_model, stop_model = bundle.experts[key]
            ht[idx] = np.asarray(target_model.predict_proba(X[idx])[:, 1], dtype=float)
            hs[idx] = np.asarray(stop_model.predict_proba(X[idx])[:, 1], dtype=float)

        ht = np.clip(ht, 0.0, p.hazard_cap)
        hs = np.clip(hs, 0.0, p.hazard_cap)
        total = ht + hs
        over = total > p.hazard_cap
        if np.any(over):
            scale = p.hazard_cap / total[over]
            ht[over] *= scale
            hs[over] *= scale

        p_target += survival * ht
        p_stop += survival * hs
        expected_duration += survival
        survival *= np.maximum(0.0, 1.0 - ht - hs)

    Xe = event_design_matrix_v41(out, bundle.feature_columns)
    timeout_r = np.asarray(bundle.timeout_pooled.predict(Xe), dtype=float)
    base_cost_r = pd.to_numeric(out["base_cost_r"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    expected_r = (
        p_target * (3.0 - base_cost_r)
        + p_stop * (-1.0 - base_cost_r)
        + survival * timeout_r
    )

    out["candidate"] = bundle.candidate
    out["seed"] = int(bundle.seed)
    out["p_target_v41"] = p_target
    out["p_stop_v41"] = p_stop
    out["p_timeout_v41"] = survival
    out["predicted_timeout_r_v41"] = timeout_r
    out["expected_duration_bars_v41"] = np.clip(expected_duration, 1.0, p.max_hold_bars)
    out["expected_r_v41"] = expected_r
    out["expert_used_v41"] = [int((str(f), int(s)) in bundle.experts) for f, s in zip(families, sides)]
    return out
