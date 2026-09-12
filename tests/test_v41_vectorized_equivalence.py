from __future__ import annotations

import numpy as np
import pandas as pd

from research_bot.event_competing_risk_v41 import (
    fit_competing_risk_bundle_v41,
    predict_competing_risks_v41,
)
from research_bot.event_competing_risk_vectorized_v41 import (
    predict_competing_risks_vectorized_v41,
)


def _events(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = pd.date_range("2025-01-01", periods=n, freq="4h", tz="UTC")
    f1 = rng.normal(size=n).astype("float32")
    f2 = rng.normal(size=n).astype("float32")
    latent = 0.8 * f1 - 0.25 * f2 + rng.normal(scale=0.8, size=n)
    outcome = np.where(latent > 0.5, "TARGET", np.where(latent < -0.4, "STOP", "TIME"))
    duration = rng.integers(1, 10, size=n)
    net_r = np.where(outcome == "TARGET", 2.9, np.where(outcome == "STOP", -1.1, 0.2 * latent - 0.08))
    return pd.DataFrame(
        {
            "series_id": "okx::BTC/USDT",
            "venue": "okx",
            "symbol": "BTC/USDT",
            "signal_time": t,
            "entry_time": t + pd.Timedelta(hours=4),
            "exit_time": t + pd.to_timedelta(duration * 4, unit="h"),
            "outcome": outcome,
            "net_r": net_r,
            "base_cost_r": 0.08,
            "side": np.where(np.arange(n) % 2, -1, 1),
            "regime": np.array(["trend", "range", "transition"])[np.arange(n) % 3],
            "event_family_v41": np.array(["ICT_MSS", "BROOKS_H2L2", "SMC_OB_RETEST"])[np.arange(n) % 3],
            "f1": f1,
            "f2": f2,
        }
    )


def test_vectorized_v41_prediction_is_protocol_equivalent() -> None:
    fit = _events(420, 11)
    target = _events(35, 12)
    bundle = fit_competing_risk_bundle_v41(
        "V41_LOGIT_CAUSE_SPECIFIC", fit, ["f1", "f2"], 314
    )
    scalar = predict_competing_risks_v41(bundle, target)
    vectorized = predict_competing_risks_vectorized_v41(bundle, target)
    for col in (
        "p_target_v41",
        "p_stop_v41",
        "p_timeout_v41",
        "predicted_timeout_r_v41",
        "expected_duration_bars_v41",
        "expected_r_v41",
    ):
        # Batched BLAS changes floating-point accumulation order by a few 1e-8;
        # this tolerance is far below any research threshold and was fixed from
        # the observed maximum implementation-only delta (3.57e-8).
        np.testing.assert_allclose(scalar[col], vectorized[col], rtol=1e-6, atol=1e-7)
