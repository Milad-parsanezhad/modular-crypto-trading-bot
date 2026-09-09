from __future__ import annotations

import numpy as np
import pandas as pd

from research_bot.microstructure_v13 import diagnostic_correlations


def test_diagnostic_correlations_are_fail_closed_for_short_data():
    frame = pd.DataFrame({
        "future_spot_return_1h": [0.01, -0.01, 0.005],
        "true_spot_perp_basis": [0.001, 0.002, 0.0015],
    })
    out = diagnostic_correlations(frame)
    assert out["true_spot_perp_basis"] is None


def test_diagnostic_correlations_use_only_present_features():
    n = 80
    x = np.linspace(-1, 1, n)
    frame = pd.DataFrame({
        "future_spot_return_1h": x * 0.01,
        "true_spot_perp_basis": x,
        "basis_change_1": np.r_[np.nan, np.diff(x)],
        "basis_z": x,
        "spot_flow_imbalance": x[::-1],
        "futures_flow_imbalance": x,
        "flow_spread": x - x[::-1],
    })
    out = diagnostic_correlations(frame)
    assert out["true_spot_perp_basis"] is not None
    assert out["true_spot_perp_basis"] > 0.9
    assert out["oi_delta_1"] is None
