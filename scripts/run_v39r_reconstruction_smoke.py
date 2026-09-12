from __future__ import annotations

"""Deterministic smoke evaluation for the v0.39R canonical reconstruction.

This script validates architecture integrity on synthetic OHLCV only. It is not
an empirical performance result and must not be used for promotion decisions.
"""

import json

import numpy as np
import pandas as pd

from research_bot.canonical_strategy_v39r import (
    canonical_manifest,
    canonical_score,
    reconstruction_metadata,
    risk_weight_from_stop,
)


def synthetic_ohlcv(n: int = 1200) -> pd.DataFrame:
    rng = np.random.default_rng(314)
    ts = pd.date_range("2023-01-01", periods=n, freq="4h", tz="UTC")
    regime = np.where(np.arange(n) < n // 2, 0.018, -0.006)
    innovation = rng.normal(regime, 0.70, n)
    close = 100.0 + np.cumsum(innovation)
    close = np.maximum(close, 5.0)
    open_ = np.r_[close[0], close[:-1]] + rng.normal(0.0, 0.12, n)
    pad = np.abs(rng.normal(0.70, 0.16, n)) + 0.10
    high = np.maximum(open_, close) + pad
    low = np.minimum(open_, close) - pad
    volume = 1000.0 + rng.lognormal(4.5, 0.25, n)
    return pd.DataFrame(
        {"timestamp": ts, "open": open_, "high": high, "low": low, "close": close, "volume": volume}
    )


def main() -> None:
    frame = synthetic_ohlcv()
    scored = canonical_score(frame)
    weights = risk_weight_from_stop(scored)
    summary = {
        **reconstruction_metadata(),
        "rows": int(len(scored)),
        "long_decisions": int((scored["canonical_direction"] == 1).sum()),
        "short_decisions": int((scored["canonical_direction"] == -1).sum()),
        "flat_decisions": int((scored["canonical_direction"] == 0).sum()),
        "conflicts": int(scored["canonical_conflict"].sum()),
        "max_research_weight": float(weights.max()),
        "manifest_components": int(len(canonical_manifest())),
        "test_scope": "synthetic_architecture_smoke_only",
        "performance_claim": False,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
