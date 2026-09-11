from __future__ import annotations

import pandas as pd
import pytest

from research_bot.failure_attribution_v32 import attribute_holdout_failure, validate_v31_source


def test_v31_source_must_be_consumed_rejected_holdout():
    with pytest.raises(ValueError):
        validate_v31_source({"decision": "PASS", "holdout_used": True, "live_execution_authorized": False})


def test_attribution_is_diagnostic_only_and_finds_negative_assets():
    ledger = pd.DataFrame({
        "entry_time": pd.to_datetime(["2025-01-01","2025-01-02","2025-04-01","2025-04-02"], utc=True),
        "symbol": ["AAA/USDT","AAA/USDT","BBB/USDT","BBB/USDT"],
        "side": [1,1,-1,-1],
        "executed_v25": [True,True,True,True],
        "account_return_v25": [0.01,0.01,-0.02,-0.01],
        "r_multiple": [1.0,1.0,-1.0,-0.5],
    })
    out = attribute_holdout_failure(ledger, {"positive_asset_fraction": 0.5, "holdout_failures": ["BREADTH","BLOCK_CI"]})
    assert out["decision"] == "V32_FAILURE_ATTRIBUTED_NO_PROMOTION"
    assert out["negative_assets"] == ["BBB/USDT"]
    assert out["observed_positive_asset_fraction"] == 0.5
    assert out["kucoin_consumed"] is True
    assert out["forward_paper_candidate_authorized"] is False
    assert out["live_execution_authorized"] is False
