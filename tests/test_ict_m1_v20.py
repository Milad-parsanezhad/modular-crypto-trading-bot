from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import json
import os
from pathlib import Path
import subprocess
import sys

from research_bot.ict_m1_v20 import (
    IctM1Config,
    REPORTING_PERIODS,
    build_ict_m1_features,
    evaluate_ict_m1_variants,
    extract_ict_m1_setups,
    simulate_ict_m1_trades,
)


def bullish_example() -> pd.DataFrame:
    """Deterministic sweep -> close-confirmed MSB -> origin return -> target."""

    rows = [
        (10.0, 10.2, 9.8, 10.0),
        (10.0, 10.8, 9.9, 10.6),
        (10.6, 11.0, 10.4, 10.8),  # swing high, confirmed on bar 3
        (10.8, 10.9, 10.0, 10.2),
        (10.2, 10.4, 9.5, 9.8),  # swing low, confirmed on bar 5
        (9.8, 10.0, 9.8, 9.9),
        (9.9, 10.3, 9.3, 9.7),  # sweep below 9.5, close back above
        (9.7, 10.5, 9.6, 10.4),
        (10.4, 11.3, 10.2, 11.2),  # close-confirmed break above 11.0
        (10.2, 10.6, 9.75, 10.0),  # return to sweep-candle body midpoint 9.8
        (10.0, 11.5, 9.95, 11.4),
        (11.4, 11.7, 11.2, 11.5),
    ]
    frame = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    frame.insert(0, "timestamp", pd.date_range("2025-01-01", periods=len(frame), freq="4h", tz="UTC"))
    frame["volume"] = np.linspace(1_000.0, 1_500.0, len(frame))
    return frame


def config(**kwargs) -> IctM1Config:
    values = {
        "atr_window": 3,
        "min_displacement_atr": 0.0,
        "max_confirmation_bars": 4,
        "pending_bars": 3,
        "max_holding_bars": 5,
        "stop_buffer_atr": 0.0,
        "ichimoku_gate": "off",
    }
    values.update(kwargs)
    return IctM1Config(**values)


def test_swing_confirmation_and_sweep_are_causal():
    features = build_ict_m1_features(bullish_example(), config())
    assert np.isnan(features.loc[4, "last_confirmed_swing_low"])
    assert features.loc[6, "last_confirmed_swing_low"] == pytest.approx(9.5)
    assert bool(features.loc[6, "bullish_sweep"])
    assert not bool(features.loc[5, "bullish_sweep"])


def test_msb_requires_close_not_only_wick():
    frame = bullish_example()
    frame.loc[8, "close"] = 10.9
    frame.loc[8, "high"] = 11.3
    cfg = config(max_confirmation_bars=2)
    setups = extract_ict_m1_setups(build_ict_m1_features(frame, cfg), cfg)
    assert setups.empty


def test_sweep_requires_close_back_inside_reference_level():
    frame = bullish_example()
    frame.loc[6, "close"] = 9.4
    features = build_ict_m1_features(frame, config())
    assert not bool(features.loc[6, "bullish_sweep"])
    assert extract_ict_m1_setups(features, config()).empty


def test_setup_and_trade_use_only_post_confirmation_bars():
    cfg = config()
    features = build_ict_m1_features(bullish_example(), cfg)
    setups = extract_ict_m1_setups(features, cfg)
    assert len(setups) == 1
    setup = setups.iloc[0]
    assert setup["direction"] == "LONG"
    assert setup["sweep_index"] == 6
    assert setup["confirm_index"] == 8
    trades = simulate_ict_m1_trades(features, setups, cfg, variant="sweep_origin")
    assert len(trades) == 1
    trade = trades.iloc[0]
    assert trade["fill_index"] == 9
    assert trade["exit_index"] >= trade["fill_index"]
    assert trade["outcome"] == "TARGET"
    assert trade["account_return"] > 0


def test_same_bar_ambiguity_is_resolved_stop_first():
    frame = bullish_example()
    frame.loc[9, ["high", "low"]] = [12.0, 9.0]
    cfg = config()
    features = build_ict_m1_features(frame, cfg)
    setups = extract_ict_m1_setups(features, cfg)
    trade = simulate_ict_m1_trades(features, setups, cfg, variant="sweep_origin").iloc[0]
    assert trade["outcome"] == "STOP"
    assert trade["account_return"] < 0


def test_future_mutation_does_not_change_historical_features_or_setups():
    cfg = config()
    original = bullish_example()
    changed = original.copy()
    changed.loc[10:, ["open", "high", "low", "close"]] *= 3.0
    a = build_ict_m1_features(original, cfg)
    b = build_ict_m1_features(changed, cfg)
    cols = [
        "last_confirmed_swing_high",
        "last_confirmed_swing_low",
        "bullish_sweep",
        "bearish_sweep",
        "atr",
    ]
    pd.testing.assert_frame_equal(a.loc[:9, cols], b.loc[:9, cols])
    sa = extract_ict_m1_setups(a, cfg)
    sb = extract_ict_m1_setups(b, cfg)
    pd.testing.assert_frame_equal(sa[sa.confirm_index <= 9].reset_index(drop=True), sb[sb.confirm_index <= 9].reset_index(drop=True))


def test_evaluation_reports_all_registered_origin_variants_and_costs():
    cfg = config()
    summary, trades, _ = evaluate_ict_m1_variants(bullish_example(), cfg)
    assert summary["variant"].tolist() == ["sweep_origin", "opposing_candle", "fvg"]
    assert set(summary["status"]) <= {"NO_TRADES", "EVALUATED"}
    assert summary["live_execution"].eq(False).all()
    if not trades.empty:
        assert trades["fee_bps"].eq(cfg.fee_bps).all()
        assert trades["slippage_bps"].eq(cfg.slippage_bps).all()
        assert trades["risk_fraction"].eq(cfg.risk_per_trade).all()


def test_invalid_ohlcv_fails_closed():
    frame = bullish_example()
    frame.loc[3, "high"] = frame.loc[3, "low"] - 1.0
    with pytest.raises(ValueError, match="invalid OHLC"):
        build_ict_m1_features(frame, config())


def test_reporting_period_boundaries_are_explicit_utc():
    for _, start, end in REPORTING_PERIODS:
        assert pd.Timestamp(start).tzinfo is not None
        assert pd.Timestamp(end).tzinfo is not None


def test_cli_writes_reproducible_research_only_artifacts(tmp_path):
    csv_path = tmp_path / "ohlcv.csv"
    output = tmp_path / "artifacts"
    bullish_example().to_csv(csv_path, index=False)
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root)
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "run_ict_m1_lab_v20.py"),
            "--input-csv",
            str(csv_path),
            "--bars",
            "100",
            "--ichimoku-gate",
            "off",
            "--output-dir",
            str(output),
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["dataset_sha256"]
    assert manifest["promotion_decision"].startswith("RESEARCH_ONLY")
    assert manifest["live_execution"] is False
    assert (output / "summary.csv").exists()
    assert (output / "period_summary.csv").exists()
