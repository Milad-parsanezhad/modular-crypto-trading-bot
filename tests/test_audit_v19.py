import numpy as np
import pandas as pd

from research_bot.audit_v19 import (
    V19AuditConfig,
    _backtest_long_only,
    audit_cost_aware_conversion,
    audit_v11_regime_replication_construction,
    expanding_crossfit_residual_mad,
    stable_frame_fingerprint,
)
from research_bot.features import FEATURE_COLUMNS, add_features


def _bars(seed: int, n: int = 1200, phase: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    drift = 0.00015 + 0.00055 * np.sin(np.arange(n) / 45.0 + phase)
    shocks = rng.normal(0.0, 0.007, n)
    close = 100.0 * np.exp(np.cumsum(drift + shocks))
    open_ = np.r_[close[0], close[:-1]]
    wick = np.abs(rng.normal(0.0035, 0.0010, n))
    high = np.maximum(open_, close) * (1.0 + wick)
    low = np.minimum(open_, close) * (1.0 - wick)
    volume = rng.lognormal(9.0, 0.45, n)
    return pd.DataFrame({
        "timestamp": ts,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


def test_stable_frame_fingerprint_is_deterministic():
    df = _bars(1, 100)
    assert stable_frame_fingerprint(df) == stable_frame_fingerprint(df.copy())
    changed = df.copy(); changed.loc[0, "close"] += 1.0
    assert stable_frame_fingerprint(df) != stable_frame_fingerprint(changed)


def test_terminal_liquidation_cost_is_charged():
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=3, freq="4h", tz="UTC"),
        "future_return": [0.01, 0.01, 0.01],
    })
    pos = pd.Series([0.0, 1.0, 1.0])
    bt = _backtest_long_only(frame, pos, 10.0, force_terminal_close=True)
    # entry 1 + terminal exit 1 = turnover 2
    assert abs(bt["turnover"].sum() - 2.0) < 1e-12
    assert abs(bt["cost"].sum() - 0.002) < 1e-12
    assert bt.attrs["terminal_close_turnover"] == 1.0


def test_crossfit_uncertainty_uses_oof_rows():
    x = add_features(_bars(2, 1000))
    x["future_return"] = x["close"].shift(-1) / x["close"] - 1.0
    x = x.dropna(subset=["future_return"]).reset_index(drop=True)
    features = [c for c in FEATURE_COLUMNS if c in x]
    cfg = V19AuditConfig(calibration_min_train=250, calibration_folds=3)
    out = expanding_crossfit_residual_mad(x.iloc[:700].copy(), features, cfg)
    assert out["oof_residual_count"] > 50
    assert out["oof_residual_mad"] > 0
    assert len(out["folds"]) >= 2
    for fold in out["folds"]:
        assert fold["train_rows"] < 700
        assert fold["validation_rows"] > 0


def test_cost_audit_is_spent_holdout_fail_closed():
    cfg = V19AuditConfig(
        train_fraction=0.70,
        calibration_min_train=220,
        bootstrap_resamples=100,
        min_holdout_periods=100,
    )
    out = audit_cost_aware_conversion({"BTC/USDT": _bars(3), "ETH/USDT": _bars(4)}, cfg)
    assert out["evidence_status"] == "SPENT_HOLDOUT_AUDIT_ONLY_NOT_NEW_EVIDENCE"
    assert out["promotion_authorized"] is False
    assert "UNCERTAINTY_MAD_WAS_IN_SAMPLE_NOW_EXPANDING_OOF" in out["bugs_corrected"]
    assert all("dataset_fingerprint" in row for row in out["per_symbol"])


def test_regime_audit_restores_cross_sectional_geometry():
    symbols = {
        f"ASSET{i}/USDT": _bars(100 + i, 1000, phase=i / 7.0)
        for i in range(8)
    }
    cfg = V19AuditConfig(
        min_cross_section_assets=6,
        bootstrap_resamples=100,
        min_holdout_periods=100,
    )
    out = audit_v11_regime_replication_construction(symbols, cfg)
    assert out["evidence_status"] == "RETROSPECTIVE_CONSTRUCTION_AUDIT_NOT_FRESH_REPLICATION"
    assert out["promotion_authorized"] is False
    assert out["panel_timestamps"] > 100
    assert len(out["symbols"]) == 8
    assert "V18B_BINARY_PER_ASSET_SIGNAL_DID_NOT_MATCH_V11_CROSS_SECTIONAL_TOP_QUARTILE" in out["bugs_corrected"]
