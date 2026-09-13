import pandas as pd
import pytest

from scripts.collect_v51_prospective_ohlcv import (
    ALLOWED_ASSETS_V51,
    ALLOWED_VENUES_V51,
    CALENDAR_COMMIT_V51,
    PREREGISTRATION_COMMIT_V51,
    PROSPECTIVE_START_V51,
    RAW_COLUMNS,
    closed_rows_from_ccxt,
    merge_raw_evidence,
)


def _ms(ts: str) -> int:
    return int(pd.Timestamp(ts, tz="UTC").timestamp() * 1000)


def test_frozen_governance_constants_are_exact():
    assert PREREGISTRATION_COMMIT_V51 == "d8ee4576aaf55750dd5910cc0d3b2efcbba3f5b2"
    assert CALENDAR_COMMIT_V51 == "6a8fa49de2d1befa9c17049aa61e13da20a028eb"
    assert PROSPECTIVE_START_V51 == pd.Timestamp("2026-09-13T08:00:00Z")
    assert set(ALLOWED_VENUES_V51) == {"coinex", "okx", "kucoin"}
    assert set(ALLOWED_ASSETS_V51) == {"BTC", "ETH", "SOL", "XRP", "DOGE"}
    assert "kraken" not in ALLOWED_VENUES_V51


def test_collector_keeps_only_fully_closed_bars():
    rows = [
        [_ms("2026-09-13 04:00:00"), 100, 110, 90, 105, 12],
        [_ms("2026-09-13 08:00:00"), 105, 115, 95, 110, 13],
    ]
    out = closed_rows_from_ccxt(
        rows,
        venue="coinex",
        asset="BTC",
        captured_at=pd.Timestamp("2026-09-13T08:05:00Z"),
    )
    assert len(out) == 1
    assert out[0]["bar_open_time"] == "2026-09-13T04:00:00+00:00"
    assert out[0]["bar_close_time"] == "2026-09-13T08:00:00+00:00"
    assert out[0]["symbol"] == "BTCUSDT"


def test_forbidden_venue_fails_closed():
    with pytest.raises(ValueError, match="forbidden"):
        closed_rows_from_ccxt(
            [[_ms("2026-09-13 04:00:00"), 100, 110, 90, 105, 12]],
            venue="kraken",
            asset="BTC",
            captured_at=pd.Timestamp("2026-09-13T09:00:00Z"),
        )


def test_nonfinite_or_impossible_ohlcv_fails_closed():
    with pytest.raises(ValueError):
        closed_rows_from_ccxt(
            [[_ms("2026-09-13 04:00:00"), 100, float("nan"), 90, 105, 12]],
            venue="coinex",
            asset="BTC",
            captured_at=pd.Timestamp("2026-09-13T09:00:00Z"),
        )
    with pytest.raises(ValueError):
        closed_rows_from_ccxt(
            [[_ms("2026-09-13 04:00:00"), 100, 99, 90, 105, 12]],
            venue="coinex",
            asset="BTC",
            captured_at=pd.Timestamp("2026-09-13T09:00:00Z"),
        )


def test_merge_is_idempotent_and_preserves_first_seen():
    row = closed_rows_from_ccxt(
        [[_ms("2026-09-13 04:00:00"), 100, 110, 90, 105, 12]],
        venue="coinex",
        asset="BTC",
        captured_at=pd.Timestamp("2026-09-13T08:05:00Z"),
    )[0]
    first = pd.DataFrame([row], columns=RAW_COLUMNS)
    later = first.copy()
    later["first_seen_at"] = "2026-09-13T12:05:00+00:00"
    merged = merge_raw_evidence(first, later)
    assert len(merged) == 1
    assert merged.iloc[0]["first_seen_at"] == "2026-09-13T08:05:00+00:00"


def test_closed_bar_revision_fails_closed():
    row = closed_rows_from_ccxt(
        [[_ms("2026-09-13 04:00:00"), 100, 110, 90, 105, 12]],
        venue="coinex",
        asset="BTC",
        captured_at=pd.Timestamp("2026-09-13T08:05:00Z"),
    )[0]
    old = pd.DataFrame([row], columns=RAW_COLUMNS)
    changed = old.copy()
    changed.loc[0, "close"] = 106.0
    with pytest.raises(RuntimeError, match="revision"):
        merge_raw_evidence(old, changed)
