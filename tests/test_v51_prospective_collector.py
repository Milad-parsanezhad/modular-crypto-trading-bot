import pandas as pd
import pytest

from research_bot.prospective_collector_v51 import (
    ALLOWED_ASSETS_V51,
    ALLOWED_VENUES_V51,
    CALENDAR_COMMIT_V51,
    EVIDENCE_INTEGRITY_ADDENDUM_COMMIT_V51,
    MAX_CAPTURE_LAG_MINUTES_V51,
    PREDICTOR_IDENTITY_AMENDMENT_COMMIT_V51,
    PREDICTOR_IDENTITY_POLICY_V51,
    PREREGISTRATION_COMMIT_V51,
    PROSPECTIVE_END_V51,
    PROSPECTIVE_START_V51,
    RAW_COLUMNS,
    REJECTED_IDENTITY_DRAFT_COMMIT_V51,
    closed_rows_from_ccxt,
    collect,
    merge_raw_evidence,
)


def _ms(ts: str) -> int:
    return int(pd.Timestamp(ts, tz="UTC").timestamp() * 1000)


def test_frozen_governance_constants_are_exact():
    assert PREREGISTRATION_COMMIT_V51 == "d8ee4576aaf55750dd5910cc0d3b2efcbba3f5b2"
    assert CALENDAR_COMMIT_V51 == "6a8fa49de2d1befa9c17049aa61e13da20a028eb"
    assert PREDICTOR_IDENTITY_AMENDMENT_COMMIT_V51 == "4838c0d98c408d358929b0767676a5ac024bd8dd"
    assert EVIDENCE_INTEGRITY_ADDENDUM_COMMIT_V51 == "da5dcf5cd3568a92c75871016f000917c9380955"
    assert PREDICTOR_IDENTITY_POLICY_V51 == "V47_C1_LATEST_CANONICAL_FOLD_PER_ASSET"
    assert REJECTED_IDENTITY_DRAFT_COMMIT_V51 == "095814ae55f49714299cf6b8c4908427c92bc6f9"
    assert MAX_CAPTURE_LAG_MINUTES_V51 == 60.0
    assert PROSPECTIVE_START_V51 == pd.Timestamp("2026-09-13T12:00:00Z")
    assert PROSPECTIVE_START_V51.hour % 4 == 0
    assert PROSPECTIVE_END_V51 == PROSPECTIVE_START_V51 + pd.Timedelta(days=150)
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
    assert out[0]["prospective_eligible_v51"] is False


def test_boundary_bar_is_eligible_when_first_seen_within_one_hour():
    rows = [[_ms("2026-09-13 08:00:00"), 105, 115, 95, 110, 13]]
    out = closed_rows_from_ccxt(
        rows,
        venue="okx",
        asset="ETH",
        captured_at=pd.Timestamp("2026-09-13T12:05:00Z"),
    )
    assert len(out) == 1
    assert pd.Timestamp(out[0]["bar_close_time"]) == PROSPECTIVE_START_V51
    assert out[0]["capture_lag_minutes"] == 5.0
    assert out[0]["prospective_eligible_v51"] is True


def test_late_backfill_is_retained_but_never_upgraded_to_prospective():
    rows = [[_ms("2026-09-13 08:00:00"), 105, 115, 95, 110, 13]]
    late = closed_rows_from_ccxt(
        rows,
        venue="kucoin",
        asset="SOL",
        captured_at=pd.Timestamp("2026-09-13T14:00:01Z"),
    )[0]
    assert late["capture_lag_minutes"] > MAX_CAPTURE_LAG_MINUTES_V51
    assert late["prospective_eligible_v51"] is False

    old = pd.DataFrame([late], columns=RAW_COLUMNS)
    timely_refetch = old.copy()
    timely_refetch["first_seen_at"] = "2026-09-13T12:10:00+00:00"
    timely_refetch["capture_lag_minutes"] = 10.0
    timely_refetch["prospective_eligible_v51"] = True
    merged = merge_raw_evidence(old, timely_refetch)
    assert len(merged) == 1
    assert merged.iloc[0]["first_seen_at"] == late["first_seen_at"]
    assert bool(merged.iloc[0]["prospective_eligible_v51"]) is False


def test_collect_fails_closed_before_repaired_start(tmp_path):
    with pytest.raises(RuntimeError, match="before amended prospective start"):
        collect(
            tmp_path,
            run_id="prestart",
            captured_at=PROSPECTIVE_START_V51 - pd.Timedelta(seconds=1),
        )


def test_forbidden_venue_fails_closed():
    with pytest.raises(ValueError, match="forbidden"):
        closed_rows_from_ccxt(
            [[_ms("2026-09-13 04:00:00"), 100, 110, 90, 105, 12]],
            venue="kraken",
            asset="BTC",
            captured_at=pd.Timestamp("2026-09-13T13:00:00Z"),
        )


def test_nonfinite_or_impossible_ohlcv_fails_closed():
    with pytest.raises(ValueError):
        closed_rows_from_ccxt(
            [[_ms("2026-09-13 04:00:00"), 100, float("nan"), 90, 105, 12]],
            venue="coinex",
            asset="BTC",
            captured_at=pd.Timestamp("2026-09-13T13:00:00Z"),
        )
    with pytest.raises(ValueError):
        closed_rows_from_ccxt(
            [[_ms("2026-09-13 04:00:00"), 100, 99, 90, 105, 12]],
            venue="coinex",
            asset="BTC",
            captured_at=pd.Timestamp("2026-09-13T13:00:00Z"),
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
