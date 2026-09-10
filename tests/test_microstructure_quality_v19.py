import pandas as pd

from research_bot.microstructure_quality_v19 import V19QualityGateConfig, evaluate_quality_gate


START = pd.Timestamp("2026-09-10T08:00:00Z")


def _snapshot(i: int, ts: pd.Timestamp, authorized: bool = True) -> dict:
    def row(symbol: str):
        return {
            "symbol": symbol,
            "feature_authorized": authorized,
            "accepted_venues": 3 if authorized else 1,
            "venue_clock_skew_seconds": 5.0 if authorized else 120.0,
            "venues": [
                {"venue": "coinex", "trade_staleness_seconds": 2.0},
                {"venue": "okx", "trade_staleness_seconds": 3.0},
                {"venue": "kucoin", "trade_staleness_seconds": 4.0},
            ] if authorized else [],
        }
    return {
        "snapshot_sha256": f"hash-{i}",
        "generated_at": ts.isoformat(),
        "symbols": [row("BTC/USDT"), row("ETH/USDT")],
        "provider_failures": [],
    }


def test_gate_q_fails_closed_before_elapsed_week():
    snaps = [_snapshot(i, START + pd.Timedelta(minutes=30 * (i + 1))) for i in range(10)]
    out = evaluate_quality_gate(
        snaps,
        pilot_start_at=START.isoformat(),
        as_of=(START + pd.Timedelta(hours=5)).isoformat(),
    )
    assert out["decision"] == "GATE_Q_NOT_MATURE"
    assert out["feature_predictive_modeling_authorized"] is False
    assert "INSUFFICIENT_ELAPSED_TIME" in out["reasons"]
    assert out["live_execution_authorized"] is False


def test_gate_q_passes_only_after_336_good_measurement_opportunities():
    snaps = [_snapshot(i, START + pd.Timedelta(minutes=30 * (i + 1))) for i in range(336)]
    out = evaluate_quality_gate(
        snaps,
        pilot_start_at=START.isoformat(),
        as_of=(START + pd.Timedelta(hours=168)).isoformat(),
    )
    assert out["expected_measurement_opportunities"] == 336
    assert out["unique_measurement_slots"] == 336
    assert out["symbols"]["BTC/USDT"]["authorized_coverage_ratio"] == 1.0
    assert out["symbols"]["ETH/USDT"]["authorized_coverage_ratio"] == 1.0
    assert out["decision"] == "GATE_Q_PASSED_FEATURE_FREEZE_ALLOWED"
    assert out["feature_predictive_modeling_authorized"] is True
    assert out["paper_strategy_replacement_authorized"] is False
    assert out["live_execution_authorized"] is False


def test_manual_duplicate_hash_cannot_inflate_coverage():
    snap = _snapshot(1, START + pd.Timedelta(minutes=30))
    out = evaluate_quality_gate(
        [snap, dict(snap), dict(snap)],
        pilot_start_at=START.isoformat(),
        as_of=(START + pd.Timedelta(hours=168)).isoformat(),
    )
    assert out["unique_snapshot_hashes"] == 1
    assert out["unique_measurement_slots"] == 1
    assert out["decision"] == "GATE_Q_NOT_MATURE"
    assert any("AUTHORIZED_COVERAGE_BELOW_TARGET" in x for x in out["reasons"])
