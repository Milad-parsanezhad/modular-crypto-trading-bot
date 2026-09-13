import pandas as pd
import pytest

from research_bot.overlap_arbitration_v51 import arbitrate_overlaps_v51


def _row(series_id, entry, exit_, score, *, net_r=0.0, selected=True):
    entry = pd.Timestamp(entry, tz="UTC")
    return {
        "venue": "coinex",
        "symbol": "BTCUSDT",
        "series_id": series_id,
        "signal_time": entry - pd.Timedelta(hours=4),
        "entry_time": entry,
        "exit_time": pd.Timestamp(exit_, tz="UTC"),
        "expected_r_v47": score,
        "selected_model": selected,
        "net_r": net_r,
        "outcome": "TARGET" if net_r > 0 else "STOP",
    }


def test_same_time_conflict_uses_highest_frozen_expected_r():
    rows = [
        _row("low", "2026-09-14", "2026-09-15", 0.1),
        _row("high", "2026-09-14", "2026-09-16", 0.4),
    ]
    out, audit = arbitrate_overlaps_v51(pd.DataFrame(rows))
    assert out["series_id"].tolist() == ["high"]
    assert audit.simultaneous_conflict_cohorts == 1
    assert audit.simultaneous_candidates == 2


def test_later_higher_score_cannot_preempt_active_trade():
    rows = [
        _row("first", "2026-09-14", "2026-09-16", 0.1),
        _row("later", "2026-09-15", "2026-09-17", 99.0),
    ]
    out, audit = arbitrate_overlaps_v51(pd.DataFrame(rows))
    assert out["series_id"].tolist() == ["first"]
    assert audit.blocked_later_candidates == 1


def test_realized_outcomes_cannot_change_arbitration():
    frame = pd.DataFrame([
        _row("a", "2026-09-14", "2026-09-15", 0.2, net_r=-1.0),
        _row("b", "2026-09-14", "2026-09-16", 0.3, net_r=3.0),
    ])
    before, _ = arbitrate_overlaps_v51(frame)
    mutated = frame.copy()
    mutated["net_r"] = [-500.0, 500.0]
    mutated["outcome"] = ["TARGET", "STOP"]
    after, _ = arbitrate_overlaps_v51(mutated)
    assert before["series_id"].tolist() == after["series_id"].tolist() == ["b"]


def test_result_is_input_order_invariant():
    frame = pd.DataFrame([
        _row("b", "2026-09-14", "2026-09-15", 0.2),
        _row("a", "2026-09-14", "2026-09-15", 0.2),
    ])
    one, _ = arbitrate_overlaps_v51(frame)
    two, _ = arbitrate_overlaps_v51(frame.iloc[::-1])
    assert one["series_id"].tolist() == two["series_id"].tolist() == ["a"]


def test_venue_and_symbol_slots_are_isolated():
    first = _row("coinex-btc", "2026-09-14", "2026-09-16", 0.2)
    other_venue = {**_row("okx-btc", "2026-09-15", "2026-09-17", 0.3), "venue": "okx"}
    other_symbol = {**_row("coinex-eth", "2026-09-15", "2026-09-17", 0.4), "symbol": "ETHUSDT"}
    out, audit = arbitrate_overlaps_v51(pd.DataFrame([first, other_venue, other_symbol]))
    assert set(out["series_id"]) == {"coinex-btc", "okx-btc", "coinex-eth"}
    assert audit.blocked_later_candidates == 0


@pytest.mark.parametrize("column,value", [
    ("expected_r_v47", float("nan")),
    ("expected_r_v47", -0.1),
    ("selected_model", False),
    ("selected_model", 1),
])
def test_invalid_or_nonadmitted_inputs_fail_closed(column, value):
    row = _row("a", "2026-09-14", "2026-09-15", 0.2)
    row[column] = value
    with pytest.raises(ValueError):
        arbitrate_overlaps_v51(pd.DataFrame([row]))


def test_duplicate_decision_identity_fails_closed():
    row = _row("a", "2026-09-14", "2026-09-15", 0.2)
    with pytest.raises(ValueError, match="identity"):
        arbitrate_overlaps_v51(pd.DataFrame([row, {**row, "net_r": 3.0}]))
