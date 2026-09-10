import pandas as pd

from research_bot.forward_microstructure_v19 import _coerce_trade_timestamp


def test_trade_timestamp_parser_handles_datetime_and_unix_units():
    expected = pd.Timestamp("2026-09-10T06:00:00Z")
    series = pd.Series([
        expected,
        1789020000,
        1789020000000,
        1789020000000000,
        1789020000000000000,
        "2026-09-10T06:00:00Z",
    ])
    parsed = _coerce_trade_timestamp(series)
    assert parsed.notna().all()
    assert all(x == expected for x in parsed)
