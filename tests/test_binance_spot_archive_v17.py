from __future__ import annotations

from io import BytesIO
from zipfile import ZipFile

import pandas as pd

from research_bot.binance_spot_archive import load_monthly_spot_archives


def test_loader_supports_mixed_millisecond_and_microsecond_archives(tmp_path):
    rows = [
        [1577836800000, 1, 2, 0.5, 1.5, 10, 0, 0, 1, 0, 0, 0],
        [1735689600000000, 2, 3, 1, 2.5, 11, 0, 0, 1, 0, 0, 0],
    ]
    path = tmp_path / "BTCUSDT-4h-2020-01.zip"
    payload = BytesIO()
    with ZipFile(payload, "w") as archive:
        archive.writestr("BTCUSDT-4h-2020-01.csv", "\n".join(",".join(map(str, row)) for row in rows))
    path.write_bytes(payload.getvalue())
    frame = load_monthly_spot_archives(tmp_path, "BTCUSDT")
    assert len(frame) == 2
    assert [(x.year, x.month, x.day) for x in frame["timestamp"]] == [(2020, 1, 1), (2025, 1, 1)]
    assert frame["timestamp"].is_monotonic_increasing
