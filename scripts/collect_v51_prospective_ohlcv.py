from __future__ import annotations

"""Research-only prospective OHLCV evidence collector for v0.51.

This collector does not score candidates, make arbitration decisions, submit
orders, or enable PAPER/LIVE execution. It only persists public, closed 4-hour
OHLCV bars for the frozen v0.51 venues/assets so later causal reconstruction
can be audited from raw evidence.

The predictor-identity amendment moved the admissible prospective boundary from
08:00 UTC to the first clean 4h boundary after a unique production scoring
object could be frozen: 2026-09-13T12:00:00Z. Pre-12:00 observations are
engineering context only and are excluded from the admissible window.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import ccxt
import numpy as np
import pandas as pd

PREREGISTRATION_COMMIT_V51 = "d8ee4576aaf55750dd5910cc0d3b2efcbba3f5b2"
CALENDAR_COMMIT_V51 = "6a8fa49de2d1befa9c17049aa61e13da20a028eb"
PREDICTOR_IDENTITY_AMENDMENT = "V47_C1_LATEST_CANONICAL_FOLD_PER_ASSET"
PROSPECTIVE_START_V51 = pd.Timestamp("2026-09-13T12:00:00Z")
PROSPECTIVE_END_V51 = PROSPECTIVE_START_V51 + pd.Timedelta(days=150)
ALLOWED_VENUES_V51 = ("coinex", "okx", "kucoin")
ALLOWED_ASSETS_V51 = ("BTC", "ETH", "SOL", "XRP", "DOGE")
TIMEFRAME_V51 = "4h"
BAR_DELTA_V51 = pd.Timedelta(hours=4)
FETCH_LIMIT_V51 = 300

RAW_COLUMNS = [
    "first_seen_at",
    "venue",
    "symbol",
    "bar_open_time",
    "bar_close_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "source",
]
RUN_COLUMNS = [
    "run_id",
    "captured_at",
    "venue",
    "symbol",
    "status",
    "bars_received",
    "closed_bars_received",
    "latest_closed_bar",
    "error_type",
    "error_message",
]
VALUE_COLUMNS = ["open", "high", "low", "close", "volume"]
IDENTITY_COLUMNS = ["venue", "symbol", "bar_open_time"]


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _iso(value) -> str:
    return pd.Timestamp(value).tz_convert("UTC").isoformat()


def _pair(asset: str) -> str:
    return f"{asset}/USDT"


def _canonical_symbol(asset: str) -> str:
    return f"{asset}USDT"


def _validate_numeric_ohlcv(row: Iterable[float]) -> tuple[float, float, float, float, float]:
    values = tuple(float(x) for x in row)
    if len(values) != 5 or not np.isfinite(np.asarray(values, dtype=float)).all():
        raise ValueError("non-finite OHLCV value")
    o, h, l, c, v = values
    if min(o, h, l, c) <= 0.0 or v < 0.0 or h < max(o, c, l) or l > min(o, c, h):
        raise ValueError("invalid OHLCV geometry")
    return values


def closed_rows_from_ccxt(
    rows: list[list[float]], *, venue: str, asset: str, captured_at: pd.Timestamp
) -> list[dict]:
    """Normalize only bars that were fully closed by ``captured_at``."""
    if venue not in ALLOWED_VENUES_V51 or asset not in ALLOWED_ASSETS_V51:
        raise ValueError("forbidden v0.51 venue/asset")
    captured_at = pd.Timestamp(captured_at)
    if captured_at.tzinfo is None:
        raise ValueError("captured_at must be timezone-aware")
    captured_at = captured_at.tz_convert("UTC")
    out: list[dict] = []
    for raw in rows:
        if len(raw) < 6:
            raise ValueError("malformed CCXT OHLCV row")
        opened = pd.to_datetime(int(raw[0]), unit="ms", utc=True)
        closed = opened + BAR_DELTA_V51
        if closed > captured_at:
            continue
        o, h, l, c, v = _validate_numeric_ohlcv(raw[1:6])
        out.append(
            {
                "first_seen_at": _iso(captured_at),
                "venue": venue,
                "symbol": _canonical_symbol(asset),
                "bar_open_time": _iso(opened),
                "bar_close_time": _iso(closed),
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "volume": v,
                "source": "ccxt_public_ohlcv",
            }
        )
    return out


def merge_raw_evidence(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    """Append new identities; fail closed if a previously closed bar is revised."""
    if existing.empty:
        return incoming.loc[:, RAW_COLUMNS].sort_values(IDENTITY_COLUMNS).reset_index(drop=True)
    old = existing.loc[:, RAW_COLUMNS].copy()
    new = incoming.loc[:, RAW_COLUMNS].copy()
    joined = old.merge(new, on=IDENTITY_COLUMNS, how="inner", suffixes=("_old", "_new"))
    for col in VALUE_COLUMNS:
        if not joined.empty and not np.allclose(
            joined[f"{col}_old"].to_numpy(dtype=float),
            joined[f"{col}_new"].to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-12,
            equal_nan=False,
        ):
            raise RuntimeError(f"closed-bar revision detected in {col}")
    combined = pd.concat([old, new], ignore_index=True)
    combined = combined.drop_duplicates(IDENTITY_COLUMNS, keep="first")
    return combined.loc[:, RAW_COLUMNS].sort_values(IDENTITY_COLUMNS).reset_index(drop=True)


def _exchange(venue: str):
    if venue == "kraken" or venue not in ALLOWED_VENUES_V51:
        raise ValueError("Kraken/forbidden venue requested")
    cls = getattr(ccxt, venue, None)
    if cls is None:
        raise RuntimeError(f"CCXT exchange unavailable: {venue}")
    return cls({"enableRateLimit": True, "timeout": 30_000})


def collect(output_dir: Path, *, run_id: str, captured_at: pd.Timestamp) -> dict:
    captured_at = pd.Timestamp(captured_at)
    if captured_at.tzinfo is None:
        raise ValueError("captured_at must be timezone-aware")
    captured_at = captured_at.tz_convert("UTC")
    if captured_at < PROSPECTIVE_START_V51:
        raise RuntimeError("v0.51 collector cannot run before amended prospective start")
    if captured_at >= PROSPECTIVE_END_V51:
        raise RuntimeError("v0.51 collector is outside frozen 150-day window")

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw_4h_ohlcv_v51.csv"
    runs_path = output_dir / "collection_runs_v51.csv"
    manifest_path = output_dir / "collection_manifest_v51.json"

    incoming_rows: list[dict] = []
    audit_rows: list[dict] = []
    for venue in ALLOWED_VENUES_V51:
        ex = _exchange(venue)
        try:
            markets = ex.load_markets()
        except Exception as exc:
            for asset in ALLOWED_ASSETS_V51:
                audit_rows.append(
                    {
                        "run_id": run_id,
                        "captured_at": _iso(captured_at),
                        "venue": venue,
                        "symbol": _canonical_symbol(asset),
                        "status": "LOAD_MARKETS_ERROR",
                        "bars_received": 0,
                        "closed_bars_received": 0,
                        "latest_closed_bar": "",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc)[:500],
                    }
                )
            continue

        for asset in ALLOWED_ASSETS_V51:
            pair = _pair(asset)
            symbol = _canonical_symbol(asset)
            if pair not in markets:
                audit_rows.append(
                    {
                        "run_id": run_id,
                        "captured_at": _iso(captured_at),
                        "venue": venue,
                        "symbol": symbol,
                        "status": "MISSING_MARKET",
                        "bars_received": 0,
                        "closed_bars_received": 0,
                        "latest_closed_bar": "",
                        "error_type": "",
                        "error_message": "",
                    }
                )
                continue
            try:
                raw = ex.fetch_ohlcv(pair, timeframe=TIMEFRAME_V51, limit=FETCH_LIMIT_V51)
                normalized = closed_rows_from_ccxt(
                    raw, venue=venue, asset=asset, captured_at=captured_at
                )
                incoming_rows.extend(normalized)
                latest = max((r["bar_close_time"] for r in normalized), default="")
                audit_rows.append(
                    {
                        "run_id": run_id,
                        "captured_at": _iso(captured_at),
                        "venue": venue,
                        "symbol": symbol,
                        "status": "OK" if normalized else "NO_CLOSED_BARS",
                        "bars_received": int(len(raw)),
                        "closed_bars_received": int(len(normalized)),
                        "latest_closed_bar": latest,
                        "error_type": "",
                        "error_message": "",
                    }
                )
            except Exception as exc:
                audit_rows.append(
                    {
                        "run_id": run_id,
                        "captured_at": _iso(captured_at),
                        "venue": venue,
                        "symbol": symbol,
                        "status": "FETCH_ERROR",
                        "bars_received": 0,
                        "closed_bars_received": 0,
                        "latest_closed_bar": "",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc)[:500],
                    }
                )
        close = getattr(ex, "close", None)
        if callable(close):
            close()

    incoming = pd.DataFrame(incoming_rows, columns=RAW_COLUMNS)
    existing = pd.read_csv(raw_path) if raw_path.exists() else pd.DataFrame(columns=RAW_COLUMNS)
    if incoming.empty and existing.empty:
        merged = pd.DataFrame(columns=RAW_COLUMNS)
    else:
        merged = merge_raw_evidence(existing, incoming)
    merged.to_csv(raw_path, index=False)

    current_audit = pd.DataFrame(audit_rows, columns=RUN_COLUMNS)
    previous_audit = pd.read_csv(runs_path) if runs_path.exists() else pd.DataFrame(columns=RUN_COLUMNS)
    audit = pd.concat([previous_audit, current_audit], ignore_index=True)
    audit = audit.drop_duplicates(["run_id", "venue", "symbol"], keep="first")
    audit.to_csv(runs_path, index=False)

    ok_count = int(current_audit["status"].eq("OK").sum())
    total_expected = len(ALLOWED_VENUES_V51) * len(ALLOWED_ASSETS_V51)
    health = "OK" if ok_count == total_expected else ("PARTIAL" if ok_count > 0 else "FAILED")
    prospective_raw = 0
    if not merged.empty:
        closes = pd.to_datetime(merged["bar_close_time"], utc=True)
        prospective_raw = int(((closes >= PROSPECTIVE_START_V51) & (closes < PROSPECTIVE_END_V51)).sum())

    manifest = {
        "experiment": "v0.51",
        "role": "raw_public_market_evidence_only",
        "preregistration_commit": PREREGISTRATION_COMMIT_V51,
        "calendar_commit": CALENDAR_COMMIT_V51,
        "predictor_identity_amendment": PREDICTOR_IDENTITY_AMENDMENT,
        "prospective_start": _iso(PROSPECTIVE_START_V51),
        "prospective_end": _iso(PROSPECTIVE_END_V51),
        "captured_at": _iso(captured_at),
        "run_id": run_id,
        "timeframe": TIMEFRAME_V51,
        "venues": list(ALLOWED_VENUES_V51),
        "assets": list(ALLOWED_ASSETS_V51),
        "kraken_touched": False,
        "paper_execution": False,
        "live_execution": False,
        "scoring_performed": False,
        "arbitration_performed": False,
        "collection_health": health,
        "successful_series_this_run": ok_count,
        "expected_series_per_run": total_expected,
        "raw_rows_total": int(len(merged)),
        "raw_rows_in_prospective_window": prospective_raw,
        "raw_csv_sha256": _sha256(raw_path),
        "collection_runs_csv_sha256": _sha256(runs_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="evidence/v51")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--captured-at", default=None)
    args = parser.parse_args()
    now = pd.Timestamp.now(tz="UTC") if args.captured_at is None else pd.Timestamp(args.captured_at)
    manifest = collect(Path(args.output_dir), run_id=str(args.run_id), captured_at=now)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
