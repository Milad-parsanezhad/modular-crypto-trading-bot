from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

import pandas as pd
import psycopg

from research_bot.prospective_collector_v51 import collect


COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
COLLECTOR_VERSION = "v51-ops-2"


def row_hash(row: pd.Series) -> str:
    payload = {
        "venue": row["venue"],
        "symbol": row["symbol"],
        "timeframe": "4h",
        "bar_open_time": row["bar_open_time"],
        "bar_close_time": row["bar_close_time"],
        "open": float(row["open"]),
        "high": float(row["high"]),
        "low": float(row["low"]),
        "close": float(row["close"]),
        "volume": float(row["volume"]),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def manifest_hash(manifest: dict) -> str:
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def resolve_source_commit() -> str:
    candidates = (
        os.environ.get("V51_SOURCE_COMMIT"),
        os.environ.get("RAILWAY_GIT_COMMIT_SHA"),
        os.environ.get("GITHUB_SHA"),
    )
    for value in candidates:
        if value and COMMIT_RE.fullmatch(value.strip().lower()):
            return value.strip().lower()
    raise RuntimeError(
        "v0.51 provenance fail-closed: V51_SOURCE_COMMIT/Git SHA must be a real 40-character hex commit"
    )


def ineligibility_reason(row: pd.Series) -> str | None:
    eligible = str(row["prospective_eligible_v51"]).strip().lower() in {"true", "1"}
    if eligible:
        return None
    close = pd.Timestamp(row["bar_close_time"])
    first_seen = pd.Timestamp(row["first_seen_at"])
    start = pd.Timestamp("2026-09-13T12:00:00Z")
    end = pd.Timestamp("2027-02-10T12:00:00Z")
    if close < start:
        return "PRE_PROSPECTIVE_START"
    if close >= end:
        return "POST_PROSPECTIVE_END"
    lag_minutes = (first_seen - close).total_seconds() / 60.0
    if lag_minutes > 60.0:
        return "LATE_FIRST_SEEN_GT_60M"
    if lag_minutes < 0.0:
        return "INVALID_FIRST_SEEN_BEFORE_CLOSE"
    return "INELIGIBLE_FAIL_CLOSED"


def main() -> None:
    dsn = os.environ["V51_DATABASE_URL"]
    source_commit = resolve_source_commit()
    captured_at = pd.Timestamp.now(tz="UTC")
    deployment = os.environ.get("RAILWAY_DEPLOYMENT_ID", "manual")
    run_id = f"{deployment}:{captured_at.strftime('%Y%m%dT%H%M%SZ')}"

    with tempfile.TemporaryDirectory(prefix="v51-collector-") as tmp:
        outdir = Path(tmp)
        manifest = collect(outdir, run_id=run_id, captured_at=captured_at)
        raw = pd.read_csv(outdir / "raw_4h_ohlcv_v51.csv")

        inserted = 0
        preserved = 0
        with psycopg.connect(dsn) as conn:
            with conn.cursor() as cur:
                for _, row in raw.iterrows():
                    digest = row_hash(row)
                    eligible = str(row["prospective_eligible_v51"]).strip().lower() in {"true", "1"}
                    reason = ineligibility_reason(row)
                    cur.execute(
                        """
                        INSERT INTO v51_research.prospective_ohlcv
                          (venue,symbol,timeframe,bar_open_at,bar_close_at,first_seen_at,open,high,low,close,volume,
                           payload_sha256,prospective_eligible,ineligibility_reason,source_commit,collector_version)
                        VALUES (%s,%s,'4h',%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        ON CONFLICT (venue,symbol,timeframe,bar_open_at) DO NOTHING
                        """,
                        (
                            row["venue"], row["symbol"], row["bar_open_time"], row["bar_close_time"],
                            row["first_seen_at"], float(row["open"]), float(row["high"]), float(row["low"]),
                            float(row["close"]), float(row["volume"]), digest, eligible, reason,
                            source_commit, COLLECTOR_VERSION,
                        ),
                    )
                    if cur.rowcount == 1:
                        inserted += 1
                    else:
                        cur.execute(
                            """
                            SELECT payload_sha256, first_seen_at, prospective_eligible
                            FROM v51_research.prospective_ohlcv
                            WHERE venue=%s AND symbol=%s AND timeframe='4h' AND bar_open_at=%s
                            """,
                            (row["venue"], row["symbol"], row["bar_open_time"]),
                        )
                        existing = cur.fetchone()
                        if existing is None or existing[0] != digest:
                            raise RuntimeError("closed-bar revision detected in persisted v0.51 evidence")
                        preserved += 1

                cur.execute(
                    """
                    INSERT INTO v51_research.collection_runs
                      (run_id,captured_at,successful_series,expected_series,collection_health,source_commit,
                       collector_version,inserted_rows,preserved_rows)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (run_id) DO NOTHING
                    """,
                    (
                        run_id,
                        captured_at.to_pydatetime(),
                        int(manifest["successful_series_this_run"]),
                        int(manifest["expected_series_per_run"]),
                        manifest["collection_health"],
                        source_commit,
                        COLLECTOR_VERSION,
                        inserted,
                        preserved,
                    ),
                )

                audit_manifest = dict(manifest)
                audit_manifest.update(
                    {
                        "source_commit": source_commit,
                        "collector_version": COLLECTOR_VERSION,
                        "inserted_rows": inserted,
                        "existing_preserved": preserved,
                    }
                )
                digest = manifest_hash(audit_manifest)
                cur.execute(
                    """
                    INSERT INTO v51_research.run_manifests (run_id,manifest,manifest_sha256)
                    VALUES (%s,%s::jsonb,%s)
                    ON CONFLICT (run_id) DO NOTHING
                    """,
                    (run_id, json.dumps(audit_manifest, sort_keys=True), digest),
                )
            conn.commit()

    print(
        json.dumps(
            {
                "status": "OK",
                "role": "raw_public_market_evidence_only",
                "captured_at": captured_at.isoformat(),
                "source_commit": source_commit,
                "collector_version": COLLECTOR_VERSION,
                "inserted": inserted,
                "existing_preserved": preserved,
                "collection_health": manifest["collection_health"],
                "prospective_rows_seen": manifest["raw_rows_prospective_eligible"],
                "kraken_touched": False,
                "paper_execution": False,
                "live_execution": False,
                "scoring_performed": False,
                "arbitration_performed": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
