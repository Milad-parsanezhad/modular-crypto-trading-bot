from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import pandas as pd
import psycopg

from research_bot.prospective_collector_v51 import collect


def row_hash(row: pd.Series) -> str:
    payload = {
        "venue": row["venue"],
        "symbol": row["symbol"],
        "bar_open_time": row["bar_open_time"],
        "bar_close_time": row["bar_close_time"],
        "open": float(row["open"]),
        "high": float(row["high"]),
        "low": float(row["low"]),
        "close": float(row["close"]),
        "volume": float(row["volume"]),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    dsn = os.environ["V51_DATABASE_URL"]
    source_commit = os.environ.get("RAILWAY_GIT_COMMIT_SHA", os.environ.get("GITHUB_SHA", "UNKNOWN"))
    captured_at = pd.Timestamp.now(tz="UTC")
    run_id = os.environ.get("RAILWAY_DEPLOYMENT_ID", captured_at.strftime("%Y%m%dT%H%M%SZ")) + ":" + captured_at.strftime("%Y%m%dT%H%M%SZ")

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
                    reason = None if eligible else "OUTSIDE_FIRST_SEEN_OR_FROZEN_WINDOW"
                    cur.execute(
                        """
                        INSERT INTO v51_research.prospective_ohlcv
                          (venue,symbol,timeframe,bar_open_at,bar_close_at,first_seen_at,open,high,low,close,volume,
                           payload_sha256,prospective_eligible,ineligibility_reason,source_commit,collector_version)
                        VALUES (%s,%s,'4h',%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'v51-ops-1')
                        ON CONFLICT (venue,symbol,timeframe,bar_open_at) DO NOTHING
                        """,
                        (
                            row["venue"], row["symbol"], row["bar_open_time"], row["bar_close_time"],
                            row["first_seen_at"], float(row["open"]), float(row["high"]), float(row["low"]),
                            float(row["close"]), float(row["volume"]), digest, eligible, reason, source_commit,
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
            conn.commit()

    print(json.dumps({
        "status": "OK",
        "role": "raw_public_market_evidence_only",
        "captured_at": captured_at.isoformat(),
        "inserted": inserted,
        "existing_preserved": preserved,
        "collection_health": manifest["collection_health"],
        "prospective_rows_seen": manifest["raw_rows_prospective_eligible"],
        "kraken_touched": False,
        "paper_execution": False,
        "live_execution": False,
        "scoring_performed": False,
        "arbitration_performed": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
