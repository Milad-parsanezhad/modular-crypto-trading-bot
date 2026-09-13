from __future__ import annotations

"""CLI wrapper for the packaged v0.51 prospective OHLCV collector."""

import argparse
import json
from pathlib import Path

import pandas as pd

from research_bot.prospective_collector_v51 import collect


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
