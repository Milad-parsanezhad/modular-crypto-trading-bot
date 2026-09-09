from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from research_bot.binance_spot_archive import load_monthly_spot_archives
from research_bot.forward_candidate_v17 import s6_candidate_signal


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an observation-only S6 v0.17 paper decision")
    parser.add_argument("--archive-cache", type=Path, required=True)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--currently-long", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("artifacts/v17-strategy-lab/latest_paper_candidate.json"))
    args = parser.parse_args()

    frame = load_monthly_spot_archives(args.archive_cache, args.symbol, timeframe="4h")
    signal = s6_candidate_signal(frame, currently_long=args.currently_long)
    payload = {
        "research_status": "FORWARD_PAPER_CANDIDATE_NOT_LIVE",
        "symbol": args.symbol,
        "decision_at_closed_bar": asdict(signal),
        "next_step": "Record or simulate a fill at the next bar open; never submit a live order.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
