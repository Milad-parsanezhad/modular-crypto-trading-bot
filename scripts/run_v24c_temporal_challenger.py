from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from research_bot.coinex_public import fetch_coinex_klines
from research_bot.ml_framework_v23r import dataframe_sha256
from research_bot.temporal_meta_v24c import TemporalMetaContract, temporal_validation_tournament
from research_bot.v24c_external_plan import (
    INTERNAL_LAST_CLOSED_BAR_UTC,
    INTERNAL_SYMBOLS_V24B,
    build_strategy_dataset_from_frames,
)

PERIOD = "4hour"
START = pd.Timestamp("2025-01-19T12:00:00Z")


def _ms(ts: pd.Timestamp) -> int:
    return int(ts.timestamp() * 1000)


def fetch_internal_frames() -> tuple[dict[str, pd.DataFrame], dict]:
    frames = {}; provenance = {}
    for symbol in INTERNAL_SYMBOLS_V24B:
        f = fetch_coinex_klines(
            symbol=symbol, period=PERIOD, market_type="spot",
            start_ms=_ms(START), end_ms=_ms(INTERNAL_LAST_CLOSED_BAR_UTC), bars=5000,
        )
        f["timestamp"] = pd.to_datetime(f["timestamp"], utc=True)
        f = f[(f["timestamp"] >= START) & (f["timestamp"] <= INTERNAL_LAST_CLOSED_BAR_UTC)].copy()
        f = f.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
        if len(f) != 3599 or f.timestamp.min() != START or f.timestamp.max() != INTERNAL_LAST_CLOSED_BAR_UTC:
            raise RuntimeError(f"frozen CoinEx coverage mismatch for {symbol}: {len(f)}")
        frames[symbol] = f
        provenance[symbol] = {
            "rows": len(f), "first": f.timestamp.min().isoformat(), "last": f.timestamp.max().isoformat(),
            "sha256": dataframe_sha256(f), "source": "CoinEx public spot OHLCV",
        }
    return frames, provenance


def main() -> None:
    ap = argparse.ArgumentParser(description="v0.24c development/validation-only temporal meta-learning challengers")
    ap.add_argument("--output-dir", default="artifacts/v24c-temporal-challenger")
    args = ap.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    contract = TemporalMetaContract()
    (out / "contract.json").write_text(json.dumps(contract.to_dict(), indent=2), encoding="utf-8")

    frames, provenance = fetch_internal_frames()
    (out / "data_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    dataset = build_strategy_dataset_from_frames(frames)
    # The v0.24b terminal segment is already seen historically. It is retained in
    # the reconstructed table only to verify split reproduction; temporal models
    # are forbidden from scoring or selecting on it in this stage.
    segment_counts = dataset.groupby(["strategy", "segment"]).size().unstack(fill_value=0)
    segment_counts.to_csv(out / "segment_counts.csv")

    leaderboard, frozen = temporal_validation_tournament(frames, dataset, contract=contract)
    leaderboard.to_csv(out / "temporal_validation_leaderboard.csv", index=False)

    import torch
    metadata = {}
    for strategy, bundle in frozen.items():
        model = bundle.pop("model")
        safe = strategy.lower().replace("/", "_")
        model_path = out / f"temporal_{safe}.pt"
        torch.save(model.state_dict(), model_path)
        metadata[strategy] = {**bundle, "state_dict_file": model_path.name}
    (out / "frozen_temporal_challengers.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    decision = {
        "version": "v0.24c",
        "track": "TEMPORAL_SEQUENCE",
        "stage": "DEVELOPMENT_FIT_VALIDATION_ONLY_SELECTION",
        "models": list(contract.model_kinds),
        "seeds": list(contract.seeds),
        "frozen_challenger_strategies": sorted(metadata),
        "terminal_v24b_test_scored": False,
        "scientific_promotion_authorized": False,
        "next_requirement": "Fresh external/future-time scoring only after temporal champion freeze; no same-test rescue tuning.",
        "forward_paper_authorized": False,
        "live_execution_authorized": False,
    }
    (out / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
