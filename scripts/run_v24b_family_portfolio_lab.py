from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from research_bot.coinex_public import PERIOD_MS, fetch_coinex_klines, utc_now_ms
from research_bot.ml_framework_v23r import SplitContract, chronological_purged_split
from research_bot.strategy_family_meta_v24b import (
    V24BContract,
    family_sample_table,
    fit_family_models,
    merge_family_predictions,
    portfolio_overlap_backtest,
)
from research_bot.strategy_meta_v24 import V24Contract, build_strategy_event_panel, target_specs

PERIOD = "4hour"
DEFAULT_SYMBOLS = (
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT", "ADA/USDT",
    "LINK/USDT", "LTC/USDT", "BCH/USDT", "TRX/USDT", "AVAX/USDT", "DOT/USDT",
)


def fetch_closed(symbol: str, bars: int) -> pd.DataFrame:
    x = fetch_coinex_klines(symbol=symbol, period=PERIOD, market_type="spot", end_ms=utc_now_ms(), bars=bars)
    x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True)
    closed = x["timestamp"] + pd.Timedelta(milliseconds=PERIOD_MS[PERIOD]) <= pd.Timestamp.now(tz="UTC")
    return x[closed].sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)


def peer_for(symbol: str, frames: dict[str, pd.DataFrame]) -> pd.DataFrame | None:
    preferred = "ETH/USDT" if symbol == "BTC/USDT" else "BTC/USDT"
    if preferred in frames and preferred != symbol:
        return frames[preferred]
    return next((f for s, f in frames.items() if s != symbol), None)


def build_dataset(symbols: list[str], bars: int, output: Path) -> tuple[pd.DataFrame, dict]:
    v24 = V24Contract()
    specs = target_specs()
    frames: dict[str, pd.DataFrame] = {}
    provenance: dict[str, dict] = {}
    for symbol in symbols:
        try:
            frame = fetch_closed(symbol, bars)
            if len(frame) < 800:
                raise RuntimeError(f"insufficient closed bars={len(frame)}")
            frames[symbol] = frame
            provenance[symbol] = {
                "rows": int(len(frame)),
                "first": str(frame.timestamp.min()),
                "last": str(frame.timestamp.max()),
                "source": "CoinEx public spot OHLCV",
                "period": PERIOD,
            }
        except Exception as exc:
            provenance[symbol] = {"error": f"{type(exc).__name__}: {exc}"}
    if len(frames) < 5:
        raise RuntimeError(f"v0.24b requires >=5 successful symbols, got {len(frames)}")

    parts: list[pd.DataFrame] = []
    coverage: list[dict] = []
    for spec in specs:
        for symbol, frame in frames.items():
            needs_peer = "CORRELATION" in spec.name or spec.family == "correlation_divergence"
            peer = peer_for(symbol, frames) if needs_peer else None
            panel = build_strategy_event_panel(spec, frame, symbol, peer=peer, contract=v24)
            if not panel.empty:
                parts.append(panel)
            coverage.append({
                "strategy": spec.name,
                "symbol": symbol,
                "events": int(len(panel)),
                "positive_fraction": float(panel["label_meta_execute"].mean()) if not panel.empty else np.nan,
            })
    if not parts:
        raise RuntimeError("no v0.24b strategy events generated")

    raw = pd.concat(parts, ignore_index=True)
    raw["signal_time"] = pd.to_datetime(raw["signal_time"], utc=True)
    raw["label_end_time"] = pd.to_datetime(raw["label_end_time"], utc=True)
    raw = raw.sort_values(["signal_time", "strategy", "symbol"], kind="mergesort").reset_index(drop=True)
    split = chronological_purged_split(
        raw,
        timestamp_col="signal_time",
        label_end_time_col="label_end_time",
        contract=SplitContract(development_fraction=0.60, validation_fraction=0.20, test_fraction=0.20, embargo_rows=2),
    )
    tagged = []
    for segment, part in split.items():
        z = part.copy()
        z["segment"] = segment
        tagged.append(z)
    dataset = pd.concat(tagged, ignore_index=True).sort_values(["signal_time", "strategy", "symbol"]).reset_index(drop=True)
    dataset.to_csv(output / "strategy_event_dataset.csv", index=False)
    pd.DataFrame(coverage).to_csv(output / "strategy_symbol_coverage.csv", index=False)
    (output / "data_provenance.json").write_text(json.dumps(provenance, indent=2, default=str), encoding="utf-8")
    return dataset, provenance


def main() -> None:
    ap = argparse.ArgumentParser(description="v0.24b family-specific meta-labeling + overlap-aware portfolio risk lab")
    ap.add_argument("--output-dir", default="artifacts/v24b-family-portfolio")
    ap.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    ap.add_argument("--bars", type=int, default=3600)
    args = ap.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    contract = V24BContract()
    (output / "contract.json").write_text(json.dumps(contract.to_dict(), indent=2), encoding="utf-8")

    symbols = [x.strip().upper() for x in args.symbols.split(",") if x.strip()]
    dataset, provenance = build_dataset(symbols, args.bars, output)
    sample_table = family_sample_table(dataset, contract)
    sample_table.to_csv(output / "family_sample_counts.csv", index=False)

    board, ablation, predictions, champions = fit_family_models(dataset, contract)
    board.to_csv(output / "family_validation_leaderboard.csv", index=False)
    ablation.to_csv(output / "family_shadow_ablation.csv", index=False)
    predictions.to_csv(output / "family_shadow_predictions.csv", index=False)
    joblib.dump(champions, output / "family_validation_frozen_champions.joblib", compress=3)

    shadow = dataset[dataset["segment"] == "test"].copy().reset_index(drop=True)
    merged = merge_family_predictions(shadow, predictions)
    merged.to_csv(output / "shadow_with_family_predictions.csv", index=False)

    base_summary, base_ledger = portfolio_overlap_backtest(shadow, contract=contract, mode="base_all_shadow_events")
    filter_mask = merged["family_meta_selected"].to_numpy(dtype=bool)
    filt_summary, filt_ledger = portfolio_overlap_backtest(
        merged,
        selected=filter_mask,
        contract=contract,
        mode="family_meta_filtered_shadow_events",
    )
    base_ledger.to_csv(output / "portfolio_ledger_base.csv", index=False)
    filt_ledger.to_csv(output / "portfolio_ledger_family_filter.csv", index=False)
    pd.DataFrame([base_summary, filt_summary]).to_csv(output / "portfolio_summary.csv", index=False)

    eligible = sample_table[sample_table["eligible_by_counts_only"] == True]["strategy"].tolist()  # noqa: E712
    insufficient = sample_table[sample_table["eligible_by_counts_only"] == False]["strategy"].tolist()  # noqa: E712
    positive_shadow = []
    if not ablation.empty and "uplift_total_return" in ablation:
        positive_shadow = ablation.loc[
            (ablation["status"] == "SHADOW_EVALUATED") & (pd.to_numeric(ablation["uplift_total_return"], errors="coerce") > 0),
            "strategy",
        ].tolist()

    final = {
        "version": "v0.24b",
        "stage": "STRATEGY_FAMILY_META_LABELING_PLUS_OVERLAP_AWARE_PORTFOLIO_RISK",
        "scientific_status": "EXPLORATORY_REUSED_SHADOW_NO_PROMOTION",
        "critical_validity_note": (
            "The v0.24 terminal CoinEx period had already been inspected before v0.24b was designed. "
            "This run is therefore an engineering/shadow experiment only; its terminal segment is not a fresh untouched test."
        ),
        "dataset_rows": int(len(dataset)),
        "successful_symbols": sorted([s for s, p in provenance.items() if "error" not in p]),
        "family_model_eligible_by_counts_only": eligible,
        "data_insufficient_families": insufficient,
        "family_shadow_positive_total_return_uplift": positive_shadow,
        "family_model_count": int(len(champions)),
        "portfolio_base_shadow": base_summary,
        "portfolio_family_filter_shadow": filt_summary,
        "portfolio_incremental_total_return": float(filt_summary["total_return"] - base_summary["total_return"]),
        "portfolio_incremental_max_realized_drawdown": float(filt_summary["max_realized_drawdown"] - base_summary["max_realized_drawdown"]),
        "portfolio_semantics": {
            "overlap_aware": True,
            "one_active_position_per_symbol": True,
            "portfolio_open_risk_cap": contract.max_open_risk_fraction,
            "strategy_open_risk_cap": contract.max_strategy_open_risk_fraction,
            "directional_open_risk_cap": contract.max_directional_open_risk_fraction,
            "max_concurrent_positions": contract.max_concurrent_positions,
            "hard_realized_drawdown_kill": contract.hard_realized_drawdown_kill,
            "mark_to_market": False,
            "costs": "inherited net r_multiple from v0.24 bracket simulator; baseline 24 bps round trip",
        },
        "next_evidence_gate": (
            "Fresh forward-time accumulation or untouched external-venue replication of the frozen family models, "
            "followed by mark-to-market portfolio risk and CPCV/multiple-testing validation."
        ),
        "forward_paper_authorized": False,
        "paper_replacement_authorized": False,
        "live_execution_authorized": False,
    }
    (output / "decision.json").write_text(json.dumps(final, indent=2, default=str), encoding="utf-8")
    print(json.dumps(final, indent=2, default=str))


if __name__ == "__main__":
    main()
