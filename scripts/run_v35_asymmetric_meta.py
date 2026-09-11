from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research_bot.asymmetric_meta_v35 import (
    DEVELOPMENT_VENUES_V35,
    TOTAL_EFFECTIVE_TRIALS_V35,
    V35_CANDIDATES,
    attach_event_context_v35,
    causal_empirical_bayes_meta_score_v35,
    cross_sectional_context_v35,
    generate_asymmetric_direction_v35,
    preregistration_manifest_v35,
    screen_three_venues_v35,
    select_candidate_events_v35,
    select_v35_winner,
    v35_decision,
)
from research_bot.causal_evaluation_v25 import evaluate_external_replication_v25
from research_bot.cross_venue_robustness_v27 import compact_external_metrics
from research_bot.drawdown_firewall_v31 import apply_drawdown_firewall_v31, allocator_diagnostics_v31
from research_bot.multitimeframe_strategies_v19 import StrategySpec, TournamentConfig
from research_bot.multitimeframe_strategies_v20 import RiskPsychologyPolicy, V20ValidationConfig, simulate_v20_trades
from research_bot.portfolio_allocator_v25 import PortfolioRiskBudgetV25
from research_bot.regime_event_alpha_v30 import V30_CANDIDATES, cross_sectional_dispersion


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def load_snapshot(path: Path) -> dict[str, pd.DataFrame]:
    x = pd.read_csv(path, compression="gzip")
    x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True, errors="raise")
    return {
        str(symbol): group[["timestamp", "open", "high", "low", "close", "volume"]]
        .sort_values("timestamp").reset_index(drop=True)
        for symbol, group in x.groupby("symbol", sort=True)
    }


def empty_eval() -> dict[str, Any]:
    return {
        "external_trades": 0,
        "external_profit_factor": np.nan,
        "external_expectancy_r": np.nan,
        "external_max_drawdown": np.nan,
        "external_positive_asset_fraction": 0.0,
        "external_block_ci_low": np.nan,
        "external_block_ci_high": np.nan,
        "external_pass_v25": False,
    }


def base_event_panel(
    frames: dict[str, pd.DataFrame],
    *,
    tournament: TournamentConfig,
    risk: RiskPsychologyPolicy,
) -> pd.DataFrame:
    if "BTC/USDT" not in frames:
        return pd.DataFrame()
    base = next(x for x in V30_CANDIDATES if x.name == "V30_D1_CUSUM_BREAKOUT_3")
    spec = StrategySpec(
        "V35_BASE_D1_ASYMMETRIC_CUSUM",
        "1d",
        "v35_asymmetric_regime",
        "New asymmetric CUSUM-breakout event family; longs UP-UP, shorts regime<=0",
        rr=base.rr,
        stop_atr=base.stop_atr,
        max_hold_bars=base.max_hold_bars,
        long_short=True,
    )
    market = frames["BTC/USDT"]
    dispersion = cross_sectional_dispersion(frames, "1d")
    context = cross_sectional_context_v35(frames)
    parts: list[pd.DataFrame] = []
    for symbol, frame in sorted(frames.items()):
        direction, features = generate_asymmetric_direction_v35(
            frame,
            market_frame=market,
            dispersion=dispersion,
        )
        ledger = simulate_v20_trades(
            spec,
            frame,
            direction,
            features,
            symbol,
            tournament=tournament,
            policy=risk,
        )
        if not ledger.empty:
            parts.append(ledger)
    if not parts:
        return pd.DataFrame()
    events = pd.concat(parts, ignore_index=True).sort_values(["signal_time", "symbol"], kind="mergesort").reset_index(drop=True)
    events = attach_event_context_v35(events, context)
    return causal_empirical_bayes_meta_score_v35(events)


def evaluate_candidate(
    candidate,
    events: pd.DataFrame,
    *,
    risk: RiskPsychologyPolicy,
    budget: PortfolioRiskBudgetV25,
    validation: V20ValidationConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    selected = select_candidate_events_v35(events, candidate)
    if selected.empty:
        return selected, empty_eval()
    base = next(x for x in V30_CANDIDATES if x.name == "V30_D1_CUSUM_BREAKOUT_3")
    spec = StrategySpec(
        candidate.name,
        "1d",
        "v35_asymmetric_regime_cross_sectional_meta",
        "Asymmetric regime + cross-sectional relative strength + causal empirical-Bayes meta-label",
        rr=base.rr,
        stop_atr=base.stop_atr,
        max_hold_bars=base.max_hold_bars,
        long_short=True,
    )
    allocated = apply_drawdown_firewall_v31(selected, spec, risk_policy=risk, budget=budget)
    metrics = dict(evaluate_external_replication_v25(allocated, validation=validation))
    metrics.update(allocator_diagnostics_v31(allocated))
    metrics["raw_selected_events_v35"] = int(len(selected))
    metrics["selected_long_events_v35"] = int((selected["side"].astype(int) > 0).sum())
    metrics["selected_short_events_v35"] = int((selected["side"].astype(int) < 0).sum())
    metrics["median_meta_score_v35"] = float(pd.to_numeric(selected["meta_score_v35"], errors="coerce").median())
    return allocated, metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="v0.35 asymmetric regime cross-sectional meta development screen")
    ap.add_argument("--v30-artifact-dir", required=True)
    ap.add_argument("--v31-artifact-dir", required=True)
    ap.add_argument("--v34-artifact-dir", required=True)
    ap.add_argument("--output-dir", default="artifacts/v35-asymmetric-meta")
    args = ap.parse_args()

    s30 = Path(args.v30_artifact_dir)
    s31 = Path(args.v31_artifact_dir)
    s34 = Path(args.v34_artifact_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    prior31 = json.loads((s31 / "decision_v31.json").read_text(encoding="utf-8"))
    prior34 = json.loads((s34 / "decision_v34.json").read_text(encoding="utf-8"))
    if prior31.get("decision") != "V31_FIREWALL_CANDIDATE_REJECTED_HOLDOUT" or not bool(prior31.get("holdout_used")):
        raise RuntimeError("v0.35 requires KuCoin to be consumed by the valid v0.31 rejection")
    if prior34.get("decision") != "BLOCKED_BY_V33" or bool(prior34.get("holdout_used")):
        raise RuntimeError("v0.35 requires v0.34 to prove Kraken remained untouched")

    frames_by_venue = {
        "coinex_consumed": load_snapshot(s30 / "coinex_1d_ohlcv_v30.csv.gz"),
        "okx_consumed": load_snapshot(s30 / "okx_1d_ohlcv_v30.csv.gz"),
        "kucoin_consumed": load_snapshot(s31 / "kucoin_1d_ohlcv_v31.csv.gz"),
    }

    risk = RiskPsychologyPolicy()
    budget = PortfolioRiskBudgetV25()
    validation = V20ValidationConfig(min_external_trades=200)
    tournament = TournamentConfig(
        risk_per_trade=risk.base_risk_per_trade,
        min_pretest_trades=1000,
        min_test_trades=200,
    )

    prereg = preregistration_manifest_v35()
    prereg["source_v30_run_id"] = 34632401230
    prereg["source_v31_run_id"] = 34633572528
    prereg["source_v34_run_id"] = 34635635582
    write_json(out / "preregistration_v35.json", prereg)

    base_events: dict[str, pd.DataFrame] = {}
    base_diag: dict[str, Any] = {}
    for venue in DEVELOPMENT_VENUES_V35:
        panel = base_event_panel(frames_by_venue[venue], tournament=tournament, risk=risk)
        base_events[venue] = panel
        if panel.empty:
            base_diag[venue] = {"events": 0}
        else:
            base_diag[venue] = {
                "events": int(len(panel)),
                "long_events": int((panel["side"].astype(int) > 0).sum()),
                "short_events": int((panel["side"].astype(int) < 0).sum()),
                "meta_history_ready_fraction": float((pd.to_numeric(panel["meta_history_n_v35"], errors="coerce").fillna(0) >= 20).mean()),
                "median_meta_score": float(pd.to_numeric(panel["meta_score_v35"], errors="coerce").median()),
            }
    write_json(out / "base_event_diagnostics_v35.json", base_diag)

    rows: list[dict[str, Any]] = []
    detailed: dict[str, dict[str, Any]] = {}
    candidate_ledgers: dict[tuple[str, str], pd.DataFrame] = {}
    for candidate in V35_CANDIDATES:
        detailed[candidate.name] = {}
        compact: dict[str, dict[str, Any]] = {}
        for venue in DEVELOPMENT_VENUES_V35:
            allocated, metrics = evaluate_candidate(
                candidate,
                base_events[venue],
                risk=risk,
                budget=budget,
                validation=validation,
            )
            candidate_ledgers[(candidate.name, venue)] = allocated
            detailed[candidate.name][venue] = metrics
            compact[venue] = compact_external_metrics(metrics, venue=venue)

        screen = screen_three_venues_v35(compact)
        row: dict[str, Any] = {
            "strategy": candidate.name,
            "timeframe": "1d",
            "long_rank_min": candidate.long_rank_min,
            "short_rank_max": candidate.short_rank_max,
            "long_meta_min": candidate.long_meta_min,
            "short_meta_min": candidate.short_meta_min,
            **screen,
        }
        for venue in DEVELOPMENT_VENUES_V35:
            tag = venue.split("_")[0]
            row.update({
                f"{tag}_trades": compact[venue].get("trades"),
                f"{tag}_profit_factor": compact[venue].get("profit_factor"),
                f"{tag}_expectancy_r": compact[venue].get("expectancy_r"),
                f"{tag}_max_drawdown": compact[venue].get("max_drawdown"),
                f"{tag}_breadth": compact[venue].get("positive_asset_fraction"),
                f"{tag}_block_ci_low": compact[venue].get("block_ci_low"),
                f"{tag}_raw_selected": detailed[candidate.name][venue].get("raw_selected_events_v35", 0),
                f"{tag}_selected_long": detailed[candidate.name][venue].get("selected_long_events_v35", 0),
                f"{tag}_selected_short": detailed[candidate.name][venue].get("selected_short_events_v35", 0),
            })
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values(
        [
            "development_eligible_v35",
            "robust_floor_block_ci_low_v35",
            "robust_floor_breadth_v35",
            "robust_floor_profit_factor_v35",
            "robust_floor_expectancy_r_v35",
        ],
        ascending=[False, False, False, False, False],
        kind="mergesort",
    ).reset_index(drop=True)
    summary.to_csv(out / "development_summary_v35.csv", index=False)
    write_json(out / "development_metrics_v35.json", detailed)

    winner = select_v35_winner(rows)
    lock = {
        "version": "v0.35",
        "locked_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "winner": None if winner is None else winner["strategy"],
        "timeframe": "1d" if winner is not None else None,
        "effective_trials": TOTAL_EFFECTIVE_TRIALS_V35,
        "development_venues": list(DEVELOPMENT_VENUES_V35),
        "reserved_holdout_venue": "kraken",
        "kraken_touched": False,
        "threshold_relaxation": False,
        "rejected_candidate_retuning": False,
        "live_execution_authorized": False,
    }
    decision = v35_decision(winner)
    write_json(out / "development_lock_v35.json", lock)
    write_json(out / "decision_v35.json", decision)

    if winner is not None:
        for venue in DEVELOPMENT_VENUES_V35:
            ledger = candidate_ledgers[(str(winner["strategy"]), venue)]
            if not ledger.empty:
                ledger.to_csv(out / f"winner_{venue}_ledger_v35.csv", index=False)

    print(json.dumps({
        "decision": decision,
        "eligible_count": int(summary["development_eligible_v35"].sum()),
        "base_event_diagnostics": base_diag,
        "top": summary.head(6).to_dict("records"),
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
