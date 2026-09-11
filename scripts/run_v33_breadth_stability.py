from __future__ import annotations

import argparse, json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from research_bot.breadth_stability_v33 import (
    DEVELOPMENT_VENUES_V33, V33_CANDIDATES, TOTAL_EFFECTIVE_TRIALS_V33,
    apply_breadth_gate, cross_sectional_breadth, preregistration_manifest_v33,
    screen_three_venues, select_v33_winner, v33_decision,
)
from research_bot.causal_evaluation_v25 import evaluate_external_replication_v25
from research_bot.cross_venue_robustness_v27 import compact_external_metrics
from research_bot.drawdown_firewall_v31 import apply_drawdown_firewall_v31, allocator_diagnostics_v31
from research_bot.multitimeframe_strategies_v19 import StrategySpec, TournamentConfig
from research_bot.multitimeframe_strategies_v20 import RiskPsychologyPolicy, V20ValidationConfig, simulate_v20_trades
from research_bot.portfolio_allocator_v25 import PortfolioRiskBudgetV25
from research_bot.regime_event_alpha_v30 import V30_CANDIDATES, cross_sectional_dispersion, generate_direction_v30


def load_snapshot(path: Path) -> dict[str, pd.DataFrame]:
    x = pd.read_csv(path, compression="gzip")
    x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True, errors="raise")
    return {str(s): g[["timestamp","open","high","low","close","volume"]].sort_values("timestamp").reset_index(drop=True) for s,g in x.groupby("symbol", sort=True)}


def empty_eval() -> dict[str, Any]:
    return {"external_trades":0,"external_profit_factor":np.nan,"external_expectancy_r":np.nan,"external_max_drawdown":np.nan,"external_positive_asset_fraction":0.0,"external_block_ci_low":np.nan,"external_block_ci_high":np.nan,"external_pass_v25":False}


def raw_attempts(v33, frames, tournament, risk):
    if "BTC/USDT" not in frames:
        return pd.DataFrame()
    base = next(x for x in V30_CANDIDATES if x.name == "V30_D1_CUSUM_BREAKOUT_3")
    market = frames["BTC/USDT"]
    dispersion = cross_sectional_dispersion(frames, "1d")
    breadth = cross_sectional_breadth(frames)
    spec = StrategySpec(v33.name, "1d", "v33_breadth_stability", "Frozen v0.30 CUSUM breakout + causal cross-sectional breadth confirmation", rr=base.rr, stop_atr=base.stop_atr, max_hold_bars=base.max_hold_bars, long_short=True)
    parts = []
    for symbol, frame in sorted(frames.items()):
        direction, features = generate_direction_v30(base, frame, market_frame=market, dispersion=dispersion)
        direction, features = apply_breadth_gate(direction, features, breadth, v33)
        ledger = simulate_v20_trades(spec, frame, direction, features, symbol, tournament=tournament, policy=risk)
        if not ledger.empty:
            parts.append(ledger)
    return pd.concat(parts, ignore_index=True).sort_values(["entry_time","symbol"], kind="mergesort").reset_index(drop=True) if parts else pd.DataFrame()


def evaluate(v33, frames, tournament, risk, budget, validation):
    raw = raw_attempts(v33, frames, tournament, risk)
    if raw.empty:
        return pd.DataFrame(), empty_eval()
    base = next(x for x in V30_CANDIDATES if x.name == "V30_D1_CUSUM_BREAKOUT_3")
    spec = StrategySpec(v33.name, "1d", "v33_breadth_stability", "Frozen v0.30 CUSUM breakout + causal cross-sectional breadth confirmation", rr=base.rr, stop_atr=base.stop_atr, max_hold_bars=base.max_hold_bars, long_short=True)
    allocated = apply_drawdown_firewall_v31(raw, spec, risk_policy=risk, budget=budget)
    m = dict(evaluate_external_replication_v25(allocated, validation=validation))
    m.update(allocator_diagnostics_v31(allocated))
    return allocated, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v30-artifact-dir", required=True)
    ap.add_argument("--v31-artifact-dir", required=True)
    ap.add_argument("--output-dir", default="artifacts/v33-breadth-stability")
    a = ap.parse_args()
    s30, s31, out = Path(a.v30_artifact_dir), Path(a.v31_artifact_dir), Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    prior = json.loads((s31/"decision_v31.json").read_text())
    if prior.get("decision") != "V31_FIREWALL_CANDIDATE_REJECTED_HOLDOUT" or not prior.get("holdout_used"):
        raise RuntimeError("v0.33 requires consumed v0.31 KuCoin rejection")

    frames = {
        "coinex_consumed": load_snapshot(s30/"coinex_1d_ohlcv_v30.csv.gz"),
        "okx_consumed": load_snapshot(s30/"okx_1d_ohlcv_v30.csv.gz"),
        "kucoin_consumed": load_snapshot(s31/"kucoin_1d_ohlcv_v31.csv.gz"),
    }
    risk, budget = RiskPsychologyPolicy(), PortfolioRiskBudgetV25()
    validation = V20ValidationConfig(min_external_trades=200)
    tournament = TournamentConfig(risk_per_trade=risk.base_risk_per_trade, min_pretest_trades=1000, min_test_trades=200)

    prereg = preregistration_manifest_v33()
    (out/"preregistration_v33.json").write_text(json.dumps(prereg, indent=2, sort_keys=True))
    rows, detailed = [], {}
    for c in V33_CANDIDATES:
        detailed[c.name] = {}
        compact = {}
        for venue in DEVELOPMENT_VENUES_V33:
            _, m = evaluate(c, frames[venue], tournament, risk, budget, validation)
            detailed[c.name][venue] = m
            compact[venue] = compact_external_metrics(m, venue=venue)
        screen = screen_three_venues(compact)
        row = {"strategy":c.name,"timeframe":"1d","breadth_threshold":c.breadth_threshold,"persistence_bars":c.persistence_bars, **screen}
        for venue in DEVELOPMENT_VENUES_V33:
            tag = venue.split("_")[0]
            row.update({f"{tag}_trades":compact[venue].get("trades"), f"{tag}_profit_factor":compact[venue].get("profit_factor"), f"{tag}_expectancy_r":compact[venue].get("expectancy_r"), f"{tag}_max_drawdown":compact[venue].get("max_drawdown"), f"{tag}_breadth":compact[venue].get("positive_asset_fraction"), f"{tag}_block_ci_low":compact[venue].get("block_ci_low")})
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values(["development_eligible_v33","robust_floor_block_ci_low_v33","robust_floor_breadth_v33","robust_floor_profit_factor_v33"], ascending=[False,False,False,False], kind="mergesort").reset_index(drop=True)
    summary.to_csv(out/"development_summary_v33.csv", index=False)
    (out/"development_metrics_v33.json").write_text(json.dumps(detailed, indent=2, sort_keys=True, default=str))
    winner = select_v33_winner(rows)
    lock = {"version":"v0.33","locked_at_utc":pd.Timestamp.now(tz="UTC").isoformat(),"winner":None if winner is None else winner["strategy"],"timeframe":"1d" if winner else None,"effective_trials":TOTAL_EFFECTIVE_TRIALS_V33,"development_venues":list(DEVELOPMENT_VENUES_V33),"reserved_holdout_venue":"kraken","kraken_touched":False,"threshold_relaxation":False,"live_execution_authorized":False}
    decision = v33_decision(winner)
    (out/"development_lock_v33.json").write_text(json.dumps(lock, indent=2, sort_keys=True))
    (out/"decision_v33.json").write_text(json.dumps(decision, indent=2, sort_keys=True))
    print(json.dumps({"decision":decision,"eligible_count":int(summary["development_eligible_v33"].sum()),"top":summary.head(6).to_dict("records")}, indent=2, default=str))

if __name__ == "__main__":
    main()
