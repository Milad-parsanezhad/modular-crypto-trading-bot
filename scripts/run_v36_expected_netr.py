from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research_bot.event_utility_v36 import (
    BASE_D1_STRATEGIES_V36,
    DEVELOPMENT_VENUES_V36,
    TOTAL_EFFECTIVE_TRIALS_V36,
    V36_CANDIDATES,
    event_metrics_v36,
    fit_score_target_v36,
    preregistration_manifest_v36,
    screen_three_venues_v36,
    select_v36_winner,
    v36_decision,
)
from research_bot.multitimeframe_strategies_v19 import TournamentConfig
from research_bot.multitimeframe_strategies_v20 import RiskPsychologyPolicy, simulate_v20_trades
from research_bot.regime_event_alpha_v30 import V30_CANDIDATES, cross_sectional_dispersion, generate_direction_v30


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


def _feature_slice_v36(features: pd.DataFrame) -> pd.DataFrame:
    f = features.copy().sort_values("timestamp").reset_index(drop=True)
    close = pd.to_numeric(f["close"], errors="coerce").replace(0, np.nan)
    volume = pd.to_numeric(f["volume"], errors="coerce")
    volume_base = volume.shift(1).rolling(50, min_periods=20).median().replace(0, np.nan)
    dispersion_cap = pd.to_numeric(f.get("dispersion_cap_v30"), errors="coerce").replace(0, np.nan)
    cloud_mid = (
        pd.to_numeric(f.get("cloud_top"), errors="coerce")
        + pd.to_numeric(f.get("cloud_bottom"), errors="coerce")
    ) / 2.0
    out = pd.DataFrame({
        "signal_time": pd.to_datetime(f["timestamp"], utc=True, errors="raise"),
        "f_atr_pct_v36": pd.to_numeric(f.get("atr_pct"), errors="coerce"),
        "f_ret1_v36": pd.to_numeric(f.get("ret1"), errors="coerce"),
        "f_ret12_v36": pd.to_numeric(f.get("ret12"), errors="coerce"),
        "f_ema20_gap_v36": (close - pd.to_numeric(f.get("ema20"), errors="coerce")) / close,
        "f_ema50_gap_v36": (close - pd.to_numeric(f.get("ema50"), errors="coerce")) / close,
        "f_ema200_gap_v36": (close - pd.to_numeric(f.get("ema200"), errors="coerce")) / close,
        "f_regime_v36": pd.to_numeric(f.get("market_regime_v30"), errors="coerce"),
        "f_dispersion_ratio_v36": pd.to_numeric(f.get("dispersion_v30"), errors="coerce") / dispersion_cap,
        "f_volume_ratio_v36": volume / volume_base,
        "f_cloud_position_v36": (close - cloud_mid) / close,
    })
    return out


def build_event_pool_v36(
    frames: dict[str, pd.DataFrame],
    *,
    tournament: TournamentConfig,
    risk: RiskPsychologyPolicy,
) -> pd.DataFrame:
    if "BTC/USDT" not in frames:
        return pd.DataFrame()
    market = frames["BTC/USDT"]
    dispersion = cross_sectional_dispersion(frames, "1d")
    wanted = [c for c in V30_CANDIDATES if c.name in BASE_D1_STRATEGIES_V36]
    if len(wanted) != len(BASE_D1_STRATEGIES_V36):
        raise RuntimeError("v0.36 base registry mismatch")
    parts: list[pd.DataFrame] = []
    for candidate in wanted:
        spec = candidate.strategy_spec()
        for symbol, frame in sorted(frames.items()):
            direction, features = generate_direction_v30(
                candidate,
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
            if ledger.empty:
                continue
            ledger = ledger.copy()
            ledger["signal_time"] = pd.to_datetime(ledger["signal_time"], utc=True, errors="raise")
            ledger["base_strategy_v36"] = candidate.name
            context = _feature_slice_v36(features)
            ledger = ledger.merge(context, on="signal_time", how="left", validate="many_to_one")
            parts.append(ledger)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out["signal_time"] = pd.to_datetime(out["signal_time"], utc=True, errors="raise")
    out["entry_time"] = pd.to_datetime(out["entry_time"], utc=True, errors="raise")
    out["exit_time"] = pd.to_datetime(out["exit_time"], utc=True, errors="raise")
    return out.sort_values(["signal_time", "base_strategy_v36", "symbol"], kind="mergesort").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="v0.36 expected net-R conformal event utility screen")
    ap.add_argument("--v30-artifact-dir", required=True)
    ap.add_argument("--v31-artifact-dir", required=True)
    ap.add_argument("--v34-artifact-dir", required=True)
    ap.add_argument("--v35-artifact-dir", required=True)
    ap.add_argument("--output-dir", default="artifacts/v36-expected-netr")
    args = ap.parse_args()

    s30, s31, s34, s35 = map(Path, [
        args.v30_artifact_dir, args.v31_artifact_dir, args.v34_artifact_dir, args.v35_artifact_dir,
    ])
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    prior31 = json.loads((s31 / "decision_v31.json").read_text(encoding="utf-8"))
    prior34 = json.loads((s34 / "decision_v34.json").read_text(encoding="utf-8"))
    prior35 = json.loads((s35 / "decision_v35.json").read_text(encoding="utf-8"))
    if prior31.get("decision") != "V31_FIREWALL_CANDIDATE_REJECTED_HOLDOUT" or not bool(prior31.get("holdout_used")):
        raise RuntimeError("v0.36 requires KuCoin to be validly consumed by v0.31")
    if prior34.get("decision") != "BLOCKED_BY_V33" or bool(prior34.get("holdout_used")):
        raise RuntimeError("v0.36 requires proof that Kraken remained untouched")
    if prior35.get("decision") != "NO_V35_ROBUST_DEVELOPMENT_CANDIDATE" or bool(prior35.get("kraken_touched")):
        raise RuntimeError("v0.36 requires completed v0.35 rejection with Kraken untouched")

    frames_by_venue = {
        "coinex_consumed": load_snapshot(s30 / "coinex_1d_ohlcv_v30.csv.gz"),
        "okx_consumed": load_snapshot(s30 / "okx_1d_ohlcv_v30.csv.gz"),
        "kucoin_consumed": load_snapshot(s31 / "kucoin_1d_ohlcv_v31.csv.gz"),
    }
    risk = RiskPsychologyPolicy()
    tournament = TournamentConfig(
        risk_per_trade=risk.base_risk_per_trade,
        min_pretest_trades=1000,
        min_test_trades=200,
    )

    prereg = preregistration_manifest_v36()
    prereg.update({
        "source_v30_run_id": 34632401230,
        "source_v31_run_id": 34633572528,
        "source_v34_run_id": 34635635582,
        "source_v35_run_id": 34637596442,
    })
    write_json(out / "preregistration_v36.json", prereg)

    pools: dict[str, pd.DataFrame] = {}
    pool_diag: dict[str, Any] = {}
    for venue in DEVELOPMENT_VENUES_V36:
        panel = build_event_pool_v36(frames_by_venue[venue], tournament=tournament, risk=risk)
        panel["venue_v36"] = venue
        pools[venue] = panel
        pool_diag[venue] = {
            "events": int(len(panel)),
            "unique_symbols": int(panel["symbol"].nunique()) if not panel.empty else 0,
            "unique_strategies": int(panel["base_strategy_v36"].nunique()) if not panel.empty else 0,
            "first_signal": None if panel.empty else panel["signal_time"].min().isoformat(),
            "last_signal": None if panel.empty else panel["signal_time"].max().isoformat(),
        }
    write_json(out / "event_pool_diagnostics_v36.json", pool_diag)

    rows: list[dict[str, Any]] = []
    detailed: dict[str, Any] = {}
    for candidate in V36_CANDIDATES:
        detailed[candidate.name] = {}
        metrics_by_venue: dict[str, dict[str, Any]] = {}
        for target_venue in DEVELOPMENT_VENUES_V36:
            source = pd.concat(
                [pools[v] for v in DEVELOPMENT_VENUES_V36 if v != target_venue],
                ignore_index=True,
            ).sort_values(["signal_time", "venue_v36", "symbol"], kind="mergesort").reset_index(drop=True)
            scored, diag = fit_score_target_v36(source, pools[target_venue], candidate)
            metrics = event_metrics_v36(scored)
            detailed[candidate.name][target_venue] = {"diagnostics": diag, "metrics": metrics}
            metrics_by_venue[target_venue] = metrics
            scored.to_csv(out / f"scored_{candidate.name}_{target_venue}.csv.gz", index=False, compression="gzip")

        screen = screen_three_venues_v36(metrics_by_venue)
        row: dict[str, Any] = {
            "strategy": candidate.name,
            "model_family": candidate.model_family,
            **screen,
        }
        for venue in DEVELOPMENT_VENUES_V36:
            tag = venue.split("_")[0]
            m = metrics_by_venue[venue]
            d = detailed[candidate.name][venue]["diagnostics"]
            row.update({
                f"{tag}_events": m.get("selected_events"),
                f"{tag}_profit_factor": m.get("profit_factor"),
                f"{tag}_expectancy_r": m.get("expectancy_r"),
                f"{tag}_breadth": m.get("positive_asset_fraction"),
                f"{tag}_block_ci_low": m.get("block_ci_low"),
                f"{tag}_stress_pf36": m.get("stress_36bps_profit_factor"),
                f"{tag}_event_dd": m.get("event_level_max_drawdown"),
                f"{tag}_calibration_cutoff": d.get("calibration_cutoff_utc"),
                f"{tag}_conformal_buffer_r": d.get("conformal_buffer_r"),
            })
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values(
        ["development_eligible_v36", "robust_floor_block_ci_low_v36", "robust_floor_breadth_v36", "robust_floor_profit_factor_v36"],
        ascending=[False, False, False, False], kind="mergesort",
    ).reset_index(drop=True)
    summary.to_csv(out / "development_summary_v36.csv", index=False)
    write_json(out / "development_metrics_v36.json", detailed)

    winner = select_v36_winner(rows)
    lock = {
        "version": "v0.36",
        "locked_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "winner": None if winner is None else winner["strategy"],
        "effective_trials": TOTAL_EFFECTIVE_TRIALS_V36,
        "development_venues": list(DEVELOPMENT_VENUES_V36),
        "reserved_holdout_venue": "kraken",
        "kraken_touched": False,
        "portfolio_gate_complete": False,
        "paper_replacement_authorized": False,
        "live_execution_authorized": False,
    }
    decision = v36_decision(winner)
    write_json(out / "development_lock_v36.json", lock)
    write_json(out / "decision_v36.json", decision)
    print(json.dumps({
        "decision": decision,
        "eligible_count": int(summary["development_eligible_v36"].sum()),
        "event_pool_diagnostics": pool_diag,
        "top": summary.head(3).to_dict("records"),
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
