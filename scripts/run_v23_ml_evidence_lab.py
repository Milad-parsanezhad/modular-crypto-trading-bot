from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
import ccxt

from research_bot.coinex_public import PERIOD_MS, fetch_coinex_klines, utc_now_ms
from research_bot.deep_temporal_v22 import TemporalConfig, fit_binary_model, make_model, predict_probability
from research_bot.multimodal_fusion_v22d import RobustNumericScaler
from research_bot.ml_evidence_v23 import (
    MLV23Config, TABULAR_FEATURES, assign_segments, build_v23_rows,
    calibrated_probability, choose_validation_threshold, conformal_binary_quantile,
    feature_matrix, final_promotion_decision, fit_platt, make_tabular_pipeline,
    moving_block_ci, portfolio_backtest, predictive_metrics, protocol_dict,
    raw_score, tabular_model_zoo,
)

DEFAULT_SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "ADA/USDT"]
DEEP_FAMILIES = ["lstm", "gru", "tcn", "transformer"]
DEEP_SEEDS = [314, 2718, 1618]


def _drop_incomplete(df: pd.DataFrame, step_ms: int) -> pd.DataFrame:
    if df.empty: return df
    x = df.copy(); x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True)
    return x[x["timestamp"] + pd.Timedelta(milliseconds=step_ms) <= pd.Timestamp.now(tz="UTC")].reset_index(drop=True)


def fetch_coinex(symbol: str, bars: int) -> pd.DataFrame:
    x = fetch_coinex_klines(symbol=symbol, period="4hour", market_type="spot", end_ms=utc_now_ms(), bars=bars)
    return _drop_incomplete(x, PERIOD_MS["4hour"])


def fetch_external(exchange_id: str, symbols: list[str], bars: int, timeframe: str = "4h") -> tuple[str, dict[str, pd.DataFrame]]:
    def _one(ex_id: str) -> dict[str, pd.DataFrame]:
        klass = getattr(ccxt, ex_id); ex = klass({"enableRateLimit": True}); ex.load_markets()
        step_ms = int(ex.parse_timeframe(timeframe) * 1000); now_ms = ex.milliseconds(); since = now_ms - int((bars + 80) * step_ms)
        out = {}
        for symbol in symbols:
            if symbol not in ex.markets: continue
            rows = []; cursor = since
            for _ in range(30):
                if cursor >= now_ms or len(rows) >= bars + 80: break
                batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=min(300, bars + 80 - len(rows)))
                if not batch: break
                rows.extend(batch); nxt = int(batch[-1][0]) + step_ms
                if nxt <= cursor: break
                cursor = nxt; time.sleep((ex.rateLimit or 50) / 1000.0)
            if rows:
                df = pd.DataFrame(rows, columns=["timestamp_ms", "open", "high", "low", "close", "volume"])
                df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
                df = df[["timestamp", "open", "high", "low", "close", "volume"]].drop_duplicates("timestamp").sort_values("timestamp")
                out[symbol] = _drop_incomplete(df, step_ms).tail(bars).reset_index(drop=True)
        ex.close(); return out
    try:
        data = _one(exchange_id)
        if len(data) >= 3: return exchange_id, data
        raise RuntimeError(f"only {sorted(data)} available")
    except Exception:
        fallback = "kucoin" if exchange_id != "kucoin" else "okx"
        return fallback, _one(fallback)


def build_all_rows(frames: dict[str, pd.DataFrame], cfg: MLV23Config) -> tuple[pd.DataFrame, dict]:
    parts, provenance = [], {}
    for symbol, frame in frames.items():
        try:
            rows = build_v23_rows(frame, symbol, cfg)
            if len(rows) < 300:
                provenance[symbol] = {"status": "insufficient", "bars": int(len(frame)), "rows": int(len(rows))}; continue
            parts.append(rows)
            provenance[symbol] = {"status": "ok", "bars": int(len(frame)), "rows": int(len(rows)), "first": rows.timestamp.min().isoformat(), "last": rows.timestamp.max().isoformat()}
        except Exception as exc:
            provenance[symbol] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    if len(parts) < 3: raise RuntimeError(f"need >=3 usable symbols; provenance={provenance}")
    rows = assign_segments(pd.concat(parts, ignore_index=True), cfg)
    return rows, provenance


def split(rows: pd.DataFrame, name: str) -> pd.DataFrame:
    return rows[rows["segment"] == name].sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def train_tabular_candidates(rows: pd.DataFrame, cfg: MLV23Config, out: Path):
    dev, cal, val = split(rows, "development"), split(rows, "calibration"), split(rows, "validation")
    xdev, xcal, xval = feature_matrix(dev), feature_matrix(cal), feature_matrix(val)
    ydev, ycal, yval = dev.target.to_numpy(int), cal.target.to_numpy(int), val.target.to_numpy(int)
    records, fitted = [], {}
    for name, estimator in tabular_model_zoo(cfg.seed).items():
        rec = {"candidate": f"tabular:{name}", "family": "tabular", "model": name, "status": "failed"}
        try:
            model = make_tabular_pipeline(estimator); model.fit(xdev, ydev)
            cal_raw = raw_score(model, xcal); calibrator = fit_platt(cal_raw, ycal, cfg.seed)
            pcal = calibrated_probability(calibrator, cal_raw); pval = calibrated_probability(calibrator, raw_score(model, xval))
            threshold, econ = choose_validation_threshold(val, pval, cfg)
            rec.update({"status": "ok", "threshold": threshold, **{f"calibration_{k}": v for k, v in predictive_metrics(ycal, pcal).items()}, **{f"validation_{k}": v for k, v in predictive_metrics(yval, pval).items()}, **{f"validation_economic_{k}": v for k, v in econ.items()}})
            conf = conformal_binary_quantile(ycal, pcal, cfg.conformal_alpha); rec.update({f"conformal_{k}": v for k, v in conf.items()})
            fitted[rec["candidate"]] = {"type": "tabular", "model": model, "calibrator": calibrator, "threshold": threshold}
        except Exception as exc:
            rec["error"] = f"{type(exc).__name__}: {exc}"
        records.append(rec); print(json.dumps(rec, default=str))
    return records, fitted


def _deep_sequences(rows: pd.DataFrame, scaler: RobustNumericScaler, segment: str, lookback: int):
    xs, ys, rs, meta = [], [], [], []
    for symbol, g0 in rows.groupby("symbol", sort=True):
        g = g0.sort_values("timestamp").reset_index(drop=True).copy(); mat = scaler.transform(feature_matrix(g).to_numpy(np.float32))
        for i in range(lookback - 1, len(g)):
            if g.loc[i, "segment"] != segment: continue
            xs.append(mat[i - lookback + 1:i + 1]); ys.append(int(g.loc[i, "target"])); rs.append(float(g.loc[i, "gross_return"])); meta.append(g.loc[i])
    if not xs: raise RuntimeError(f"no deep sequences for {segment}")
    return np.asarray(xs, np.float32), np.asarray(ys, np.int8), np.asarray(rs, np.float32), pd.DataFrame(meta).reset_index(drop=True)


def train_deep_candidates(rows: pd.DataFrame, cfg: MLV23Config, out: Path, epochs: int, deep_seeds: list[int]):
    dev_rows = split(rows, "development"); scaler = RobustNumericScaler.fit(feature_matrix(dev_rows).to_numpy(np.float32)); lookback = 64
    Xd, yd, _, md = _deep_sequences(rows, scaler, "development", lookback); Xc, yc, _, mc = _deep_sequences(rows, scaler, "calibration", lookback); Xv, yv, _, mv = _deep_sequences(rows, scaler, "validation", lookback)
    order = np.argsort(pd.to_datetime(md["timestamp"], utc=True).astype("int64").to_numpy()); Xd, yd = Xd[order], yd[order]
    cut = max(100, int(len(Xd) * 0.85)); cut = min(cut, len(Xd) - 50); Xt0, yt0, Xes, yes = Xd[:cut], yd[:cut], Xd[cut:], yd[cut:]
    tcfg = TemporalConfig(lookback=lookback, hidden_dim=64, layers=2, dropout=0.15, batch_size=256, epochs=epochs, patience=2)
    records, fitted = [], {}
    for family in DEEP_FAMILIES:
        seed_models = []; pcal_seeds = []; pval_seeds = []; seed_status = []
        for seed in deep_seeds:
            try:
                model = make_model(family, Xd.shape[-1], tcfg); model, history = fit_binary_model(model, Xt0, yt0.astype(np.float32), Xes, yes.astype(np.float32), tcfg, seed)
                raw_cal_p = np.clip(predict_probability(model, Xc), 1e-6, 1 - 1e-6); raw_cal = np.log(raw_cal_p / (1 - raw_cal_p)); calibrator = fit_platt(raw_cal, yc, seed); pcal = calibrated_probability(calibrator, raw_cal)
                raw_val_p = np.clip(predict_probability(model, Xv), 1e-6, 1 - 1e-6); raw_val = np.log(raw_val_p / (1 - raw_val_p)); pval = calibrated_probability(calibrator, raw_val)
                seed_models.append((seed, model, calibrator)); pcal_seeds.append(pcal); pval_seeds.append(pval); seed_status.append("ok"); history.to_csv(out / f"deep_history_{family}_seed{seed}.csv", index=False)
            except Exception as exc:
                seed_status.append(f"failed:{type(exc).__name__}:{exc}")
        rec = {"candidate": f"deep:{family}", "family": "deep", "model": family, "seeds_requested": len(deep_seeds), "seeds_ok": len(seed_models), "status": "failed", "seed_status": "|".join(seed_status)}
        if seed_models:
            pcal = np.mean(np.vstack(pcal_seeds), axis=0); pval = np.mean(np.vstack(pval_seeds), axis=0); threshold, econ = choose_validation_threshold(mv, pval, cfg)
            rec.update({"status": "ok", "threshold": threshold, **{f"calibration_{k}": v for k, v in predictive_metrics(yc, pcal).items()}, **{f"validation_{k}": v for k, v in predictive_metrics(yv, pval).items()}, **{f"validation_economic_{k}": v for k, v in econ.items()}})
            conf = conformal_binary_quantile(yc, pcal, cfg.conformal_alpha); rec.update({f"conformal_{k}": v for k, v in conf.items()}); fitted[rec["candidate"]] = {"type": "deep", "models": seed_models, "scaler": scaler, "threshold": threshold, "lookback": lookback, "temporal_config": tcfg}
        records.append(rec); print(json.dumps(rec, default=str))
    return records, fitted


def validation_objective(rec: dict) -> float:
    v = rec.get("validation_economic_validation_objective", -np.inf)
    try: return float(v) if np.isfinite(float(v)) else -np.inf
    except Exception: return -np.inf


def predict_candidate(candidate: dict, rows_all: pd.DataFrame, segment: str):
    if candidate["type"] == "tabular":
        r = split(rows_all, segment); p = calibrated_probability(candidate["calibrator"], raw_score(candidate["model"], feature_matrix(r))); return r, p
    X, y, gross, r = _deep_sequences(rows_all, candidate["scaler"], segment, candidate["lookback"]); probs = []
    for seed, model, calibrator in candidate["models"]:
        rawp = np.clip(predict_probability(model, X), 1e-6, 1 - 1e-6); raw = np.log(rawp / (1 - rawp)); probs.append(calibrated_probability(calibrator, raw))
    return r, np.mean(np.vstack(probs), axis=0)


def save_champion(candidate_name: str, candidate: dict, out: Path) -> None:
    if candidate["type"] == "tabular":
        joblib.dump({"candidate": candidate_name, **candidate}, out / "champion_tabular.joblib", compress=3)
    else:
        import torch
        payload = {"candidate": candidate_name, "threshold": candidate["threshold"], "lookback": candidate["lookback"], "feature_names": list(TABULAR_FEATURES), "scaler_median": candidate["scaler"].median, "scaler_scale": candidate["scaler"].scale, "models": []}
        for seed, model, calibrator in candidate["models"]:
            payload["models"].append({"seed": seed, "state_dict": model.state_dict(), "calibrator": calibrator, "kind": candidate_name.split(":", 1)[1], "config": asdict(candidate["temporal_config"])})
        torch.save(payload, out / "champion_deep.pt")


def main() -> None:
    ap = argparse.ArgumentParser(description="v0.23 strict ML evidence lab")
    ap.add_argument("--output-dir", default="artifacts/v23-ml-evidence-lab"); ap.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS)); ap.add_argument("--bars", type=int, default=4200); ap.add_argument("--external-bars", type=int, default=2600); ap.add_argument("--external-exchange", default="okx"); ap.add_argument("--deep-epochs", type=int, default=6); ap.add_argument("--deep-seeds", default=",".join(map(str, DEEP_SEEDS))); args = ap.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True); cfg = MLV23Config(); symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]; deep_seeds = [int(s) for s in args.deep_seeds.split(",") if s.strip()]
    (out / "protocol.json").write_text(json.dumps(protocol_dict(cfg), indent=2), encoding="utf-8")
    frames = {}
    for s in symbols:
        try:
            x = fetch_coinex(s, args.bars)
            if len(x) >= 500: frames[s] = x
        except Exception as exc: print(json.dumps({"coinex_fetch_failed": s, "error": str(exc)}))
    rows, provenance = build_all_rows(frames, cfg); rows.to_csv(out / "coinex_labeled_rows.csv", index=False); (out / "coinex_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    counts = rows.groupby(["symbol", "segment"]).size().unstack(fill_value=0).to_dict(); (out / "split_manifest.json").write_text(json.dumps({"counts": counts, "segments": rows.segment.value_counts().to_dict()}, indent=2, default=str), encoding="utf-8")
    tab_rec, tab_fit = train_tabular_candidates(rows, cfg, out); deep_rec, deep_fit = train_deep_candidates(rows, cfg, out, args.deep_epochs, deep_seeds); records = tab_rec + deep_rec; fitted = {**tab_fit, **deep_fit}
    board = pd.DataFrame(records); board["selection_score"] = board.apply(lambda r: validation_objective(r.to_dict()), axis=1); board = board.sort_values(["status", "selection_score"], ascending=[False, False], na_position="last"); board.to_csv(out / "validation_leaderboard.csv", index=False)
    eligible = board[(board.status == "ok") & np.isfinite(board.selection_score)]
    if eligible.empty: raise RuntimeError("no valid ML candidate; fail closed")
    champion_name = str(eligible.iloc[0].candidate); champion = fitted[champion_name]; threshold = float(champion["threshold"])
    (out / "champion_freeze.json").write_text(json.dumps({"champion": champion_name, "threshold": threshold, "selection_basis": "validation economic objective only", "final_test_seen_at_selection": False}, indent=2), encoding="utf-8"); save_champion(champion_name, champion, out)
    final_rows, pfinal = predict_candidate(champion, rows, "final_test"); final_pred = predictive_metrics(final_rows.target.to_numpy(int), pfinal); final_metrics, final_eq = portfolio_backtest(final_rows, pfinal, threshold, cfg, cfg.one_way_cost_bps); final_rows.assign(probability=pfinal).to_csv(out / "final_test_predictions_champion.csv", index=False); final_eq.to_csv(out / "final_test_equity_champion.csv", index=False)
    stress = {}
    for c in cfg.stress_one_way_cost_bps: stress[str(c)] = portfolio_backtest(final_rows, pfinal, threshold, cfg, c)[0]
    (out / "final_cost_stress.json").write_text(json.dumps(stress, indent=2, default=str), encoding="utf-8")
    baseline = fitted.get("tabular:logistic"); baseline_metrics = None; comparison = None
    if baseline is not None:
        b_rows, pb = predict_candidate(baseline, rows, "final_test"); bm, beq = portfolio_backtest(b_rows, pb, float(baseline["threshold"]), cfg, cfg.one_way_cost_bps); baseline_metrics = bm
        merged = final_eq[["timestamp", "realized_net_return"]].rename(columns={"realized_net_return": "champion"}).merge(beq[["timestamp", "realized_net_return"]].rename(columns={"realized_net_return": "logistic"}), on="timestamp", how="outer").fillna(0).sort_values("timestamp"); comparison = moving_block_ci(merged.champion - merged.logistic, 1000, 12, cfg.seed); merged.to_csv(out / "final_champion_vs_logistic_period_returns.csv", index=False)
    external_id, external_frames = fetch_external(args.external_exchange, symbols, args.external_bars); external_metrics = None; external_pred = None
    if len(external_frames) >= 3:
        ext_rows, ext_prov = build_all_rows(external_frames, cfg); (out / "external_provenance.json").write_text(json.dumps({"exchange": external_id, "symbols": ext_prov, "note": "venue replication; economically correlated with CoinEx and not an independent asset class"}, indent=2), encoding="utf-8"); erows, pe = predict_candidate(champion, ext_rows, "final_test"); external_pred = predictive_metrics(erows.target.to_numpy(int), pe); external_metrics, eeq = portfolio_backtest(erows, pe, threshold, cfg, cfg.one_way_cost_bps); erows.assign(probability=pe).to_csv(out / "external_final_predictions_champion.csv", index=False); eeq.to_csv(out / "external_final_equity_champion.csv", index=False)
    decision = final_promotion_decision(final_metrics, external_metrics, cfg); stress_60 = stress.get("30.0", {}); stress_pass = bool(float(stress_60.get("total_return", -1)) > 0 and abs(float(stress_60.get("max_drawdown", -1))) <= cfg.max_drawdown and not bool(stress_60.get("kill_switch_triggered", False)))
    if decision["decision"] == "FORWARD_PAPER_ML_CANDIDATE" and not stress_pass: decision["decision"] = "NO_ML_ALPHA_PROMOTION"; decision["stress_gate_failed"] = True
    decision.update({"version": "v0.23", "champion": champion_name, "threshold": threshold, "final_predictive": final_pred, "final_economic": final_metrics, "external_exchange": external_id, "external_predictive": external_pred, "external_economic": external_metrics, "cost_stress": stress, "stress_60bps_pass": stress_pass, "logistic_baseline_final": baseline_metrics, "moving_block_ci_champion_minus_logistic": comparison, "candidate_trials": int(len(eligible)), "multiple_testing_note": "candidate selection used validation only; final test inspected once after champion freeze; DSR/PBO/SPA remain required before any live authorization", "v22c_localization_prerequisite": "PASSED_REPRESENTATION_GATE", "v22d_multimodal_prerequisite": "PASSED_REPRESENTATION_GATE_BUT_NOT_ALPHA_PROOF", "rl_policy": "do not connect vision/multimodal state to RL unless v0.23 economic/external gates and prospective PAPER evidence support it", "live_execution_authorized": False})
    raw = json.dumps(decision, sort_keys=True, default=str).encode(); decision["decision_sha256"] = hashlib.sha256(raw).hexdigest(); (out / "decision.json").write_text(json.dumps(decision, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"champion": champion_name, "decision": decision["decision"], "final": final_metrics, "external": external_metrics, "output": str(out)}, indent=2, default=str))


if __name__ == "__main__": main()
