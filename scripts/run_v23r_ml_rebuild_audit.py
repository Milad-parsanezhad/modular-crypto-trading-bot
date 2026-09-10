from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer

from research_bot.coinex_public import PERIOD_MS, fetch_coinex_klines, utc_now_ms
from research_bot.ml_audit_v23r import audit_dataset, write_audit_bundle
from research_bot.ml_framework_v23r import (
    MLResearchContract,
    SplitContract,
    chronological_purged_split,
    choose_validation_threshold,
    classification_metrics,
    cost_aware_next_open_labels,
    economic_metrics,
    make_preprocessor,
    probability_score,
    research_candidate_decision,
    select_feature_columns,
    supervised_model_registry,
    unsupervised_model_registry,
    write_contract_manifest,
)

PERIOD = "4hour"
TIMEFRAME = "4h"
DEFAULT_SYMBOLS = ("BTC/USDT", "ETH/USDT", "SOL/USDT")


def fetch_closed(symbol: str, bars: int) -> pd.DataFrame:
    x = fetch_coinex_klines(symbol=symbol, period=PERIOD, market_type="spot", end_ms=utc_now_ms(), bars=bars)
    x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True)
    closed = x["timestamp"] + pd.Timedelta(milliseconds=PERIOD_MS[PERIOD]) <= pd.Timestamp.now(tz="UTC")
    return x[closed].sort_values("timestamp").reset_index(drop=True)


def build_causal_features(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.copy().sort_values("timestamp").reset_index(drop=True)
    close = pd.to_numeric(x["close"], errors="coerce")
    high = pd.to_numeric(x["high"], errors="coerce")
    low = pd.to_numeric(x["low"], errors="coerce")
    volume = pd.to_numeric(x["volume"], errors="coerce")
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr14 = tr.rolling(14, min_periods=14).mean()
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()
    out = pd.DataFrame({"timestamp": x["timestamp"]})
    out["f_ret1"] = close.pct_change(1)
    out["f_ret6"] = close.pct_change(6)
    out["f_ret24"] = close.pct_change(24)
    out["f_realized_vol20"] = close.pct_change().shift(1).rolling(20, min_periods=20).std()
    out["f_atr14_pct"] = atr14 / close.replace(0, np.nan)
    out["f_ema20_rel"] = ema20 / close.replace(0, np.nan) - 1.0
    out["f_ema50_rel"] = ema50 / close.replace(0, np.nan) - 1.0
    out["f_ema200_rel"] = ema200 / close.replace(0, np.nan) - 1.0
    out["f_trend_20_50"] = (ema20 - ema50) / atr14.replace(0, np.nan)
    out["f_trend_50_200"] = (ema50 - ema200) / atr14.replace(0, np.nan)
    out["f_volume_ratio20"] = volume / volume.shift(1).rolling(20, min_periods=20).median().replace(0, np.nan)
    out["f_range_atr"] = (high - low) / atr14.replace(0, np.nan)
    out["f_body_atr"] = (close - pd.to_numeric(x["open"], errors="coerce")) / atr14.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def assemble_symbol(symbol: str, bars: int, contract: MLResearchContract) -> pd.DataFrame:
    raw = fetch_closed(symbol, bars)
    if len(raw) < 500:
        raise RuntimeError(f"{symbol}: insufficient closed bars={len(raw)}")
    features = build_causal_features(raw)
    labels = cost_aware_next_open_labels(raw[["timestamp", "open"]], contract.cost)
    merged = labels[[
        "signal_time", "label_end_time", "label_future_gross_return",
        "label_future_net_return", "label_positive_net", "label_direction_3class",
    ]].merge(features.rename(columns={"timestamp": "signal_time"}), on="signal_time", how="left", validate="one_to_one")
    merged["symbol"] = symbol
    merged["timeframe"] = TIMEFRAME
    merged["side"] = 1
    return merged


def fit_supervised(splits: dict[str, pd.DataFrame], contract: MLResearchContract) -> tuple[pd.DataFrame, dict]:
    dev, val, test = splits["development"], splits["validation"], splits["test"]
    numeric, categorical = select_feature_columns(dev, include_context=True, include_identity=False)
    xcols = numeric + categorical
    rows = []
    for seed in contract.search.seeds:
        for name, estimator in supervised_model_registry(seed).items():
            pipe = Pipeline([("prep", make_preprocessor(numeric, categorical)), ("model", clone(estimator))])
            status, error = "ok", ""
            try:
                pipe.fit(dev[xcols], dev["label_positive_net"].astype(int))
                pv = probability_score(pipe, val[xcols])
                threshold, econ_v = choose_validation_threshold(
                    pv, val["label_future_net_return"], search=contract.search, risk=contract.risk,
                )
                pt = probability_score(pipe, test[xcols])
                econ_t = economic_metrics(test["label_future_net_return"], pt >= threshold)
                cm_v = classification_metrics(val["label_positive_net"], pv)
                cm_t = classification_metrics(test["label_positive_net"], pt)
                rows.append({
                    "model": name, "seed": seed, "status": status, "threshold": threshold,
                    **{f"validation_{k}": v for k, v in cm_v.items()},
                    **{f"validation_{k}": v for k, v in econ_v.items()},
                    **{f"test_{k}": v for k, v in cm_t.items()},
                    **{f"test_{k}": v for k, v in econ_t.items()},
                })
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                rows.append({"model": name, "seed": seed, "status": "failed", "error": error})
    board = pd.DataFrame(rows)
    ok = board[board["status"] == "ok"].copy()
    if ok.empty:
        return board, {"decision": "NO_ML_MODEL_PROMOTED", "reason": "all supervised fits failed", "live_execution_authorized": False}
    family = ok.groupby("model", as_index=False).agg(
        validation_objective_mean=("validation_validation_objective", "mean"),
        validation_objective_std=("validation_validation_objective", "std"),
        validation_auc_mean=("validation_roc_auc", "mean"),
        test_auc_mean=("test_roc_auc", "mean"),
        test_mean_net_return_mean=("test_mean_net_return", "mean"),
        test_profit_factor_mean=("test_profit_factor", "mean"),
        test_selected_mean=("test_selected", "mean"),
        seeds=("seed", "nunique"),
    ).sort_values(["validation_objective_mean", "validation_objective_std"], ascending=[False, True])
    champion_name = str(family.iloc[0]["model"])
    champion_seed_row = ok[ok["model"] == champion_name].sort_values("validation_validation_objective", ascending=False).iloc[0]
    test_metrics = {
        "selected": int(champion_seed_row["test_selected"]),
        "mean_net_return": float(champion_seed_row["test_mean_net_return"]),
        "profit_factor": float(champion_seed_row["test_profit_factor"]),
    }
    decision = {
        **research_candidate_decision(test_metrics, risk=contract.risk),
        "champion_family": champion_name,
        "champion_seed": int(champion_seed_row["seed"]),
        "threshold": float(champion_seed_row["threshold"]),
        "selection_basis": "model family by mean validation objective across seeds; seed by validation objective; test read only after freeze",
        "family_summary": family.to_dict(orient="records"),
    }
    return board, decision


def fit_unsupervised(dev: pd.DataFrame) -> pd.DataFrame:
    numeric, _ = select_feature_columns(dev, include_context=False, include_identity=False)
    x = dev[numeric].copy()
    prep = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", RobustScaler())])
    z = prep.fit_transform(x)
    rows = []
    for name, model in unsupervised_model_registry(314).items():
        try:
            if name == "isolation_forest":
                label = model.fit_predict(z)
            else:
                label = model.fit_predict(z)
            values, counts = np.unique(label, return_counts=True)
            rows.append({
                "model": name, "status": "ok", "development_rows": int(len(z)),
                "state_count": int(len(values)), "state_distribution": json.dumps({str(int(k)): int(v) for k, v in zip(values, counts)}),
                "role": "regime_or_anomaly_diagnostic_only",
                "alpha_authorized": False,
            })
        except Exception as exc:
            rows.append({"model": name, "status": "failed", "error": f"{type(exc).__name__}: {exc}", "alpha_authorized": False})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Clean v0.23r ML framework rebuild + audit")
    ap.add_argument("--output-dir", default="artifacts/v23r-ml-rebuild-audit")
    ap.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    ap.add_argument("--bars", type=int, default=2600)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    contract = MLResearchContract(split=SplitContract(embargo_rows=2))
    write_contract_manifest(out, contract, extra={"baseline_commit": "2a5b62c84354b9cbffd35c0bdd6591d4219bf216", "track": "clean_rebuild_audit2"})
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    parts, provenance = [], {}
    for symbol in symbols:
        try:
            part = assemble_symbol(symbol, args.bars, contract)
            parts.append(part)
            provenance[symbol] = {"rows": int(len(part)), "first": str(part.signal_time.min()), "last": str(part.signal_time.max())}
        except Exception as exc:
            provenance[symbol] = {"error": f"{type(exc).__name__}: {exc}"}
    if not parts:
        raise RuntimeError("no real-data ML rows available")
    dataset = pd.concat(parts, ignore_index=True).sort_values(["signal_time", "symbol"]).reset_index(drop=True)
    splits = chronological_purged_split(dataset, label_end_time_col="label_end_time", contract=contract.split)
    tagged = []
    for segment, part in splits.items():
        z = part.copy(); z["segment"] = segment; tagged.append(z)
    audited = pd.concat(tagged, ignore_index=True).sort_values(["signal_time", "symbol"]).reset_index(drop=True)
    audited.to_csv(out / "labeled_dataset.csv", index=False)
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2, default=str), encoding="utf-8")

    audit_report, audit_summary = audit_dataset(audited, include_identity_features=False)
    write_audit_bundle(out, audit_report, audit_summary, name="ml_dataset_audit")
    if audit_summary["decision"] != "AUDIT_PASS":
        raise RuntimeError(f"ML dataset audit failed: {audit_summary['fatal_codes']}")

    split_map = {k: audited[audited.segment == k].copy() for k in ("development", "validation", "test")}
    board, supervised_decision = fit_supervised(split_map, contract)
    board.to_csv(out / "supervised_per_seed_leaderboard.csv", index=False)
    unsup = fit_unsupervised(split_map["development"])
    unsup.to_csv(out / "unsupervised_diagnostics.csv", index=False)

    final = {
        "version": "v0.23r",
        "stage": "CLEAN_ML_FRAMEWORK_REBUILD_AND_AUDIT",
        "dataset_audit": audit_summary,
        "supervised": supervised_decision,
        "unsupervised_models_succeeded": int((unsup.status == "ok").sum()) if not unsup.empty else 0,
        "deep_temporal_track": "separate; no mixing tabular test selection with LSTM/GRU/TCN/Transformer",
        "vision_track": "v0.22c/v0.22d evidence preserved; not connected to RL here",
        "forward_paper_authorized": False,
        "live_execution_authorized": False,
    }
    (out / "decision.json").write_text(json.dumps(final, indent=2, default=str), encoding="utf-8")
    print(json.dumps(final, indent=2, default=str))


if __name__ == "__main__":
    main()
