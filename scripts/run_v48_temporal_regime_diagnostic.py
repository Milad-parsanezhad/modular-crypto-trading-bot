from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_bot.economic_state_v46 import R1, encode_state_v46
from research_bot.event_competing_risk_v41 import V41_FEATURES
from research_bot.temporal_regime_diagnostic_v48 import (
    BH_Q_V48,
    bh_adjust,
    chronological_halves,
    fold_cluster_bootstrap_spearman,
    js_divergence,
    one_sided_sign_p,
    robust_scale_fit,
    standardized_wasserstein,
)

C1 = "C1_TEMPERATURE_R1"
C0 = "C0_IDENTITY_R1"
EXPECTED_V44_RUN = 34700944062
EXPECTED_V47_RUN = 34705324352
EXPECTED_V47_SCIENTIFIC_HEAD = "802b0cde38549617589b6a4a313949361dd9425c"
EXPECTED_V47_ARTIFACT_DIGEST = "sha256:89c824d1bd88cc8d91a3fc6a511f2564e3171c3091c51a6149c5ee988ec80118"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def window_slice(events: pd.DataFrame, row: pd.Series):
    signal = pd.to_datetime(events["signal_time"], utc=True)
    exit_ = pd.to_datetime(events["exit_time"], utc=True)
    cal_start = pd.Timestamp(row["cal_start"])
    pretest_cut = pd.Timestamp(row["pretest_cut"])
    test_start = pd.Timestamp(row["test_start"])
    test_end = pd.Timestamp(row["test_end"])
    fit = events[(signal < cal_start) & (exit_ < cal_start)].copy()
    cal = events[(signal >= cal_start) & (signal < pretest_cut) & (exit_ < test_start)].copy()
    test = events[(signal >= test_start) & (signal <= test_end)].copy()
    return fit, cal, test


def profit_factor(values: pd.Series) -> float | None:
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(x) == 0:
        return None
    gains = float(x[x > 0].sum())
    losses = float(-x[x < 0].sum())
    if losses <= 0:
        return float("inf") if gains > 0 else None
    return gains / losses


def econ_rows(trades: pd.DataFrame) -> list[dict]:
    t = trades.copy()
    t["entry_time"] = pd.to_datetime(t["entry_time"], utc=True)
    t["month"] = t["entry_time"].dt.strftime("%Y-%m")
    specs = [
        ("fold", ["fold"]),
        ("month", ["month"]),
        ("venue", ["venue"]),
        ("regime", ["regime"]),
        ("event_family", ["event_family_v41"]),
        ("fold_x_venue", ["fold", "venue"]),
        ("fold_x_regime", ["fold", "regime"]),
    ]
    rows: list[dict] = []
    for dimension, keys in specs:
        for key, g in t.groupby(keys, dropna=False, sort=True):
            vals = key if isinstance(key, tuple) else (key,)
            row = {"dimension": dimension, "key": " | ".join(map(str, vals))}
            row.update({
                "n": int(len(g)),
                "expectancy_r": float(pd.to_numeric(g["net_r"], errors="coerce").mean()),
                "profit_factor": profit_factor(g["net_r"]),
                "stress_expectancy_r": float(pd.to_numeric(g["stress_net_r"], errors="coerce").mean()),
                "stress_profit_factor": profit_factor(g["stress_net_r"]),
            })
            rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v44-dir", required=True)
    ap.add_argument("--v47-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    v44 = Path(args.v44_dir)
    v47 = Path(args.v47_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Exact v0.44 provenance/hash guard.
    prep = json.loads((v44 / "prep_status_v44.json").read_text())
    ph = json.loads((v44 / "prepared_hashes_v44.json").read_text())
    events_path = v44 / "events_v44.pkl.gz"
    if prep.get("status") != "READY" or prep.get("kraken_touched") is not False:
        raise RuntimeError("v0.44 source not READY or Kraken firewall violated")
    if sha256(events_path) != ph.get("events_v44_pickle_sha256"):
        raise RuntimeError("v0.44 event hash mismatch")

    # Exact v0.47 internal evidence guard.
    required = ["decision_v47.json", "support_manifest_v47.csv", "forecast_metrics_v47.csv", "post_governor_trades_v47.csv.gz", "output_hashes_v47.json", "execution_provenance_v47.json"]
    for name in required:
        if not (v47 / name).is_file():
            raise RuntimeError(f"missing v0.47 file: {name}")
    hashes = json.loads((v47 / "output_hashes_v47.json").read_text())
    for name in ["decision_v47.json", "support_manifest_v47.csv", "forecast_metrics_v47.csv", "post_governor_trades_v47.csv.gz"]:
        if hashes.get(name) != sha256(v47 / name):
            raise RuntimeError(f"v0.47 file hash mismatch: {name}")
    prov = json.loads((v47 / "execution_provenance_v47.json").read_text())
    if int(prov.get("workflow_run_id")) != EXPECTED_V47_RUN:
        raise RuntimeError("unexpected canonical v0.47 workflow run")
    if prov.get("head_sha") != EXPECTED_V47_SCIENTIFIC_HEAD:
        raise RuntimeError("unexpected canonical v0.47 scientific head")
    if prov.get("kraken_touched") is not False:
        raise RuntimeError("canonical v0.47 touched Kraken")

    events = pd.read_pickle(events_path, compression="gzip")
    folds = pd.read_csv(v44 / "fold_index_v44.csv", parse_dates=["cal_start", "pretest_cut", "test_start", "test_end"])
    support = pd.read_csv(v47 / "support_manifest_v47.csv")
    metrics = pd.read_csv(v47 / "forecast_metrics_v47.csv")
    trades = pd.read_csv(v47 / "post_governor_trades_v47.csv.gz", low_memory=False)

    support_bool = support["supported"].astype(str).str.lower().isin(["true", "1"])
    canonical = support.loc[support["arm"].astype(str).eq(C1) & support_bool, ["fold", "symbol"]].drop_duplicates()
    canonical = canonical.sort_values(["fold", "symbol"]).reset_index(drop=True)
    if len(canonical) != 120:
        raise RuntimeError(f"unexpected C1 supported-unit count: {len(canonical)}")

    missing = sorted(set(V41_FEATURES) - set(events.columns))
    needed = {"signal_time", "exit_time", "symbol", "outcome", "regime", "event_family_v41", "side", "venue"}
    if missing or not needed.issubset(events.columns):
        raise RuntimeError(f"v0.44 event schema mismatch: missing_features={missing}, missing_needed={sorted(needed-set(events.columns))}")

    unit_rows: list[dict] = []
    for fold_no in range(1, 6):
        fr = folds.loc[folds["fold"].astype(int).eq(fold_no)].iloc[0]
        fit_all, cal_all, test_all = window_slice(events, fr)
        for symbol in canonical.loc[canonical["fold"].astype(int).eq(fold_no), "symbol"].astype(str):
            fit = fit_all.loc[fit_all["symbol"].astype(str).eq(symbol)].copy()
            cal = cal_all.loc[cal_all["symbol"].astype(str).eq(symbol)].copy()
            test = test_all.loc[test_all["symbol"].astype(str).eq(symbol)].copy()
            early, late = chronological_halves(cal, "signal_time")
            if min(len(fit), len(early), len(late), len(test)) <= 0:
                raise RuntimeError(f"empty diagnostic partition for fold={fold_no} symbol={symbol}")

            feature_excess: list[float] = []
            for feature in V41_FEATURES:
                scale = robust_scale_fit(fit[feature])
                internal = standardized_wasserstein(early[feature], late[feature], scale)
                forward = standardized_wasserstein(late[feature], test[feature], scale)
                if internal is not None and forward is not None:
                    feature_excess.append(float(forward - internal))
            if not feature_excess:
                raise RuntimeError(f"no finite drift features for fold={fold_no} symbol={symbol}")

            early_state = encode_state_v46(early, R1).astype(str)
            late_state = encode_state_v46(late, R1).astype(str)
            test_state = encode_state_v46(test, R1).astype(str)

            def excess_js(a, b, c):
                x = js_divergence(a, b)
                y = js_divergence(b, c)
                return np.nan if x is None or y is None else float(y - x)

            row = {
                "fold": fold_no,
                "symbol": symbol,
                "fit_n": int(len(fit)),
                "cal_early_n": int(len(early)),
                "cal_late_n": int(len(late)),
                "test_n": int(len(test)),
                "continuous_excess_drift": float(np.median(feature_excess)),
                "r1_state_excess_js": excess_js(early_state, late_state, test_state),
                "common_outcome_excess_js": excess_js(early["outcome"].astype(str), late["outcome"].astype(str), test["outcome"].astype(str)),
            }
            for col, name in [("regime", "regime"), ("event_family_v41", "event_family"), ("side", "side"), ("venue", "venue")]:
                row[f"{name}_excess_js"] = excess_js(early[col], late[col], test[col])
            unit_rows.append(row)

    units = pd.DataFrame(unit_rows).sort_values(["fold", "symbol"]).reset_index(drop=True)
    if len(units) != len(canonical):
        raise RuntimeError("unit diagnostic coverage differs from canonical v0.47 C1 coverage")

    families = {
        "continuous": "continuous_excess_drift",
        "r1_state": "r1_state_excess_js",
        "common_outcome": "common_outcome_excess_js",
        "regime": "regime_excess_js",
        "event_family": "event_family_excess_js",
        "side": "side_excess_js",
        "venue": "venue_excess_js",
    }
    test_rows: list[dict] = []
    for family, col in families.items():
        tmp = []
        for fold_no in range(1, 6):
            vals = pd.to_numeric(units.loc[units["fold"].eq(fold_no), col], errors="coerce")
            k, n, p = one_sided_sign_p(vals)
            tmp.append({
                "family": family,
                "metric": col,
                "fold": fold_no,
                "median_excess": float(vals.median()),
                "positive_units": k,
                "nonzero_units": n,
                "p_value": p,
            })
        q = bh_adjust([r["p_value"] for r in tmp])
        for r, qv in zip(tmp, q):
            r["q_value"] = float(qv)
            r["fold_positive"] = bool(r["median_excess"] > 0 and qv <= BH_Q_V48)
            test_rows.append(r)
    fold_tests = pd.DataFrame(test_rows)

    # Link drift to canonical v0.47 forecast quality.
    c1_metrics = metrics.loc[metrics["arm"].astype(str).eq(C1)].copy()
    linked = units.merge(c1_metrics, on=["fold", "symbol"], how="inner", validate="one_to_one")
    if len(linked) != len(units):
        raise RuntimeError("forecast-link join lost canonical C1 units")
    links = []
    for metric, harmful in [
        ("common_multiclass_brier", "positive"),
        ("common_mean_reliability", "positive"),
        ("common_macro_ovr_auc", "negative"),
    ]:
        d = fold_cluster_bootstrap_spearman(linked, "continuous_excess_drift", metric)
        harmful_ci = bool(
            (harmful == "positive" and d["ci90_low"] is not None and d["ci90_low"] > 0)
            or (harmful == "negative" and d["ci90_high"] is not None and d["ci90_high"] < 0)
        )
        links.append({"metric": metric, "harmful_direction": harmful, **d, "harmful_ci_excludes_zero": harmful_ci})
    link_df = pd.DataFrame(links)

    # Frozen economic decompositions from canonical C1 trades.
    executed = trades.loc[trades["arm"].astype(str).eq(C1) & trades["executed"].astype(str).str.lower().isin(["true", "1"])].copy()
    econ = pd.DataFrame(econ_rows(executed))

    positive_counts = fold_tests.groupby("family")["fold_positive"].sum().astype(int).to_dict()
    continuous_ok = int(positive_counts.get("continuous", 0)) >= 3
    secondary_ok = any(int(positive_counts.get(f, 0)) >= 3 for f in families if f != "continuous")
    link_ok = bool(link_df["harmful_ci_excludes_zero"].any())
    if continuous_ok and secondary_ok and link_ok:
        decision = "V48_DIAGNOSTIC_SUPPORTS_TEMPORAL_DISTRIBUTION_SHIFT"
    elif continuous_ok and secondary_ok:
        decision = "V48_SHIFT_PRESENT_LINK_TO_FORECAST_FAILURE_INCONCLUSIVE"
    else:
        decision = "V48_NONSTATIONARITY_EVIDENCE_INCONCLUSIVE"

    decision_obj = {
        "version": "v0.48",
        "experiment": "TEMPORAL_REGIME_NONSTATIONARITY_DIAGNOSTIC",
        "decision": decision,
        "candidate_promotion_allowed": False,
        "primary_diagnostic_arm": C1,
        "sensitivity_reference_arm": C0,
        "supported_units": int(len(units)),
        "positive_fold_counts": {k: int(v) for k, v in positive_counts.items()},
        "continuous_condition": continuous_ok,
        "secondary_condition": secondary_ok,
        "forecast_link_condition": link_ok,
        "kraken_touched": False,
        "paper_execution": False,
        "live_execution": False,
        "post_result_pruning": False,
        "source_v44_run": EXPECTED_V44_RUN,
        "source_v47_run": EXPECTED_V47_RUN,
        "source_v47_scientific_head": EXPECTED_V47_SCIENTIFIC_HEAD,
        "source_v47_artifact_digest": EXPECTED_V47_ARTIFACT_DIGEST,
    }

    units.to_csv(out / "unit_drift_v48.csv", index=False)
    fold_tests.to_csv(out / "fold_drift_tests_v48.csv", index=False)
    link_df.to_csv(out / "forecast_link_v48.csv", index=False)
    econ.to_csv(out / "economic_decomposition_v48.csv", index=False)
    (out / "decision_v48.json").write_text(json.dumps(decision_obj, indent=2, sort_keys=True))

    output_hashes = {}
    for path in sorted(out.iterdir()):
        if path.is_file() and path.name != "output_hashes_v48.json":
            output_hashes[path.name] = sha256(path)
    (out / "output_hashes_v48.json").write_text(json.dumps(output_hashes, indent=2, sort_keys=True))
    print(json.dumps(decision_obj, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
