from __future__ import annotations

"""Execute the frozen v0.50 expected-R / economic utility-mapping diagnostic."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_bot.economic_state_v46 import R1, R1_STATES, allocate_portfolio_risk_writable_v46, encode_state_v46
from research_bot.expected_r_utility_mapping_v50 import (
    C1,
    MIN_RANKING_ROWS_V50,
    MIN_STATE_COUNT_V50,
    MIN_SUPPORTED_UNITS_V50,
    broad_harmful_stage_v50,
    fold_direction_tests_v50,
    profit_factor_v50,
    ranking_diagnostic_v50,
    route_decision_v50,
    state_transport_metrics_v50,
)

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_V44_RUN = 34700944062
EXPECTED_V47_RUN = 34705324352
EXPECTED_V47_HEAD = "802b0cde38549617589b6a4a313949361dd9425c"
EXPECTED_V47_DIGEST = "sha256:89c824d1bd88cc8d91a3fc6a511f2564e3171c3091c51a6149c5ee988ec80118"
EXPECTED_V49_RUN = 34707339422


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _load(ROOT / "scripts" / "run_v39_development_characterization.py", "v39_base_for_v50")
corrected = _load(ROOT / "scripts" / "run_v39_development_characterization_corrected.py", "v39_corrected_for_v50")
base.allocate_portfolio_risk = allocate_portfolio_risk_writable_v46
corrected.base.allocate_portfolio_risk = allocate_portfolio_risk_writable_v46


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


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1"])


def _stage_summary(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"n": 0, "expectancy_r": None, "profit_factor": None}
    x = pd.to_numeric(frame["net_r"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return {
        "n": int(len(x)),
        "expectancy_r": float(x.mean()) if len(x) else None,
        "profit_factor": profit_factor_v50(x.to_numpy(dtype=float)),
    }


def _context_rows(pred: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    specs = [
        ("venue", "venue"),
        ("side", "side"),
        ("regime", "regime"),
        ("event_family", "event_family_v41"),
    ]
    for dimension, col in specs:
        for key, g in pred.groupby(col, dropna=False, sort=True):
            d = ranking_diagnostic_v50(g)
            rows.append({
                "dimension": dimension,
                "key": str(key),
                "n": d.n,
                "spearman_rho": d.spearman_rho,
                "top_bottom_spread": d.top_bottom_spread,
                "ols_intercept": d.ols_intercept,
                "ols_slope": d.ols_slope,
                "mean_bias": d.mean_bias,
            })
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

    # v0.44 exact prepared evidence.
    prep = json.loads((v44 / "prep_status_v44.json").read_text(encoding="utf-8"))
    ph = json.loads((v44 / "prepared_hashes_v44.json").read_text(encoding="utf-8"))
    events_path = v44 / "events_v44.pkl.gz"
    if prep.get("status") != "READY" or prep.get("kraken_touched") is not False:
        raise RuntimeError("v0.44 source not READY or Kraken firewall violated")
    if sha256(events_path) != ph.get("events_v44_pickle_sha256"):
        raise RuntimeError("v0.44 event-table hash mismatch")

    # v0.47 exact canonical artifact internals.
    required_v47 = [
        "decision_v47.json",
        "execution_provenance_v47.json",
        "support_manifest_v47.csv",
        "oos_predictions_v47.csv.gz",
        "post_governor_trades_v47.csv.gz",
        "fold_economics_v47.csv",
        "output_hashes_v47.json",
    ]
    for name in required_v47:
        if not (v47 / name).is_file():
            raise RuntimeError(f"missing canonical v0.47 file: {name}")
    hashes = json.loads((v47 / "output_hashes_v47.json").read_text(encoding="utf-8"))
    for name in ["decision_v47.json", "support_manifest_v47.csv", "oos_predictions_v47.csv.gz", "post_governor_trades_v47.csv.gz", "fold_economics_v47.csv"]:
        if hashes.get(name) != sha256(v47 / name):
            raise RuntimeError(f"v0.47 internal file hash mismatch: {name}")
    prov = json.loads((v47 / "execution_provenance_v47.json").read_text(encoding="utf-8"))
    if int(prov.get("workflow_run_id")) != EXPECTED_V47_RUN:
        raise RuntimeError("unexpected v0.47 workflow run")
    if prov.get("head_sha") != EXPECTED_V47_HEAD:
        raise RuntimeError("unexpected v0.47 scientific head")
    if prov.get("kraken_touched") is not False:
        raise RuntimeError("canonical v0.47 touched Kraken")
    decision47 = json.loads((v47 / "decision_v47.json").read_text(encoding="utf-8"))
    if int(decision47.get("arms", {}).get(C1, {}).get("supported_asset_folds", -1)) != 120:
        raise RuntimeError("canonical v0.47 C1 support is not 120 units")

    events = pd.read_pickle(events_path, compression="gzip")
    folds = pd.read_csv(v44 / "fold_index_v44.csv", parse_dates=["cal_start", "pretest_cut", "test_start", "test_end"])
    if len(folds) != 5 or sorted(folds["fold"].astype(int).tolist()) != [1, 2, 3, 4, 5]:
        raise RuntimeError("v0.50 requires exact five frozen folds")

    support = pd.read_csv(v47 / "support_manifest_v47.csv")
    support_ok = _as_bool(support["supported"])
    canonical_units = support.loc[support["arm"].astype(str).eq(C1) & support_ok, ["fold", "symbol"]].drop_duplicates()
    canonical_units = canonical_units.sort_values(["fold", "symbol"]).reset_index(drop=True)
    if len(canonical_units) != 120:
        raise RuntimeError(f"unexpected canonical C1 unit count: {len(canonical_units)}")

    predictions = pd.read_csv(v47 / "oos_predictions_v47.csv.gz", low_memory=False)
    pred = predictions.loc[predictions["arm"].astype(str).eq(C1)].copy()
    for col in ["signal_time", "entry_time", "exit_time"]:
        pred[col] = pd.to_datetime(pred[col], utc=True, errors="raise")
    pred["selected_model"] = _as_bool(pred["selected_model"])
    pred["expected_r_v47"] = pd.to_numeric(pred["expected_r_v47"], errors="raise")
    pred["net_r"] = pd.to_numeric(pred["net_r"], errors="raise")
    if pred[["fold", "symbol"]].drop_duplicates().sort_values(["fold", "symbol"]).reset_index(drop=True).equals(canonical_units) is False:
        raise RuntimeError("canonical C1 prediction units differ from support manifest")

    # Family A: FIT state-utility transport.
    unit_state_rows: list[dict] = []
    state_detail_rows: list[dict] = []
    for fold_no in range(1, 6):
        fr = folds.loc[folds["fold"].astype(int).eq(fold_no)].iloc[0]
        fit_all, cal_all, test_all = window_slice(events, fr)
        for symbol in canonical_units.loc[canonical_units["fold"].astype(int).eq(fold_no), "symbol"].astype(str):
            parts = {
                "FIT": fit_all.loc[fit_all["symbol"].astype(str).eq(symbol)].copy(),
                "CAL": cal_all.loc[cal_all["symbol"].astype(str).eq(symbol)].copy(),
                "TEST": test_all.loc[test_all["symbol"].astype(str).eq(symbol)].copy(),
            }
            counts: dict[str, dict[str, int]] = {}
            means: dict[str, dict[str, float]] = {}
            for part_name, frame in parts.items():
                state = encode_state_v46(frame, R1).astype(str)
                tmp = frame.assign(state_v50=state)
                c = tmp.groupby("state_v50")["net_r"].size().to_dict()
                m = tmp.groupby("state_v50")["net_r"].mean().to_dict()
                counts[part_name] = {str(k): int(v) for k, v in c.items()}
                means[part_name] = {str(k): float(v) for k, v in m.items()}
            eligible = [
                state for state in R1_STATES
                if all(counts[p].get(state, 0) >= MIN_STATE_COUNT_V50 for p in ("FIT", "CAL", "TEST"))
            ]
            diag = state_transport_metrics_v50(means["FIT"], means["CAL"], means["TEST"], eligible)
            unit_state_rows.append({"fold": fold_no, "symbol": symbol, **diag})
            for state in R1_STATES:
                state_detail_rows.append({
                    "fold": fold_no,
                    "symbol": symbol,
                    "state": state,
                    "eligible": state in eligible,
                    "fit_n": counts["FIT"].get(state, 0),
                    "cal_n": counts["CAL"].get(state, 0),
                    "test_n": counts["TEST"].get(state, 0),
                    "fit_mean_r": means["FIT"].get(state),
                    "cal_mean_r": means["CAL"].get(state),
                    "test_mean_r": means["TEST"].get(state),
                    "fit_to_cal_bias": None if state not in means["FIT"] or state not in means["CAL"] else means["FIT"][state] - means["CAL"][state],
                    "fit_to_test_bias": None if state not in means["FIT"] or state not in means["TEST"] else means["FIT"][state] - means["TEST"][state],
                })
    state_units = pd.DataFrame(unit_state_rows)
    supported_state_units = state_units.loc[state_units["supported"].astype(bool)].copy()
    if len(supported_state_units) < MIN_SUPPORTED_UNITS_V50:
        raise RuntimeError(f"insufficient state-transport unit support: {len(supported_state_units)}")
    state_tests = fold_direction_tests_v50(supported_state_units, "state_transport_excess")
    state_failure = int(state_tests["positive_fold"].sum()) >= 3

    # Family B: exact canonical expected-R ranking versus realized net-R.
    ranking_rows: list[dict] = []
    for unit in canonical_units.itertuples(index=False):
        g = pred.loc[pred["fold"].astype(int).eq(int(unit.fold)) & pred["symbol"].astype(str).eq(str(unit.symbol))].copy()
        d = ranking_diagnostic_v50(g)
        ranking_rows.append({
            "fold": int(unit.fold), "symbol": str(unit.symbol),
            "n": d.n, "spearman_rho": d.spearman_rho,
            "top_bottom_spread": d.top_bottom_spread,
            "ols_intercept": d.ols_intercept, "ols_slope": d.ols_slope,
            "mean_bias": d.mean_bias,
        })
    ranking = pd.DataFrame(ranking_rows)
    supported_ranking = ranking.loc[ranking["spearman_rho"].notna() & ranking["top_bottom_spread"].notna()].copy()
    if len(supported_ranking) < MIN_SUPPORTED_UNITS_V50:
        raise RuntimeError(f"insufficient ranking unit support: {len(supported_ranking)}")
    rho_tests = fold_direction_tests_v50(supported_ranking, "spearman_rho")
    spread_tests = fold_direction_tests_v50(supported_ranking, "top_bottom_spread")
    ranking_fold_rows: list[dict] = []
    for fold_no in range(1, 6):
        r = rho_tests.loc[rho_tests["fold"].eq(fold_no)].iloc[0]
        s = spread_tests.loc[spread_tests["fold"].eq(fold_no)].iloc[0]
        ranking_fold_rows.append({
            "fold": fold_no,
            "rho_median": r["median"], "rho_q_positive": r["q_positive"], "rho_q_negative": r["q_negative"],
            "spread_median": s["median"], "spread_q_positive": s["q_positive"], "spread_q_negative": s["q_negative"],
            "ranking_preserved": bool(r["positive_fold"] and s["positive_fold"]),
            "ranking_harmful": bool(r["negative_fold"] or s["negative_fold"]),
        })
    ranking_folds = pd.DataFrame(ranking_fold_rows)
    ranking_failure = int(ranking_folds["ranking_harmful"].sum()) >= 3

    # Family C: exact pipeline attribution from canonical C1 predictions.
    fold_econ47 = pd.read_csv(v47 / "fold_economics_v47.csv")
    canonical_counts = fold_econ47.loc[fold_econ47["arm"].astype(str).eq(C1), ["fold", "n"]].copy()
    canonical_counts["fold"] = canonical_counts["fold"].astype(int)
    canonical_counts["n"] = canonical_counts["n"].astype(int)
    if sorted(canonical_counts["fold"].tolist()) != [1, 2, 3, 4, 5]:
        raise RuntimeError("canonical v0.47 fold economics incomplete")

    stage_rows: list[dict] = []
    admission_deltas: list[float] = []
    nonoverlap_deltas: list[float] = []
    governor_deltas: list[float] = []
    for fold_no in range(1, 6):
        s0 = pred.loc[pred["fold"].astype(int).eq(fold_no)].copy()
        s1 = s0.loc[s0["selected_model"].astype(bool)].copy()
        s2 = base._realize_nonoverlap(s1) if not s1.empty else s1
        governed = corrected._financial_simulation_independent_by_venue(s2) if not s2.empty else s2
        s3 = governed.loc[governed["executed"].astype(bool)].copy() if not governed.empty else governed
        expected_n = int(canonical_counts.loc[canonical_counts["fold"].eq(fold_no), "n"].iloc[0])
        if len(s3) != expected_n:
            raise RuntimeError(f"Financial Governor reproduction mismatch fold {fold_no}: {len(s3)} != {expected_n}")
        summaries = {
            "ALL_TEST": _stage_summary(s0),
            "EXPECTED_R_POSITIVE": _stage_summary(s1),
            "NONOVERLAP": _stage_summary(s2),
            "GOVERNOR_EXECUTED": _stage_summary(s3),
        }
        for stage, vals in summaries.items():
            stage_rows.append({"fold": fold_no, "stage": stage, **vals})
        e0 = summaries["ALL_TEST"]["expectancy_r"]
        e1 = summaries["EXPECTED_R_POSITIVE"]["expectancy_r"]
        e2 = summaries["NONOVERLAP"]["expectancy_r"]
        e3 = summaries["GOVERNOR_EXECUTED"]["expectancy_r"]
        if any(v is None for v in (e0, e1, e2, e3)):
            raise RuntimeError(f"empty v0.50 pipeline stage in fold {fold_no}")
        admission_deltas.append(float(e1 - e0))
        nonoverlap_deltas.append(float(e2 - e1))
        governor_deltas.append(float(e3 - e2))

    stages = pd.DataFrame(stage_rows)
    admission_failure = broad_harmful_stage_v50(admission_deltas)
    nonoverlap_failure = broad_harmful_stage_v50(nonoverlap_deltas)
    governor_failure = broad_harmful_stage_v50(governor_deltas)

    decision = route_decision_v50(
        state_failure=state_failure,
        ranking_failure=ranking_failure,
        admission_failure=admission_failure,
        nonoverlap_failure=nonoverlap_failure,
        governor_failure=governor_failure,
    )

    # Context decomposition is diagnostic only.
    context = pd.DataFrame(_context_rows(pred))
    state_detail = pd.DataFrame(state_detail_rows)
    state_tests.insert(0, "family", "state_transport")
    rho_tests.insert(0, "family", "expected_r_spearman")
    spread_tests.insert(0, "family", "top_bottom_spread")
    fold_tests = pd.concat([state_tests, rho_tests, spread_tests], ignore_index=True, sort=False)

    decision_obj = {
        "version": "v0.50",
        "experiment": "EXPECTED_R_ECONOMIC_UTILITY_MAPPING_FAILURE_ATTRIBUTION",
        "decision": decision,
        "candidate_promotion_allowed": False,
        "supported_state_units": int(len(supported_state_units)),
        "supported_ranking_units": int(len(supported_ranking)),
        "conditions": {
            "state_utility_transport_failure_supported": bool(state_failure),
            "expected_r_ranking_failure_supported": bool(ranking_failure),
            "expected_r_admission_failure_supported": bool(admission_failure),
            "nonoverlap_failure_supported": bool(nonoverlap_failure),
            "governor_failure_supported": bool(governor_failure),
        },
        "positive_state_transport_folds": int(state_tests["positive_fold"].sum()),
        "harmful_ranking_folds": int(ranking_folds["ranking_harmful"].sum()),
        "preserved_ranking_folds": int(ranking_folds["ranking_preserved"].sum()),
        "admission_deltas_by_fold": admission_deltas,
        "nonoverlap_deltas_by_fold": nonoverlap_deltas,
        "governor_deltas_by_fold": governor_deltas,
        "canonical_governor_counts_reproduced": True,
        "primary_expected_r_source": "EXACT_CANONICAL_V47_C1_ARTIFACT_NO_REFIT",
        "source_v44_run": EXPECTED_V44_RUN,
        "source_v47_run": EXPECTED_V47_RUN,
        "source_v47_scientific_head": EXPECTED_V47_HEAD,
        "source_v47_artifact_digest": EXPECTED_V47_DIGEST,
        "source_v49_run": EXPECTED_V49_RUN,
        "kraken_touched": False,
        "paper_execution": False,
        "live_execution": False,
        "post_result_threshold_tuning": False,
        "post_result_pruning": False,
    }

    state_units.to_csv(out / "state_utility_transport_v50.csv", index=False)
    state_detail.to_csv(out / "state_utility_detail_v50.csv", index=False)
    ranking.to_csv(out / "expected_r_ranking_v50.csv", index=False)
    ranking_folds.to_csv(out / "ranking_fold_summary_v50.csv", index=False)
    fold_tests.to_csv(out / "fold_direction_tests_v50.csv", index=False)
    stages.to_csv(out / "pipeline_stages_v50.csv", index=False)
    context.to_csv(out / "context_decomposition_v50.csv", index=False)
    (out / "decision_v50.json").write_text(json.dumps(decision_obj, indent=2, sort_keys=True), encoding="utf-8")

    output_hashes = {}
    for path in sorted(out.iterdir()):
        if path.is_file() and path.name != "output_hashes_v50.json":
            output_hashes[path.name] = sha256(path)
    (out / "output_hashes_v50.json").write_text(json.dumps(output_hashes, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(decision_obj, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
