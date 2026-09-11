from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from research_bot.frozen_snapshot_v24d import (
    FrozenSnapshotContract,
    assert_runtime_compatible,
    load_exact_frozen_bundles,
    replay_archived_validation,
    verify_snapshot_files,
)
from research_bot.risk_ranker_v25 import (
    V25RankingContract,
    _realized_priority_replay,
    attach_exact_frozen_gate,
    fit_ranking_tournament,
    score_ranker,
)


def _dump(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str, allow_nan=True), encoding="utf-8")


def _sha_json(obj: dict) -> str:
    return sha256(json.dumps(obj, sort_keys=True, default=str, allow_nan=True).encode()).hexdigest()


def _spent_external_diagnostic(root: Path, snapshot: dict, contract: V25RankingContract) -> pd.DataFrame:
    rows = []
    for venue in ("okx", "kucoin"):
        path = root / venue / "external_scored_events_full.csv"
        if not path.is_file():
            rows.append({"venue": venue, "status": "SPENT_DATA_NOT_AVAILABLE"})
            continue
        frame = pd.read_csv(path)
        frame = frame[frame["strategy"].isin(("H4_S6_BREAKOUT", "H4_D1_OB_BOS_RISK"))].copy().reset_index(drop=True)
        if frame.empty:
            rows.append({"venue": venue, "status": "SPENT_DATA_EMPTY"})
            continue
        frame["frozen_score"] = pd.to_numeric(frame["external_score"], errors="coerce")
        frame["frozen_threshold"] = pd.to_numeric(frame["frozen_threshold"], errors="coerce")
        frame["frozen_selected"] = frame["external_selected"].astype(bool)
        frame["frozen_score_margin"] = frame["frozen_score"] - frame["frozen_threshold"]
        learned_priority = score_ranker(frame, snapshot)
        base_priority = frame["frozen_score"].to_numpy(dtype=float)
        baseline = _realized_priority_replay(frame, base_priority, contract)
        challenger = _realized_priority_replay(frame, learned_priority, contract)
        rows.append({
            "venue": venue,
            "status": "SPENT_DIAGNOSTIC_NOT_PROMOTION",
            "events": int(len(frame)),
            "frozen_selected": int(frame["frozen_selected"].sum()),
            "baseline_total_return": baseline["total_return"],
            "baseline_profit_factor": baseline["profit_factor"],
            "baseline_mean_r": baseline["mean_r"],
            "baseline_max_realized_drawdown": baseline["max_realized_drawdown"],
            "challenger_total_return": challenger["total_return"],
            "challenger_profit_factor": challenger["profit_factor"],
            "challenger_mean_r": challenger["mean_r"],
            "challenger_max_realized_drawdown": challenger["max_realized_drawdown"],
            "uplift_total_return": challenger["total_return"] - baseline["total_return"],
            "spent_sample_can_promote": False,
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v24b-dir", required=True)
    parser.add_argument("--v24d-dir", default=None)
    parser.add_argument("--output-dir", default="artifacts/v25-risk-aware-ranking")
    args = parser.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    v24b = Path(args.v24b_dir)
    contract = V25RankingContract()
    frozen_contract = FrozenSnapshotContract()

    observed_hashes = verify_snapshot_files(v24b, frozen_contract)
    runtime = assert_runtime_compatible(frozen_contract)
    bundles = load_exact_frozen_bundles(v24b, frozen_contract)
    replay = replay_archived_validation(v24b, bundles, frozen_contract)
    replay.to_csv(out / "exact_frozen_replay.csv", index=False)

    dataset = pd.read_csv(v24b / frozen_contract.dataset_file)
    dataset = attach_exact_frozen_gate(dataset, bundles)
    dataset.to_csv(out / "archived_dataset_with_frozen_gate.csv", index=False)
    dev = dataset[dataset["segment"] == "development"].copy().reset_index(drop=True)
    val = dataset[dataset["segment"] == "validation"].copy().reset_index(drop=True)
    if dev.empty or val.empty:
        raise RuntimeError("V25_DEVELOPMENT_OR_VALIDATION_EMPTY")

    leaderboard, snapshot = fit_ranking_tournament(dev, val, contract)
    leaderboard.to_csv(out / "ranker_validation_leaderboard.csv", index=False)
    joblib.dump(snapshot, out / "ranker_snapshot.joblib")

    manifest = {
        "version": "v0.25",
        "scientific_role": "PORTFOLIO_ADMISSION_RANKER_DEVELOPMENT",
        "contract": contract.to_dict(),
        "frozen_v24b_contract": frozen_contract.to_dict(),
        "frozen_file_sha256": observed_hashes,
        "runtime": runtime,
        "exact_replay": replay.to_dict(orient="records"),
        "development_rows": int(len(dev)),
        "validation_rows": int(len(val)),
        "champion": snapshot["champion"],
        "rank_features": snapshot["features"],
        "future_start_utc": contract.future_start_utc,
        "v24d_external_status": "SPENT_POST_HOC_DIAGNOSTIC_ONLY",
        "model_refit_on_future": False,
        "same_test_rescue_tuning_allowed": False,
        "paper_replacement_authorized": False,
        "forward_paper_authorized": False,
        "live_execution_authorized": False,
    }
    manifest["manifest_sha256"] = _sha_json(manifest)
    _dump(out / "ranker_manifest.json", manifest)

    if args.v24d_dir:
        diagnostic = _spent_external_diagnostic(Path(args.v24d_dir), snapshot, contract)
        diagnostic.to_csv(out / "spent_v24d_allocator_diagnostic.csv", index=False)
    else:
        diagnostic = pd.DataFrame()

    winner = leaderboard.loc[leaderboard["ranker"] == snapshot["champion"]].iloc[0].to_dict()
    decision = {
        "version": "v0.25",
        "state": "RANKER_FROZEN_FOR_FUTURE_TIME_RESEARCH",
        "scientific_label": "CHALLENGER_NOT_PROMOTED",
        "champion": snapshot["champion"],
        "validation_winner": winner,
        "frozen_score_baseline": leaderboard.loc[leaderboard["ranker"] == "frozen_score"].iloc[0].to_dict(),
        "v24d_spent_diagnostic_rows": int(len(diagnostic)),
        "future_start_utc": contract.future_start_utc,
        "promotion_evidence_consumed": False,
        "next_gate": "PROSPECTIVE_FUTURE_TIME_COLLECTION_THEN_MTM_CVAR_PORTFOLIO_REPLAY",
        "cpcv_pbo_dsr_required_before_paper_replacement": True,
        "forward_paper_authorized": False,
        "paper_replacement_authorized": False,
        "live_execution_authorized": False,
    }
    decision["decision_sha256"] = _sha_json(decision)
    _dump(out / "decision.json", decision)
    print(json.dumps(decision, indent=2, default=str, allow_nan=True))


if __name__ == "__main__":
    main()
