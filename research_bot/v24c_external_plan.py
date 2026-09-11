from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from research_bot.ml_framework_v23r import (
    SplitContract,
    chronological_purged_split,
    make_preprocessor,
    probability_score,
    select_feature_columns,
    supervised_model_registry,
)
from research_bot.strategy_meta_v24 import V24Contract, build_strategy_event_panel, economic_metrics, target_specs


INTERNAL_SYMBOLS_V24B: tuple[str, ...] = (
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT", "ADA/USDT",
    "LINK/USDT", "LTC/USDT", "BCH/USDT", "TRX/USDT", "AVAX/USDT", "DOT/USDT",
)

# Pre-registered OOD/cross-sectional universe. These symbols were not part of the
# v0.24b 12-symbol training panel. Availability is an admissible filter; outcome
# quality is never used to add/drop symbols after the external read.
EXTERNAL_BYBIT_SYMBOLS_V24C: tuple[str, ...] = (
    "ATOM/USDT", "ETC/USDT", "FIL/USDT", "NEAR/USDT", "UNI/USDT", "APT/USDT",
    "ARB/USDT", "OP/USDT", "SUI/USDT", "INJ/USDT", "AAVE/USDT", "TON/USDT",
)

# The v0.24b artifact reported its final closed CoinEx 4h bar at 04:00 UTC on
# 2026-09-11. Reproduction must not move the internal development/validation/test
# clock forward merely because the external experiment is run later.
INTERNAL_LAST_CLOSED_BAR_UTC = pd.Timestamp("2026-09-11T04:00:00Z")
INTERNAL_FETCH_END_UTC = pd.Timestamp("2026-09-11T08:00:00Z")


@dataclass(frozen=True)
class FrozenFamilyCandidate:
    strategy: str
    model_name: str
    seed: int
    threshold: float
    expected_development_events: int
    expected_validation_events: int
    expected_shadow_events: int
    expected_validation_selected: int
    feature_schema_sha256: str


FROZEN_CANDIDATES_V24C: tuple[FrozenFamilyCandidate, ...] = (
    FrozenFamilyCandidate(
        strategy="H4_S6_BREAKOUT",
        model_name="sgd_logistic",
        seed=314,
        threshold=0.2992513650430247,
        expected_development_events=695,
        expected_validation_events=236,
        expected_shadow_events=217,
        expected_validation_selected=118,
        feature_schema_sha256="5d5e345cd3eec021bc4c2c28e533f0d22c14257d0659165389d097c6f8fd39d1",
    ),
    FrozenFamilyCandidate(
        strategy="H4_D1_OB_BOS_RISK",
        model_name="logistic",
        seed=314,
        threshold=0.5481968244713689,
        expected_development_events=566,
        expected_validation_events=185,
        expected_shadow_events=178,
        expected_validation_selected=47,
        feature_schema_sha256="5d5e345cd3eec021bc4c2c28e533f0d22c14257d0659165389d097c6f8fd39d1",
    ),
)


@dataclass(frozen=True)
class FrozenModelBundle:
    candidate: FrozenFamilyCandidate
    model: Pipeline
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]

    @property
    def features(self) -> tuple[str, ...]:
        return self.numeric_features + self.categorical_features


def _schema_sha(numeric: list[str] | tuple[str, ...], categorical: list[str] | tuple[str, ...]) -> str:
    raw = json.dumps({"numeric": list(numeric), "categorical": list(categorical)}, sort_keys=True).encode("utf-8")
    return sha256(raw).hexdigest()


def _peer_for(symbol: str, frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame | None:
    preferred = "ETH/USDT" if symbol == "BTC/USDT" else "BTC/USDT"
    if preferred in frames and preferred != symbol:
        return frames[preferred]
    return next((f for s, f in frames.items() if s != symbol), None)


def build_strategy_dataset_from_frames(frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Rebuild the exact v0.24 event-panel semantics from already-fetched frames.

    The split is constructed over the full seven-strategy panel before selecting
    the two frozen v0.24c hypotheses. This is required because the v0.24b split
    boundaries were determined from the union of all strategy-event timestamps.
    """
    if len(frames) < 5:
        raise ValueError("at least five symbol frames are required")
    contract = V24Contract()
    parts: list[pd.DataFrame] = []
    for spec in target_specs():
        for symbol, raw in frames.items():
            frame = raw.copy().sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
            needs_peer = "CORRELATION" in spec.name or spec.family == "correlation_divergence"
            peer = _peer_for(symbol, frames) if needs_peer else None
            panel = build_strategy_event_panel(spec, frame, symbol, peer=peer, contract=contract)
            if not panel.empty:
                parts.append(panel)
    if not parts:
        raise RuntimeError("no strategy events generated")
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
    tagged: list[pd.DataFrame] = []
    for segment, part in split.items():
        z = part.copy()
        z["segment"] = segment
        tagged.append(z)
    return pd.concat(tagged, ignore_index=True).sort_values(["signal_time", "strategy", "symbol"]).reset_index(drop=True)


def reproduce_frozen_models(dataset: pd.DataFrame) -> dict[str, FrozenModelBundle]:
    """Deterministically reconstruct the two v0.24b frozen family models.

    No model/seed/threshold search is performed. The function aborts if the
    historical sample counts, feature schema, or frozen-threshold validation
    selection count differ from the recorded v0.24b evidence bundle.
    """
    bundles: dict[str, FrozenModelBundle] = {}
    for cand in FROZEN_CANDIDATES_V24C:
        fam = dataset[dataset["strategy"] == cand.strategy].copy()
        dev = fam[fam["segment"] == "development"].reset_index(drop=True)
        val = fam[fam["segment"] == "validation"].reset_index(drop=True)
        shadow = fam[fam["segment"] == "test"].reset_index(drop=True)
        counts = (len(dev), len(val), len(shadow))
        expected = (cand.expected_development_events, cand.expected_validation_events, cand.expected_shadow_events)
        if counts != expected:
            raise RuntimeError(f"FROZEN_REPRODUCTION_COUNT_MISMATCH {cand.strategy}: got={counts} expected={expected}")

        numeric, categorical = select_feature_columns(dev, include_context=True, include_identity=False)
        numeric = [c for c in numeric if not c.startswith("f_strategy_")]
        schema = _schema_sha(numeric, categorical)
        if schema != cand.feature_schema_sha256:
            raise RuntimeError(
                f"FROZEN_FEATURE_SCHEMA_MISMATCH {cand.strategy}: got={schema} expected={cand.feature_schema_sha256}"
            )
        registry = supervised_model_registry(cand.seed)
        if cand.model_name not in registry:
            raise RuntimeError(f"frozen model family unavailable: {cand.model_name}")
        pipe = Pipeline([
            ("prep", make_preprocessor(numeric, categorical)),
            ("model", clone(registry[cand.model_name])),
        ])
        xcols = numeric + categorical
        pipe.fit(dev[xcols], dev["label_meta_execute"].astype(int))
        val_score = probability_score(pipe, val[xcols])
        selected = val_score >= cand.threshold
        if int(selected.sum()) != cand.expected_validation_selected:
            raise RuntimeError(
                f"FROZEN_THRESHOLD_REPRODUCTION_MISMATCH {cand.strategy}: "
                f"got={int(selected.sum())} expected={cand.expected_validation_selected}"
            )
        bundles[cand.strategy] = FrozenModelBundle(
            candidate=cand,
            model=pipe,
            numeric_features=tuple(numeric),
            categorical_features=tuple(categorical),
        )
    return bundles


def build_external_candidate_events(
    frames: Mapping[str, pd.DataFrame],
    *,
    candidates: tuple[FrozenFamilyCandidate, ...] = FROZEN_CANDIDATES_V24C,
) -> pd.DataFrame:
    by_name = {s.name: s for s in target_specs()}
    parts: list[pd.DataFrame] = []
    for cand in candidates:
        spec = by_name[cand.strategy]
        for symbol, frame in frames.items():
            panel = build_strategy_event_panel(spec, frame, symbol, contract=V24Contract())
            if not panel.empty:
                panel = panel.copy()
                panel["external_source"] = "bybit_public_spot"
                panel["external_symbol_unseen_in_v24b"] = symbol not in INTERNAL_SYMBOLS_V24B
                parts.append(panel)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out["signal_time"] = pd.to_datetime(out["signal_time"], utc=True)
    out["entry_time"] = pd.to_datetime(out["entry_time"], utc=True)
    out["exit_time"] = pd.to_datetime(out["exit_time"], utc=True)
    return out.sort_values(["signal_time", "strategy", "symbol"], kind="mergesort").reset_index(drop=True)


def score_external_events(
    events: pd.DataFrame,
    bundles: Mapping[str, FrozenModelBundle],
    *,
    risk_per_trade: float = 0.0025,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply frozen models/thresholds to external events without refit or retune."""
    if events.empty:
        return events.copy(), pd.DataFrame()
    pred_parts: list[pd.DataFrame] = []
    rows: list[dict] = []
    for strategy, bundle in bundles.items():
        x = events[events["strategy"] == strategy].copy().reset_index(drop=True)
        if x.empty:
            rows.append({"strategy": strategy, "status": "DATA_INSUFFICIENT", "external_events": 0})
            continue
        missing = [c for c in bundle.features if c not in x.columns]
        if missing:
            raise RuntimeError(f"external feature schema mismatch {strategy}: {missing}")
        score = probability_score(bundle.model, x[list(bundle.features)])
        selected = score >= bundle.candidate.threshold
        base = economic_metrics(x, risk_per_trade=risk_per_trade)
        filtered = economic_metrics(x, selected, risk_per_trade=risk_per_trade)
        rows.append({
            "strategy": strategy,
            "status": "EXTERNAL_EVALUATED",
            "external_events": int(len(x)),
            "external_symbols": int(x["symbol"].nunique()),
            "selected": int(selected.sum()),
            "frozen_model": bundle.candidate.model_name,
            "frozen_seed": bundle.candidate.seed,
            "frozen_threshold": bundle.candidate.threshold,
            **{f"base_{k}": v for k, v in base.items()},
            **{f"filtered_{k}": v for k, v in filtered.items()},
            "model_refit_on_external": False,
            "threshold_tuned_on_external": False,
        })
        pred = x[[
            "strategy", "symbol", "signal_time", "entry_time", "exit_time", "side",
            "entry", "exit", "stop", "target", "r_multiple", "exit_reason",
        ]].copy()
        pred["external_score"] = score
        pred["external_selected"] = selected
        pred["frozen_threshold"] = bundle.candidate.threshold
        pred_parts.append(pred)
    predictions = pd.concat(pred_parts, ignore_index=True) if pred_parts else pd.DataFrame()
    return predictions, pd.DataFrame(rows)


def frozen_manifest() -> dict:
    return {
        "version": "v0.24c",
        "source_experiment": "v0.24b",
        "internal_last_closed_bar_utc": INTERNAL_LAST_CLOSED_BAR_UTC.isoformat(),
        "internal_fetch_end_utc": INTERNAL_FETCH_END_UTC.isoformat(),
        "internal_symbols": list(INTERNAL_SYMBOLS_V24B),
        "external_bybit_symbols": list(EXTERNAL_BYBIT_SYMBOLS_V24C),
        "external_universe_rule": "fixed pre-registered symbols disjoint from v0.24b internal 12-symbol panel; availability-only filtering",
        "candidates": [asdict(x) for x in FROZEN_CANDIDATES_V24C],
        "external_model_refit_allowed": False,
        "external_threshold_tuning_allowed": False,
        "live_execution_authorized": False,
    }
