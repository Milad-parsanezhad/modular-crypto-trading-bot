from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Callable
import json

import numpy as np
import pandas as pd

from research_bot.ml_framework_v23r import (
    BANNED_FEATURE_EXACT,
    BANNED_SUBSTRINGS,
    assert_temporal_separation,
    assert_unique_columns,
    dataframe_sha256,
    select_feature_columns,
)


@dataclass(frozen=True)
class AuditFinding:
    severity: str
    code: str
    message: str
    passed: bool


@dataclass(frozen=True)
class AuditThresholds:
    max_duplicate_observation_fraction: float = 0.0
    max_feature_missing_fraction: float = 0.35
    max_constant_feature_fraction: float = 0.25
    min_binary_class_fraction: float = 0.05
    max_binary_class_fraction: float = 0.95


def _finding(ok: bool, code: str, success: str, failure: str, *, severity: str = "ERROR") -> AuditFinding:
    return AuditFinding("INFO" if ok else severity, code, success if ok else failure, ok)


def _name_banned(name: str) -> bool:
    low = name.lower().strip()
    return low in BANNED_FEATURE_EXACT or any(token in low for token in BANNED_SUBSTRINGS)


def audit_dataset(
    df: pd.DataFrame,
    *,
    timestamp_col: str = "signal_time",
    label_col: str = "label_positive_net",
    split_col: str = "segment",
    thresholds: AuditThresholds | None = None,
    include_identity_features: bool = False,
) -> tuple[pd.DataFrame, dict]:
    thresholds = thresholds or AuditThresholds()
    findings: list[AuditFinding] = []
    try:
        assert_unique_columns(df)
        findings.append(_finding(True, "UNIQUE_COLUMNS", "all columns are unique", "duplicate columns detected"))
    except Exception as exc:
        findings.append(_finding(False, "UNIQUE_COLUMNS", "", str(exc)))

    if timestamp_col not in df:
        findings.append(_finding(False, "TIMESTAMP_PRESENT", "", f"missing timestamp column {timestamp_col}"))
        report = pd.DataFrame([asdict(x) for x in findings])
        return report, {"decision": "AUDIT_FAIL", "rows": int(len(df)), "fatal_codes": ["TIMESTAMP_PRESENT"]}

    ts = pd.to_datetime(df[timestamp_col], utc=True, errors="coerce")
    findings.append(_finding(not ts.isna().any(), "TIMESTAMP_PARSE", "timestamps parse cleanly", "unparseable timestamps exist"))

    # In a multi-asset panel, repeating the same timestamp across symbols is expected.
    # What is forbidden is a duplicate observation key: same symbol and timestamp
    # (or same timestamp when symbol is unavailable).
    key_cols = [timestamp_col]
    if "symbol" in df.columns:
        key_cols = ["symbol", timestamp_col]
    dup_obs_frac = float(df.duplicated(key_cols).mean()) if len(df) else 0.0
    findings.append(_finding(
        dup_obs_frac <= thresholds.max_duplicate_observation_fraction,
        "DUPLICATE_OBSERVATION_KEYS",
        f"duplicate observation-key fraction={dup_obs_frac:.6f} using {key_cols}",
        f"duplicate observation-key fraction={dup_obs_frac:.6f} exceeds {thresholds.max_duplicate_observation_fraction:.6f} using {key_cols}",
    ))
    panel_timestamp_repeat_fraction = float(ts.duplicated().mean()) if len(ts) else 0.0
    findings.append(_finding(
        True,
        "PANEL_TIMESTAMP_MULTIPLICITY",
        f"timestamp repeat fraction={panel_timestamp_repeat_fraction:.6f}; repeats across symbols are expected in panel data",
        "",
    ))

    if split_col in df:
        expected = ("development", "validation", "test")
        actual = set(df[split_col].dropna().astype(str).unique())
        findings.append(_finding(set(expected).issubset(actual), "SPLIT_LABELS", f"split labels present: {sorted(actual)}", f"expected {sorted(expected)}, observed {sorted(actual)}"))
        try:
            split_map = {k: df[df[split_col] == k].sort_values(timestamp_col).copy() for k in expected}
            assert_temporal_separation(split_map, timestamp_col=timestamp_col)
            findings.append(_finding(True, "TEMPORAL_SEPARATION", "development < validation < test with no shared timestamp", ""))
        except Exception as exc:
            findings.append(_finding(False, "TEMPORAL_SEPARATION", "", str(exc)))
    else:
        findings.append(_finding(False, "SPLIT_LABELS", "", f"missing split column {split_col}"))

    try:
        numeric, categorical = select_feature_columns(
            df,
            include_context=True,
            include_identity=include_identity_features,
        )
        findings.append(_finding(True, "FEATURE_WHITELIST", f"selected {len(numeric)} numeric and {len(categorical)} categorical features", ""))
    except Exception as exc:
        numeric, categorical = [], []
        findings.append(_finding(False, "FEATURE_WHITELIST", "", str(exc)))

    bad_names = [c for c in numeric + categorical if _name_banned(c)]
    findings.append(_finding(not bad_names, "OUTCOME_NAME_GUARD", "no outcome-like feature names", f"banned feature names: {bad_names}"))

    missing = {c: float(df[c].isna().mean()) for c in numeric}
    worst_missing = max(missing.values(), default=0.0)
    findings.append(_finding(
        worst_missing <= thresholds.max_feature_missing_fraction,
        "FEATURE_MISSINGNESS",
        f"worst numeric feature missingness={worst_missing:.4f}",
        f"worst numeric feature missingness={worst_missing:.4f} > {thresholds.max_feature_missing_fraction:.4f}",
    ))

    inf_features = []
    for c in numeric:
        values = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        if np.isinf(values).any():
            inf_features.append(c)
    findings.append(_finding(not inf_features, "FINITE_FEATURES", "no +/-inf in numeric features", f"infinite values in: {inf_features}"))

    constant = [c for c in numeric if df[c].nunique(dropna=True) <= 1]
    const_frac = float(len(constant) / max(len(numeric), 1))
    findings.append(_finding(
        const_frac <= thresholds.max_constant_feature_fraction,
        "CONSTANT_FEATURES",
        f"constant feature fraction={const_frac:.4f}",
        f"constant feature fraction={const_frac:.4f} > {thresholds.max_constant_feature_fraction:.4f}; {constant}",
        severity="WARN",
    ))

    if label_col in df:
        y = pd.to_numeric(df[label_col], errors="coerce")
        clean = y.dropna()
        classes = set(clean.unique())
        findings.append(_finding(classes.issubset({0, 1}) and len(classes) == 2, "BINARY_LABEL", f"binary label classes={sorted(classes)}", f"invalid/degenerate label classes={sorted(classes)}"))
        positive = float(clean.mean()) if len(clean) else np.nan
        balance_ok = bool(np.isfinite(positive) and thresholds.min_binary_class_fraction <= positive <= thresholds.max_binary_class_fraction)
        findings.append(_finding(balance_ok, "LABEL_BALANCE", f"positive class fraction={positive:.4f}", f"positive class fraction={positive:.4f} outside [{thresholds.min_binary_class_fraction:.2f}, {thresholds.max_binary_class_fraction:.2f}]", severity="WARN"))
    else:
        positive = np.nan
        findings.append(_finding(False, "BINARY_LABEL", "", f"missing label column {label_col}"))

    fatal = [f for f in findings if not f.passed and f.severity == "ERROR"]
    report = pd.DataFrame([asdict(x) for x in findings])
    summary = {
        "decision": "AUDIT_PASS" if not fatal else "AUDIT_FAIL",
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "numeric_feature_count": int(len(numeric)),
        "categorical_feature_count": int(len(categorical)),
        "positive_fraction": positive,
        "duplicate_observation_fraction": dup_obs_frac,
        "panel_timestamp_repeat_fraction": panel_timestamp_repeat_fraction,
        "worst_missing_fraction": worst_missing,
        "constant_features": constant,
        "dataframe_sha256": dataframe_sha256(df),
        "fatal_codes": [f.code for f in fatal],
        "warnings": [f.code for f in findings if not f.passed and f.severity == "WARN"],
    }
    return report, summary


def audit_prefix_invariance(
    frame: pd.DataFrame,
    feature_builder: Callable[[pd.DataFrame], pd.DataFrame],
    *,
    cut_fraction: float = 0.70,
    feature_prefix: str = "f_",
    atol: float = 1e-10,
) -> dict:
    """Adding future rows must not alter historical features."""
    if not 0.2 <= cut_fraction <= 0.9:
        raise ValueError("cut_fraction must be between 0.2 and 0.9")
    n = len(frame)
    cut = int(n * cut_fraction)
    if cut < 50:
        raise ValueError("not enough rows for prefix-invariance audit")
    full = feature_builder(frame.copy()).reset_index(drop=True)
    prefix = feature_builder(frame.iloc[:cut].copy()).reset_index(drop=True)
    cols = [c for c in prefix.columns if c.startswith(feature_prefix) and c in full.columns]
    if not cols:
        raise ValueError("feature builder returned no auditable f_* features")
    bad = []
    for c in cols:
        a = pd.to_numeric(full.loc[: cut - 1, c], errors="coerce").to_numpy(dtype=float)
        b = pd.to_numeric(prefix[c], errors="coerce").to_numpy(dtype=float)
        if len(a) != len(b) or not np.allclose(a, b, equal_nan=True, atol=atol, rtol=1e-9):
            bad.append(c)
    return {"passed": not bad, "cut": cut, "features_checked": len(cols), "violating_features": bad}


def audit_future_mutation_invariance(
    frame: pd.DataFrame,
    feature_builder: Callable[[pd.DataFrame], pd.DataFrame],
    *,
    cut_fraction: float = 0.70,
    numeric_market_columns: tuple[str, ...] = ("open", "high", "low", "close", "volume"),
) -> dict:
    """Mutate future candles adversarially; historical f_* values must stay fixed."""
    n = len(frame)
    cut = int(n * cut_fraction)
    if cut < 50 or cut >= n - 2:
        raise ValueError("not enough rows for future-mutation audit")
    original = feature_builder(frame.copy()).reset_index(drop=True)
    mutated = frame.copy()
    future_idx = mutated.index[cut:]
    for c in numeric_market_columns:
        if c in mutated:
            factor = 1.37 if c != "volume" else 3.0
            mutated.loc[future_idx, c] = pd.to_numeric(mutated.loc[future_idx, c], errors="coerce") * factor
    changed = feature_builder(mutated).reset_index(drop=True)
    cols = [c for c in original.columns if c.startswith("f_") and c in changed.columns]
    bad = []
    for c in cols:
        a = pd.to_numeric(original.loc[: cut - 1, c], errors="coerce").to_numpy(dtype=float)
        b = pd.to_numeric(changed.loc[: cut - 1, c], errors="coerce").to_numpy(dtype=float)
        if not np.allclose(a, b, equal_nan=True, atol=1e-10, rtol=1e-9):
            bad.append(c)
    return {"passed": not bad, "cut": cut, "features_checked": len(cols), "violating_features": bad}


def audit_prediction_selection(
    predictions: pd.DataFrame,
    *,
    split_col: str = "segment",
    threshold_source_col: str = "threshold_source",
    model_selection_source_col: str = "model_selection_source",
) -> tuple[pd.DataFrame, dict]:
    """Verify that test rows did not choose model family or threshold."""
    findings: list[AuditFinding] = []
    has_split = split_col in predictions
    findings.append(_finding(has_split, "PREDICTION_SPLIT", "prediction split column present", f"missing {split_col}"))
    if has_split and threshold_source_col in predictions:
        bad = predictions[predictions[split_col] == "test"][threshold_source_col].astype(str).str.lower().eq("test").any()
        findings.append(_finding(not bad, "TEST_THRESHOLD_FREEZE", "test never selected threshold", "test selected its own threshold"))
    else:
        findings.append(_finding(False, "TEST_THRESHOLD_FREEZE", "", f"missing {threshold_source_col} or split", severity="WARN"))
    if has_split and model_selection_source_col in predictions:
        bad = predictions[predictions[split_col] == "test"][model_selection_source_col].astype(str).str.lower().eq("test").any()
        findings.append(_finding(not bad, "TEST_MODEL_FREEZE", "test never selected model", "test selected model family"))
    else:
        findings.append(_finding(False, "TEST_MODEL_FREEZE", "", f"missing {model_selection_source_col} or split", severity="WARN"))
    fatal = [x for x in findings if not x.passed and x.severity == "ERROR"]
    return pd.DataFrame([asdict(x) for x in findings]), {"decision": "AUDIT_PASS" if not fatal else "AUDIT_FAIL", "fatal_codes": [x.code for x in fatal]}


def write_audit_bundle(
    output_dir: str | Path,
    report: pd.DataFrame,
    summary: dict,
    *,
    name: str = "dataset_audit",
) -> tuple[Path, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"{name}.csv"
    json_path = out / f"{name}.json"
    report.to_csv(csv_path, index=False)
    payload = dict(summary)
    payload["report_sha256"] = sha256(csv_path.read_bytes()).hexdigest()
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return csv_path, json_path
