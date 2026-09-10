from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

from research_bot.vision_ict_v22 import (
    CHANNEL_NAMES,
    WEAK_LABEL_NAMES,
    VisionConfig,
    make_vision_model,
    require_torch,
)

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover
    torch = None
    nn = None
    DataLoader = TensorDataset = None


CHANNEL_MODES: dict[str, tuple[int, ...]] = {
    # Raw visual detector: no engineered Ichimoku/ICT structure is shown.
    "raw_candles": (0, 1, 2, 3),
    # Separates classical chart context from explicitly engineered ICT channels.
    "raw_plus_ichimoku": (0, 1, 2, 3, 4),
    # Full downstream representation: includes deterministic structure channels.
    "structure_augmented": tuple(range(len(CHANNEL_NAMES))),
}


@dataclass(frozen=True)
class VisionAblationConfig:
    weak_loss_weight: float = 0.70
    outcome_loss_weight: float = 0.30
    min_label_positives: int = 8
    min_label_negatives: int = 8
    max_pos_weight: float = 25.0
    min_validation_supported_macro_f1: float = 0.25
    min_test_supported_macro_f1: float = 0.20
    min_supported_labels: int = 8
    min_validation_outcome_auc: float = 0.52
    min_test_outcome_auc: float = 0.50
    min_augmented_auc_gain: float = 0.015


def select_channels(x: np.ndarray, mode: str) -> np.ndarray:
    if mode not in CHANNEL_MODES:
        raise ValueError(f"unknown channel mode: {mode}")
    if x.ndim != 4 or x.shape[1] != len(CHANNEL_NAMES):
        raise ValueError(f"expected [N,{len(CHANNEL_NAMES)},H,W], got {x.shape}")
    return x[:, CHANNEL_MODES[mode], :, :]


def weak_positive_weights(y: np.ndarray, max_weight: float = 25.0) -> np.ndarray:
    """Training-only positive class weights for imbalanced multi-label structure tasks."""
    if y.ndim != 2:
        raise ValueError("weak labels must be 2-D")
    pos = y.sum(axis=0).astype(float)
    neg = len(y) - pos
    # sqrt dampening avoids allowing one ultra-rare proxy to dominate the entire loss.
    ratio = np.sqrt((neg + 1.0) / (pos + 1.0))
    return np.clip(ratio, 1.0, max_weight).astype(np.float32)


def label_support(y: np.ndarray, min_pos: int = 8, min_neg: int = 8) -> np.ndarray:
    pos = y.sum(axis=0)
    neg = len(y) - pos
    return (pos >= min_pos) & (neg >= min_neg)


def supported_multilabel_metrics(y: np.ndarray, probability: np.ndarray, cfg: VisionAblationConfig | None = None) -> dict:
    c = cfg or VisionAblationConfig()
    if y.shape != probability.shape:
        raise ValueError("label/probability shapes differ")
    supported = label_support(y, c.min_label_positives, c.min_label_negatives)
    f1s: list[float] = []
    per_label: dict[str, dict] = {}
    pred = probability >= 0.5
    for j, name in enumerate(WEAK_LABEL_NAMES):
        score = float(f1_score(y[:, j], pred[:, j], zero_division=0))
        per_label[name] = {
            "f1": score,
            "positives": int(y[:, j].sum()),
            "negatives": int(len(y) - y[:, j].sum()),
            "supported": bool(supported[j]),
        }
        if supported[j]:
            f1s.append(score)
    return {
        "supported_labels": int(supported.sum()),
        "supported_macro_f1": float(np.mean(f1s)) if f1s else np.nan,
        "all_label_macro_f1": float(np.mean([v["f1"] for v in per_label.values()])),
        "per_label": per_label,
    }


def outcome_metrics(y: np.ndarray, probability: np.ndarray) -> dict:
    pred = (probability >= 0.5).astype(int)
    try:
        auc = float(roc_auc_score(y, probability))
    except Exception:
        auc = np.nan
    return {
        "auc": auc,
        "balanced_accuracy": float(balanced_accuracy_score(y.astype(int), pred)),
    }


def _weighted_weak_loss(logits, target, pos_weight):
    return nn.functional.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)


def fit_balanced_multitask_model(
    model,
    x_train: np.ndarray,
    weak_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    weak_val: np.ndarray,
    y_val: np.ndarray,
    vision_cfg: VisionConfig,
    ablation_cfg: VisionAblationConfig | None = None,
):
    """Train with development-only imbalance weights and validation-only early stopping."""
    require_torch()
    c = ablation_cfg or VisionAblationConfig()
    torch.manual_seed(vision_cfg.seed)
    np.random.seed(vision_cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    ds = TensorDataset(torch.tensor(x_train), torch.tensor(weak_train), torch.tensor(y_train))
    loader = DataLoader(
        ds,
        batch_size=vision_cfg.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(vision_cfg.seed),
    )
    weak_pw = torch.tensor(weak_positive_weights(weak_train, c.max_pos_weight), device=device)
    pos = float((y_train == 1).sum())
    neg = float((y_train == 0).sum())
    outcome_pw = torch.tensor(max(1.0, neg / max(1.0, pos)), device=device)
    outcome_loss = nn.BCEWithLogitsLoss(pos_weight=outcome_pw)
    opt = torch.optim.AdamW(model.parameters(), lr=vision_cfg.learning_rate, weight_decay=vision_cfg.weight_decay)
    xv = torch.tensor(x_val, device=device)
    wv = torch.tensor(weak_val, device=device)
    yv = torch.tensor(y_val, device=device)
    best = np.inf
    best_state = None
    stale = 0
    history = []
    for epoch in range(vision_cfg.epochs):
        model.train()
        total = 0.0
        n = 0
        for xb, wb, yb in loader:
            xb, wb, yb = xb.to(device), wb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            wl, ol, _ = model(xb)
            loss = c.weak_loss_weight * _weighted_weak_loss(wl, wb, weak_pw) + c.outcome_loss_weight * outcome_loss(ol, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach()) * len(xb)
            n += len(xb)
        model.eval()
        with torch.no_grad():
            wl, ol, _ = model(xv)
            val_loss = float((c.weak_loss_weight * _weighted_weak_loss(wl, wv, weak_pw) + c.outcome_loss_weight * outcome_loss(ol, yv)).cpu())
        history.append({
            "epoch": epoch + 1,
            "train_loss": total / max(1, n),
            "validation_loss": val_loss,
            "weak_loss_weight": c.weak_loss_weight,
            "outcome_loss_weight": c.outcome_loss_weight,
        })
        if val_loss < best - 1e-5:
            best = val_loss
            stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= vision_cfg.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model.cpu(), pd.DataFrame(history), weak_pw.cpu().numpy()


def predict_multitask(model, x: np.ndarray, batch_size: int = 512) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    require_torch()
    model.eval()
    weak, outcome, emb = [], [], []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            wl, ol, z = model(torch.tensor(x[i:i + batch_size]))
            weak.append(torch.sigmoid(wl).cpu().numpy())
            outcome.append(torch.sigmoid(ol).cpu().numpy())
            emb.append(z.cpu().numpy())
    return (
        np.concatenate(weak) if weak else np.empty((0, len(WEAK_LABEL_NAMES))),
        np.concatenate(outcome) if outcome else np.empty(0),
        np.concatenate(emb) if emb else np.empty((0, 0)),
    )


def model_record(
    *,
    architecture: str,
    channel_mode: str,
    weak_val: np.ndarray,
    weak_prob_val: np.ndarray,
    y_val: np.ndarray,
    outcome_prob_val: np.ndarray,
    weak_test: np.ndarray,
    weak_prob_test: np.ndarray,
    y_test: np.ndarray,
    outcome_prob_test: np.ndarray,
    cfg: VisionAblationConfig | None = None,
) -> tuple[dict, dict, dict]:
    c = cfg or VisionAblationConfig()
    vm = supported_multilabel_metrics(weak_val, weak_prob_val, c)
    tm = supported_multilabel_metrics(weak_test, weak_prob_test, c)
    vo = outcome_metrics(y_val, outcome_prob_val)
    to = outcome_metrics(y_test, outcome_prob_test)
    objective = (
        (vm["supported_macro_f1"] if np.isfinite(vm["supported_macro_f1"]) else 0.0)
        + 0.20 * (vo["auc"] if np.isfinite(vo["auc"]) else 0.5)
    )
    rec = {
        "architecture": architecture,
        "channel_mode": channel_mode,
        "validation_objective": float(objective),
        "validation_supported_labels": vm["supported_labels"],
        "validation_supported_macro_f1": vm["supported_macro_f1"],
        "validation_all_label_macro_f1": vm["all_label_macro_f1"],
        "validation_outcome_auc": vo["auc"],
        "validation_outcome_balanced_accuracy": vo["balanced_accuracy"],
        "test_supported_labels": tm["supported_labels"],
        "test_supported_macro_f1": tm["supported_macro_f1"],
        "test_all_label_macro_f1": tm["all_label_macro_f1"],
        "test_outcome_auc": to["auc"],
        "test_outcome_balanced_accuracy": to["balanced_accuracy"],
    }
    return rec, vm, tm


def decide_ablation(board: pd.DataFrame, cfg: VisionAblationConfig | None = None) -> dict:
    """Fail-closed decision. Validation selects; test only confirms the frozen choice."""
    c = cfg or VisionAblationConfig()
    if board.empty:
        return {"decision": "NO_VISION_PROMOTION", "reason": "empty leaderboard", "live_execution_authorized": False}
    ok = board[board["status"] == "ok"].copy() if "status" in board else board.copy()
    raw = ok[ok["channel_mode"] == "raw_candles"].copy()
    if raw.empty:
        return {"decision": "NO_VISION_PROMOTION", "reason": "no raw-candlestick detector completed", "live_execution_authorized": False}

    # Freeze the raw detector using validation only.
    raw_champ = raw.sort_values("validation_objective", ascending=False).iloc[0]
    raw_pass = bool(
        int(raw_champ["validation_supported_labels"]) >= c.min_supported_labels
        and float(raw_champ["validation_supported_macro_f1"]) >= c.min_validation_supported_macro_f1
        and int(raw_champ["test_supported_labels"]) >= c.min_supported_labels
        and float(raw_champ["test_supported_macro_f1"]) >= c.min_test_supported_macro_f1
    )

    augmented = ok[ok["channel_mode"] == "structure_augmented"].copy()
    augmented_result = None
    if not augmented.empty:
        aug_champ = augmented.sort_values("validation_objective", ascending=False).iloc[0]
        val_gain = float(aug_champ["validation_outcome_auc"] - raw_champ["validation_outcome_auc"])
        augmented_pass = bool(
            np.isfinite(float(aug_champ["validation_outcome_auc"]))
            and float(aug_champ["validation_outcome_auc"]) >= c.min_validation_outcome_auc
            and val_gain >= c.min_augmented_auc_gain
            and np.isfinite(float(aug_champ["test_outcome_auc"]))
            and float(aug_champ["test_outcome_auc"]) >= c.min_test_outcome_auc
        )
        augmented_result = {
            "champion": f"{aug_champ['architecture']}:{aug_champ['channel_mode']}",
            "validation_auc_gain_vs_raw": val_gain,
            "passed": augmented_pass,
        }
    else:
        augmented_pass = False

    if raw_pass and augmented_pass:
        decision = "RAW_ICT_DETECTOR_AND_AUGMENTED_REPRESENTATION_CANDIDATE"
    elif raw_pass:
        decision = "RAW_ICT_DETECTOR_CANDIDATE"
    elif augmented_pass:
        decision = "AUGMENTED_REPRESENTATION_CANDIDATE_ONLY"
    else:
        decision = "NO_VISION_PROMOTION"
    return {
        "decision": decision,
        "raw_champion": f"{raw_champ['architecture']}:{raw_champ['channel_mode']}",
        "raw_detector_pass": raw_pass,
        "augmented": augmented_result,
        "selection_basis": "validation only; test used only to confirm pre-registered acceptance gates",
        "paper_replacement_authorized": False,
        "live_execution_authorized": False,
    }


def save_ablation_protocol(path: Path, cfg: VisionAblationConfig) -> None:
    path.write_text(json.dumps({
        "version": "v0.22b",
        "channel_modes": {k: [CHANNEL_NAMES[i] for i in v] for k, v in CHANNEL_MODES.items()},
        "gates": cfg.__dict__,
        "scientific_distinction": {
            "raw_candles": "tests whether CV can infer formalized ICT structure from candle/volume geometry without engineered structure channels",
            "raw_plus_ichimoku": "tests added value of causal Ichimoku visual context",
            "structure_augmented": "tests engineered ICT channels as a downstream representation; weak-label F1 here is not evidence of independent visual discovery",
        },
    }, indent=2), encoding="utf-8")
