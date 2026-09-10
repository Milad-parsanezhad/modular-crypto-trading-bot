import numpy as np
import pandas as pd

from research_bot.vision_ablation_v22b import (
    CHANNEL_MODES,
    VisionAblationConfig,
    decide_ablation,
    label_support,
    select_channels,
    supported_multilabel_metrics,
    weak_positive_weights,
)
from research_bot.vision_ict_v22 import CHANNEL_NAMES, WEAK_LABEL_NAMES


def test_raw_channel_mode_excludes_engineered_ict_and_ichimoku_channels():
    x = np.zeros((3, len(CHANNEL_NAMES), 16, 24), dtype=np.float32)
    raw = select_channels(x, "raw_candles")
    assert raw.shape == (3, 4, 16, 24)
    assert [CHANNEL_NAMES[i] for i in CHANNEL_MODES["raw_candles"]] == ["bull_body", "bear_body", "wick", "volume"]


def test_positive_weights_are_training_stable_and_upweight_rare_labels():
    y = np.zeros((100, 3), dtype=np.float32)
    y[:50, 0] = 1
    y[:10, 1] = 1
    y[:2, 2] = 1
    w = weak_positive_weights(y, max_weight=20)
    assert np.all(np.isfinite(w))
    assert np.all(w >= 1)
    assert w[2] > w[1] > w[0]
    assert w.max() <= 20


def test_supported_metric_does_not_let_zero_support_labels_dominate_macro_f1():
    y = np.zeros((40, len(WEAK_LABEL_NAMES)), dtype=np.float32)
    y[:20, 0] = 1
    y[20:, 1] = 1
    p = np.zeros_like(y) + 0.1
    p[:20, 0] = 0.9
    p[20:, 1] = 0.9
    cfg = VisionAblationConfig(min_label_positives=5, min_label_negatives=5)
    m = supported_multilabel_metrics(y, p, cfg)
    assert m["supported_labels"] == 2
    assert m["supported_macro_f1"] > 0.99
    assert m["all_label_macro_f1"] < m["supported_macro_f1"]


def test_ablation_decision_fails_closed_for_weak_raw_detector():
    board = pd.DataFrame([
        {"status": "ok", "architecture": "small_cnn", "channel_mode": "raw_candles", "validation_objective": 0.2, "validation_supported_labels": 12, "validation_supported_macro_f1": 0.10, "validation_outcome_auc": 0.50, "test_supported_labels": 12, "test_supported_macro_f1": 0.12, "test_outcome_auc": 0.51},
        {"status": "ok", "architecture": "small_cnn", "channel_mode": "structure_augmented", "validation_objective": 0.3, "validation_supported_labels": 12, "validation_supported_macro_f1": 0.40, "validation_outcome_auc": 0.53, "test_supported_labels": 12, "test_supported_macro_f1": 0.35, "test_outcome_auc": 0.49},
    ])
    d = decide_ablation(board)
    assert d["decision"] == "NO_VISION_PROMOTION"
    assert d["raw_detector_pass"] is False
    assert d["live_execution_authorized"] is False


def test_ablation_can_accept_detector_only_after_validation_and_test_gates():
    board = pd.DataFrame([
        {"status": "ok", "architecture": "small_cnn", "channel_mode": "raw_candles", "validation_objective": 0.45, "validation_supported_labels": 10, "validation_supported_macro_f1": 0.35, "validation_outcome_auc": 0.54, "test_supported_labels": 10, "test_supported_macro_f1": 0.28, "test_outcome_auc": 0.51},
        {"status": "ok", "architecture": "small_vit", "channel_mode": "raw_candles", "validation_objective": 0.40, "validation_supported_labels": 10, "validation_supported_macro_f1": 0.30, "validation_outcome_auc": 0.70, "test_supported_labels": 10, "test_supported_macro_f1": 0.30, "test_outcome_auc": 0.70},
    ])
    d = decide_ablation(board)
    assert d["raw_champion"] == "small_cnn:raw_candles"
    assert d["raw_detector_pass"] is True
    assert d["decision"] == "RAW_ICT_DETECTOR_CANDIDATE"
    assert d["live_execution_authorized"] is False
