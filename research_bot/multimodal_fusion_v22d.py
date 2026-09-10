from __future__ import annotations

"""v0.22d causal multimodal chart + numerical representation lab.

The primary eligible multimodal arm fuses RAW chart geometry with causal numeric
features, including only the scientifically supported component tier from
``scientific_liquidity_wyckoff_v22d``. Course-specific Wyckoff proxies are
isolated in an exploratory arm and can never promote a strategy in this module.

Outcome target: signal after close[t], hypothetical fill at open[t+1], exit at
open[t+2], and label positive only when gross return exceeds the configured
round-trip cost hurdle. This is a representation diagnostic, not a trading
strategy or live authorization.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from research_bot.ict_localization_v22c import LocalizationConfig, raw_candle_tensor
from research_bot.scientific_liquidity_wyckoff_v22d import (
    COURSE_HYPOTHESIS_FEATURES,
    SUPPORTED_COMPONENT_FEATURES,
    build_scientific_liquidity_features,
)

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover
    torch = None
    nn = None
    DataLoader = TensorDataset = None


CORE_NUMERIC_FEATURES = (
    "ret1",
    "ret12",
    "atr_pct",
    "ema20_rel",
    "ema50_rel",
    "ema200_rel",
    "ema200_slope_atr",
    "kijun_rel",
    "cloud_top_rel",
    "cloud_bottom_rel",
    "bull_retracement",
    "bear_retracement",
)


@dataclass(frozen=True)
class MultimodalConfig:
    lookback: int = 64
    height: int = 64
    width: int = 96
    min_history: int = 220
    max_samples: int = 480
    fee_bps: float = 10.0
    slippage_bps: float = 2.0
    batch_size: int = 96
    epochs: int = 6
    learning_rate: float = 8e-4
    weight_decay: float = 1e-4
    patience: int = 2
    seed: int = 314
    min_validation_auc_gain: float = 0.015
    min_test_auc: float = 0.53
    max_test_auc_regret: float = 0.005
    min_test_balanced_accuracy: float = 0.51

    @property
    def roundtrip_cost(self) -> float:
        return 2.0 * (self.fee_bps + self.slippage_bps) / 10000.0


@dataclass
class RobustNumericScaler:
    median: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "RobustNumericScaler":
        med = np.nanmedian(x, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        q25 = np.nanpercentile(x, 25, axis=0)
        q75 = np.nanpercentile(x, 75, axis=0)
        scale = q75 - q25
        scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)
        return cls(med.astype(np.float32), scale.astype(np.float32))

    def transform(self, x: np.ndarray) -> np.ndarray:
        z = np.where(np.isfinite(x), x, self.median)
        return ((z - self.median) / self.scale).clip(-12, 12).astype(np.float32)


def _require_torch() -> None:
    if torch is None:
        raise ImportError("Install deep dependencies with pip install -e '.[deep]'")


def _feature_views(f: pd.DataFrame) -> pd.DataFrame:
    x = f.copy()
    close = x["close"].replace(0, np.nan)
    atr = x["atr"].replace(0, np.nan)
    x["ema20_rel"] = x["ema20"] / close - 1.0
    x["ema50_rel"] = x["ema50"] / close - 1.0
    x["ema200_rel"] = x["ema200"] / close - 1.0
    x["ema200_slope_atr"] = x["ema200_slope"] / atr
    x["kijun_rel"] = x["kijun"] / close - 1.0
    x["cloud_top_rel"] = x["cloud_top"] / close - 1.0
    x["cloud_bottom_rel"] = x["cloud_bottom"] / close - 1.0
    return x


def build_multimodal_dataset(
    frame: pd.DataFrame,
    config: MultimodalConfig | None = None,
) -> dict[str, object]:
    cfg = config or MultimodalConfig()
    f = _feature_views(build_scientific_liquidity_features(frame)).reset_index(drop=True)
    if len(f) < cfg.min_history + 3:
        raise ValueError("insufficient history")

    start = max(cfg.lookback - 1, cfg.min_history - 1)
    end_exclusive = len(f) - 2  # t+1 fill and t+2 exit must exist.
    indices = list(range(start, end_exclusive))
    if len(indices) > cfg.max_samples:
        pick = np.linspace(0, len(indices) - 1, cfg.max_samples, dtype=int)
        indices = [indices[i] for i in pick]

    loc_cfg = LocalizationConfig(
        lookback=cfg.lookback,
        height=cfg.height,
        width=cfg.width,
        min_history=cfg.min_history,
    )
    raw, core, supported, course, targets, gross, times = [], [], [], [], [], [], []
    opens = f["open"].to_numpy(float)
    for i in indices:
        img, _ = raw_candle_tensor(f[["timestamp", "open", "high", "low", "close", "volume"]], i, loc_cfg)
        g = float(opens[i + 2] / opens[i + 1] - 1.0)
        raw.append(img)
        core.append(f.loc[i, list(CORE_NUMERIC_FEATURES)].to_numpy(float))
        supported.append(f.loc[i, list(SUPPORTED_COMPONENT_FEATURES)].to_numpy(float))
        course.append(f.loc[i, list(COURSE_HYPOTHESIS_FEATURES)].to_numpy(float))
        targets.append(float(g > cfg.roundtrip_cost))
        gross.append(g)
        times.append(f.loc[i, "timestamp"])

    return {
        "image": np.asarray(raw, dtype=np.float32),
        "core": np.asarray(core, dtype=np.float32),
        "supported": np.asarray(supported, dtype=np.float32),
        "course": np.asarray(course, dtype=np.float32),
        "target": np.asarray(targets, dtype=np.float32),
        "gross_return": np.asarray(gross, dtype=np.float32),
        "timestamp": pd.DatetimeIndex(times),
    }


def chronological_split(n: int, development: float = 0.60, validation: float = 0.20):
    a = int(n * development)
    b = int(n * (development + validation))
    return slice(0, a), slice(a, b), slice(b, n)


if nn is not None:
    class ImageEncoder(nn.Module):
        def __init__(self, in_channels: int = 4, dim: int = 64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, 24, 5, stride=2, padding=2), nn.GELU(),
                nn.Conv2d(24, 48, 3, stride=2, padding=1), nn.GELU(),
                nn.Conv2d(48, 72, 3, stride=2, padding=1), nn.GELU(),
                nn.Conv2d(72, dim, 3, stride=2, padding=1), nn.GELU(),
                nn.AdaptiveAvgPool2d(1),
            )
            self.norm = nn.LayerNorm(dim)

        def forward(self, x):
            return self.norm(self.net(x).flatten(1))


    class NumericEncoder(nn.Module):
        def __init__(self, n: int, dim: int = 48):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(n, 96), nn.GELU(), nn.Dropout(0.10), nn.Linear(96, dim), nn.GELU())
            self.norm = nn.LayerNorm(dim)

        def forward(self, x):
            return self.norm(self.net(x))


    class NumericOnly(nn.Module):
        def __init__(self, n: int):
            super().__init__(); self.enc = NumericEncoder(n); self.head = nn.Linear(48, 1)
        def forward(self, image, numeric):
            z = self.enc(numeric); return self.head(z).squeeze(-1), z


    class ImageOnly(nn.Module):
        def __init__(self):
            super().__init__(); self.enc = ImageEncoder(); self.head = nn.Linear(64, 1)
        def forward(self, image, numeric):
            z = self.enc(image); return self.head(z).squeeze(-1), z


    class FusionNet(nn.Module):
        def __init__(self, n: int):
            super().__init__()
            self.image = ImageEncoder()
            self.numeric = NumericEncoder(n)
            self.fuse = nn.Sequential(nn.Linear(64 + 48, 64), nn.GELU(), nn.Dropout(0.10), nn.Linear(64, 1))
        def forward(self, image, numeric):
            zi, zn = self.image(image), self.numeric(numeric)
            z = torch.cat([zi, zn], dim=1)
            return self.fuse(z).squeeze(-1), z


def make_model(kind: str, numeric_dim: int):
    _require_torch()
    if kind == "numeric": return NumericOnly(numeric_dim)
    if kind == "image": return ImageOnly()
    if kind == "fusion": return FusionNet(numeric_dim)
    raise ValueError(kind)


def fit_binary_model(model, image_dev, num_dev, y_dev, image_val, num_val, y_val, config: MultimodalConfig | None = None):
    _require_torch()
    cfg = config or MultimodalConfig()
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    torch.set_num_threads(min(4, max(1, torch.get_num_threads())))
    ds = TensorDataset(torch.from_numpy(image_dev), torch.from_numpy(num_dev), torch.from_numpy(y_dev.astype(np.float32)))
    g = torch.Generator().manual_seed(cfg.seed)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, generator=g)
    positives = max(float(y_dev.sum()), 1.0)
    negatives = max(float(len(y_dev) - y_dev.sum()), 1.0)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negatives / positives, dtype=torch.float32))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    best = None; best_loss = float("inf"); stale = 0; history = []
    xv_i, xv_n, yv = torch.from_numpy(image_val), torch.from_numpy(num_val), torch.from_numpy(y_val.astype(np.float32))
    for epoch in range(cfg.epochs):
        model.train(); losses = []
        for xi, xn, yy in dl:
            opt.zero_grad(set_to_none=True)
            logits, _ = model(xi, xn)
            loss = loss_fn(logits, yy)
            loss.backward(); opt.step(); losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            logits, _ = model(xv_i, xv_n); val_loss = float(loss_fn(logits, yv))
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "validation_loss": val_loss})
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= cfg.patience: break
    if best is not None: model.load_state_dict(best)
    return model, pd.DataFrame(history)


def predict_binary(model, image, numeric):
    _require_torch(); model.eval()
    with torch.no_grad():
        logits, embedding = model(torch.from_numpy(image), torch.from_numpy(numeric))
        prob = torch.sigmoid(logits).cpu().numpy()
        emb = embedding.cpu().numpy()
    return prob.astype(np.float32), emb.astype(np.float32)


def binary_metrics(y_true: np.ndarray, prob: np.ndarray) -> dict:
    y = y_true.astype(int); p = np.asarray(prob, dtype=float)
    if len(np.unique(y)) < 2:
        auc = float("nan")
    else:
        auc = float(roc_auc_score(y, p))
    pred = (p >= 0.5).astype(int)
    return {
        "n": int(len(y)),
        "positive_rate": float(y.mean()) if len(y) else float("nan"),
        "auc": auc,
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)) if len(y) else float("nan"),
    }


def economic_diagnostic(gross_return: np.ndarray, prob: np.ndarray, roundtrip_cost: float, threshold: float = 0.60) -> dict:
    chosen = np.asarray(prob) >= threshold
    net = np.asarray(gross_return, dtype=float)[chosen] - roundtrip_cost
    return {
        "threshold": float(threshold),
        "selected_events": int(chosen.sum()),
        "selection_rate": float(chosen.mean()) if len(chosen) else 0.0,
        "mean_net_return_per_selected_event": float(net.mean()) if len(net) else float("nan"),
        "median_net_return_per_selected_event": float(np.median(net)) if len(net) else float("nan"),
        "net_positive_hit_rate": float((net > 0).mean()) if len(net) else float("nan"),
        "compounded_selected_event_return": float(np.prod(1.0 + net) - 1.0) if len(net) else float("nan"),
        "note": "diagnostic on sampled independent decision events; not an annualized backtest",
    }


def decide_multimodal(records: dict[str, dict], cfg: MultimodalConfig) -> dict:
    base_v = records["numeric_supported"]["validation"]["auc"]
    base_t = records["numeric_supported"]["test"]["auc"]
    img_v = records["image_raw"]["validation"]["auc"]
    img_t = records["image_raw"]["test"]["auc"]
    fus_v = records["fusion_supported"]["validation"]["auc"]
    fus_t = records["fusion_supported"]["test"]["auc"]
    fus_bacc = records["fusion_supported"]["test"]["balanced_accuracy"]
    val_gain = fus_v - max(base_v, img_v)
    test_regret = max(base_t, img_t) - fus_t
    passed = bool(
        np.isfinite(val_gain) and val_gain >= cfg.min_validation_auc_gain
        and np.isfinite(fus_t) and fus_t >= cfg.min_test_auc
        and np.isfinite(test_regret) and test_regret <= cfg.max_test_auc_regret
        and np.isfinite(fus_bacc) and fus_bacc >= cfg.min_test_balanced_accuracy
    )
    course_v_gain = records["fusion_supported_plus_course"]["validation"]["auc"] - fus_v
    course_t_gain = records["fusion_supported_plus_course"]["test"]["auc"] - fus_t
    course_candidate = bool(course_v_gain >= 0.01 and course_t_gain >= 0.0)
    return {
        "decision": "MULTIMODAL_REPRESENTATION_CANDIDATE" if passed else "NO_MULTIMODAL_ENCODER_PROMOTED",
        "passed": passed,
        "primary_comparison": "fusion_supported vs max(numeric_supported, image_raw)",
        "validation_auc_gain": float(val_gain),
        "test_auc_regret": float(test_regret),
        "gates": {
            "min_validation_auc_gain": cfg.min_validation_auc_gain,
            "min_test_auc": cfg.min_test_auc,
            "max_test_auc_regret": cfg.max_test_auc_regret,
            "min_test_balanced_accuracy": cfg.min_test_balanced_accuracy,
        },
        "course_proxy_exploratory": {
            "validation_auc_gain_vs_supported_fusion": float(course_v_gain),
            "test_auc_gain_vs_supported_fusion": float(course_t_gain),
            "status": "COURSE_PROXY_INCREMENTAL_SIGNAL_CANDIDATE" if course_candidate else "NO_COURSE_PROXY_INCREMENTAL_EVIDENCE",
            "eligible_for_strategy_promotion": False,
        },
        "vision_to_rl_state_connected": False,
        "paper_replacement_authorized": False,
        "live_execution_authorized": False,
    }


def save_protocol(path: str | Path, cfg: MultimodalConfig) -> None:
    Path(path).write_text(json.dumps({
        "version": "v0.22d",
        "config": asdict(cfg),
        "core_features": list(CORE_NUMERIC_FEATURES),
        "supported_component_features": list(SUPPORTED_COMPONENT_FEATURES),
        "course_hypothesis_features": list(COURSE_HYPOTHESIS_FEATURES),
        "scientific_contract": "Course hypotheses are isolated and never eligible for strategy promotion in v0.22d. Test is untouched until training/early stopping are frozen. No RL connection.",
    }, indent=2), encoding="utf-8")
