from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd

from research_bot.multitimeframe_strategies_v19 import build_features
from research_bot.vision_ict_v22 import CHANNEL_NAMES, VisionConfig, _validate_ohlcv, render_multichannel_window

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover
    torch = None
    nn = None
    DataLoader = TensorDataset = None


LOCALIZATION_LABELS = (
    "swing_high",
    "swing_low",
    "bos_up",
    "bos_down",
    "choch_up",
    "choch_down",
    "sweep_up",
    "sweep_down",
    "bull_fvg_zone",
    "bear_fvg_zone",
    "bull_ob_origin",
    "bear_ob_origin",
)

RAW_INPUT_CHANNELS = (0, 1, 2, 3)


@dataclass(frozen=True)
class LocalizationConfig:
    lookback: int = 64
    height: int = 64
    width: int = 96
    volume_fraction: float = 0.22
    min_history: int = 160
    event_radius_x: int = 1
    event_radius_y: int = 2
    batch_size: int = 64
    epochs: int = 8
    learning_rate: float = 8e-4
    weight_decay: float = 1e-4
    patience: int = 3
    bce_weight: float = 0.65
    dice_weight: float = 0.35
    max_pos_weight: float = 30.0
    min_supported_positive_pixels: int = 30
    seed: int = 314


def require_torch() -> None:
    if torch is None:
        raise ImportError("PyTorch is required for the v0.22c localization model")


def _price_y(value: float, lo: float, hi: float, price_h: int) -> int:
    if not np.isfinite(value) or hi <= lo:
        return price_h // 2
    return int(np.clip(round((hi - value) / (hi - lo) * (price_h - 1)), 0, price_h - 1))


def _paint(mask: np.ndarray, channel: int, x0: int, x1: int, y0: int, y1: int) -> None:
    xa, xb = sorted((max(0, int(x0)), min(mask.shape[2] - 1, int(x1))))
    ya, yb = sorted((max(0, int(y0)), min(mask.shape[1] - 1, int(y1))))
    if xa <= xb and ya <= yb:
        mask[channel, ya:yb + 1, xa:xb + 1] = 1.0


def _event_box(mask: np.ndarray, channel: int, x: int, y: int, cfg: LocalizationConfig) -> None:
    _paint(mask, channel, x - cfg.event_radius_x, x + cfg.event_radius_x, y - cfg.event_radius_y, y + cfg.event_radius_y)


def _window_geometry(window: pd.DataFrame, cfg: LocalizationConfig) -> tuple[int, float, float, np.ndarray, int]:
    price_h = max(16, int(round(cfg.height * (1.0 - cfg.volume_fraction))))
    hi = float(np.nanmax(window["high"].to_numpy(float)))
    lo = float(np.nanmin(window["low"].to_numpy(float)))
    pad = max((hi - lo) * 0.03, abs(hi) * 1e-8)
    hi += pad
    lo -= pad
    xs = np.linspace(2, cfg.width - 3, len(window)).round().astype(int)
    body_half = max(1, int(round((cfg.width / max(1, len(window))) * 0.28)))
    return price_h, lo, hi, xs, body_half


def ict_localization_masks(frame: pd.DataFrame, end_index: int, config: LocalizationConfig | None = None) -> tuple[np.ndarray, dict]:
    """Build causal spatial masks for formalized ICT/SMC structures.

    The mask for a sample ending at t may use confirmation information available
    by the close of t, but never bars after t. Swing masks therefore identify a
    previous bar only after its right-hand confirmation has closed.
    """
    cfg = config or LocalizationConfig()
    x = _validate_ohlcv(frame)
    if end_index < cfg.lookback - 1 or end_index >= len(x):
        raise IndexError("end_index does not have the requested causal lookback")
    start = end_index - cfg.lookback + 1
    prefix = x.iloc[:end_index + 1].copy()
    f = build_features(prefix).reset_index(drop=True)
    w = prefix.iloc[start:end_index + 1].reset_index(drop=True)
    fw = f.iloc[start:end_index + 1].reset_index(drop=True)
    price_h, lo, hi, xs, body_half = _window_geometry(w, cfg)
    masks = np.zeros((len(LOCALIZATION_LABELS), cfg.height, cfg.width), dtype=np.float32)

    # Structure state is computed on the complete causal prefix, then sliced.
    structure = pd.Series(np.nan, index=f.index, dtype=float)
    structure.loc[f["bos_up"].fillna(False)] = 1.0
    structure.loc[f["bos_down"].fillna(False)] = -1.0
    prior_structure = structure.ffill().shift(1)

    for local_i in range(len(w)):
        global_i = start + local_i
        xx = int(xs[local_i])
        row = w.iloc[local_i]
        feat = f.iloc[global_i]

        # A swing at bar i-1 becomes knowable only when bar i closes.
        if global_i >= 2:
            prev = x.iloc[global_i - 1]
            prev2 = x.iloc[global_i - 2]
            confirmed_sh = bool(prev.high > prev2.high and prev.high >= row.high)
            confirmed_sl = bool(prev.low < prev2.low and prev.low <= row.low)
            if local_i - 1 >= 0:
                prev_x = int(xs[local_i - 1])
                if confirmed_sh:
                    _event_box(masks, 0, prev_x, _price_y(float(prev.high), lo, hi, price_h), cfg)
                if confirmed_sl:
                    _event_box(masks, 1, prev_x, _price_y(float(prev.low), lo, hi, price_h), cfg)

        prior_hi = feat.get("last_swing_high", np.nan)
        prior_lo = feat.get("last_swing_low", np.nan)
        bos_up = bool(feat.get("bos_up", False))
        bos_down = bool(feat.get("bos_down", False))
        if bos_up:
            _event_box(masks, 2, xx, _price_y(float(prior_hi), lo, hi, price_h), cfg)
            if prior_structure.iloc[global_i] == -1:
                _event_box(masks, 4, xx, _price_y(float(prior_hi), lo, hi, price_h), cfg)
        if bos_down:
            _event_box(masks, 3, xx, _price_y(float(prior_lo), lo, hi, price_h), cfg)
            if prior_structure.iloc[global_i] == 1:
                _event_box(masks, 5, xx, _price_y(float(prior_lo), lo, hi, price_h), cfg)

        if bool(feat.get("sweep_up", False)):
            _event_box(masks, 6, xx, _price_y(float(prior_hi), lo, hi, price_h), cfg)
        if bool(feat.get("sweep_down", False)):
            _event_box(masks, 7, xx, _price_y(float(prior_lo), lo, hi, price_h), cfg)

        # FVG geometry is directly defined by bars t and t-2; fill the gap zone.
        if global_i >= 2:
            old = x.iloc[global_i - 2]
            if row.low > old.high:  # bullish FVG [old.high, row.low]
                y0 = _price_y(float(row.low), lo, hi, price_h)
                y1 = _price_y(float(old.high), lo, hi, price_h)
                left_x = int(xs[max(0, local_i - 2)])
                _paint(masks, 8, left_x, xx, y0, y1)
            if row.high < old.low:  # bearish FVG [row.high, old.low]
                y0 = _price_y(float(old.low), lo, hi, price_h)
                y1 = _price_y(float(row.high), lo, hi, price_h)
                left_x = int(xs[max(0, local_i - 2)])
                _paint(masks, 9, left_x, xx, y0, y1)

        # Order-block origin is labelled only after the BOS confirmation occurs.
        # The target points back to the immediately preceding opposing candle.
        if global_i >= 1 and local_i >= 1:
            prev = x.iloc[global_i - 1]
            px = int(xs[local_i - 1])
            if bos_up and prev.close < prev.open:
                _paint(masks, 10, px - body_half, px + body_half,
                       _price_y(float(prev.high), lo, hi, price_h),
                       _price_y(float(prev.low), lo, hi, price_h))
            if bos_down and prev.close > prev.open:
                _paint(masks, 11, px - body_half, px + body_half,
                       _price_y(float(prev.high), lo, hi, price_h),
                       _price_y(float(prev.low), lo, hi, price_h))

    meta = {
        "signal_time": w["timestamp"].iloc[-1].isoformat(),
        "labels": list(LOCALIZATION_LABELS),
        "shape": list(masks.shape),
        "positive_pixels": {name: int(masks[j].sum()) for j, name in enumerate(LOCALIZATION_LABELS)},
        "causal": True,
        "input_requirement": "raw candlestick/volume channels only for independent structure-localization experiments",
    }
    return masks, meta


def raw_candle_tensor(frame: pd.DataFrame, end_index: int, config: LocalizationConfig | None = None) -> tuple[np.ndarray, dict]:
    cfg = config or LocalizationConfig()
    vcfg = VisionConfig(
        lookback=cfg.lookback,
        height=cfg.height,
        width=cfg.width,
        volume_fraction=cfg.volume_fraction,
        min_history=cfg.min_history,
    )
    full, meta = render_multichannel_window(frame, end_index, vcfg)
    raw = full[list(RAW_INPUT_CHANNELS)].copy()
    meta = dict(meta)
    meta["input_channels"] = [CHANNEL_NAMES[i] for i in RAW_INPUT_CHANNELS]
    meta["engineered_ict_channels_in_input"] = False
    return raw, meta


def build_localization_dataset(
    frame: pd.DataFrame,
    config: LocalizationConfig | None = None,
    stride: int = 1,
    max_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, list[dict]]:
    cfg = config or LocalizationConfig()
    x = _validate_ohlcv(frame)
    start = max(cfg.lookback - 1, cfg.min_history - 1)
    indices = list(range(start, len(x), max(1, stride)))
    if max_samples is not None and len(indices) > max_samples:
        pick = np.linspace(0, len(indices) - 1, max_samples, dtype=int)
        indices = [indices[i] for i in pick]
    inputs, targets, times, metas = [], [], [], []
    for i in indices:
        raw, rmeta = raw_candle_tensor(x, i, cfg)
        mask, mmeta = ict_localization_masks(x, i, cfg)
        inputs.append(raw)
        targets.append(mask)
        times.append(x["timestamp"].iloc[i])
        metas.append({"renderer": rmeta, "mask": mmeta})
    return (
        np.asarray(inputs, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
        pd.DatetimeIndex(times),
        metas,
    )


def chronological_split(n: int, development: float = 0.60, validation: float = 0.20) -> tuple[slice, slice, slice]:
    a = int(n * development)
    b = int(n * (development + validation))
    return slice(0, a), slice(a, b), slice(b, n)


def localization_pos_weights(targets: np.ndarray, max_weight: float = 30.0) -> np.ndarray:
    """Development-only damped positive pixel weights."""
    if targets.ndim != 4 or targets.shape[1] != len(LOCALIZATION_LABELS):
        raise ValueError("expected [N,C,H,W] localization targets")
    pos = targets.sum(axis=(0, 2, 3)).astype(float)
    total = targets.shape[0] * targets.shape[2] * targets.shape[3]
    neg = total - pos
    weight = np.sqrt((neg + 1.0) / (pos + 1.0))
    return np.clip(weight, 1.0, max_weight).astype(np.float32)


def _binary_spatial_metrics(target: np.ndarray, probability: np.ndarray, min_positive_pixels: int = 30) -> dict:
    if target.shape != probability.shape:
        raise ValueError("target/probability shapes differ")
    pred = probability >= 0.5
    truth = target >= 0.5
    rows = {}
    dices, ious = [], []
    for j, name in enumerate(LOCALIZATION_LABELS):
        tp = int(np.logical_and(pred[:, j], truth[:, j]).sum())
        fp = int(np.logical_and(pred[:, j], ~truth[:, j]).sum())
        fn = int(np.logical_and(~pred[:, j], truth[:, j]).sum())
        positives = int(truth[:, j].sum())
        dice = float((2 * tp) / max(1, 2 * tp + fp + fn))
        iou = float(tp / max(1, tp + fp + fn))
        supported = positives >= min_positive_pixels
        rows[name] = {"dice": dice, "iou": iou, "positive_pixels": positives, "supported": supported}
        if supported:
            dices.append(dice); ious.append(iou)
    return {
        "supported_labels": len(dices),
        "supported_macro_dice": float(np.mean(dices)) if dices else np.nan,
        "supported_macro_iou": float(np.mean(ious)) if ious else np.nan,
        "per_label": rows,
    }


if nn is not None:
    def _block(cin: int, cout: int):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.GELU(),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.GELU(),
        )


    class SmallUNet(nn.Module):
        def __init__(self, in_channels: int = 4, out_channels: int = len(LOCALIZATION_LABELS)):
            super().__init__()
            self.e1 = _block(in_channels, 32)
            self.e2 = _block(32, 64)
            self.b = _block(64, 128)
            self.pool = nn.MaxPool2d(2)
            self.u2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
            self.d2 = _block(128, 64)
            self.u1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
            self.d1 = _block(64, 32)
            self.out = nn.Conv2d(32, out_channels, 1)

        def forward(self, x):
            e1 = self.e1(x)
            e2 = self.e2(self.pool(e1))
            b = self.b(self.pool(e2))
            d2 = self.d2(torch.cat([self.u2(b), e2], dim=1))
            d1 = self.d1(torch.cat([self.u1(d2), e1], dim=1))
            return self.out(d1)


def make_localizer(kind: str = "small_unet", in_channels: int = 4):
    require_torch()
    if kind == "small_unet":
        return SmallUNet(in_channels=in_channels)
    raise ValueError(f"unknown localizer: {kind}")


def _dice_loss_from_logits(logits, target, eps: float = 1.0):
    p = torch.sigmoid(logits)
    dims = (0, 2, 3)
    inter = (p * target).sum(dims)
    denom = p.sum(dims) + target.sum(dims)
    return 1.0 - ((2.0 * inter + eps) / (denom + eps)).mean()


def fit_localizer(
    model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    config: LocalizationConfig | None = None,
):
    require_torch()
    cfg = config or LocalizationConfig()
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    ds = TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, generator=torch.Generator().manual_seed(cfg.seed))
    pw = torch.tensor(localization_pos_weights(y_train, cfg.max_pos_weight), device=device).view(1, -1, 1, 1)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    xv = torch.tensor(x_val, device=device); yv = torch.tensor(y_val, device=device)
    best, best_state, stale, history = np.inf, None, 0, []
    for epoch in range(cfg.epochs):
        model.train(); total = 0.0; n = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            bce = nn.functional.binary_cross_entropy_with_logits(logits, yb, pos_weight=pw)
            dice = _dice_loss_from_logits(logits, yb)
            loss = cfg.bce_weight * bce + cfg.dice_weight * dice
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            total += float(loss.detach()) * len(xb); n += len(xb)
        model.eval()
        with torch.no_grad():
            logits = model(xv)
            vbce = nn.functional.binary_cross_entropy_with_logits(logits, yv, pos_weight=pw)
            vdice = _dice_loss_from_logits(logits, yv)
            vloss = float((cfg.bce_weight * vbce + cfg.dice_weight * vdice).cpu())
        history.append({"epoch": epoch + 1, "train_loss": total / max(1, n), "validation_loss": vloss})
        if vloss < best - 1e-5:
            best, stale = vloss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= cfg.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model.cpu(), pd.DataFrame(history), pw.view(-1).cpu().numpy()


def predict_localizer(model, x: np.ndarray, batch_size: int = 128) -> np.ndarray:
    require_torch(); model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            out.append(torch.sigmoid(model(torch.tensor(x[i:i + batch_size]))).cpu().numpy())
    return np.concatenate(out) if out else np.empty((0, len(LOCALIZATION_LABELS), 0, 0), dtype=np.float32)


def localization_metrics(target: np.ndarray, probability: np.ndarray, config: LocalizationConfig | None = None) -> dict:
    cfg = config or LocalizationConfig()
    return _binary_spatial_metrics(target, probability, cfg.min_supported_positive_pixels)


def save_localization_manifest(path: Path, cfg: LocalizationConfig) -> None:
    path.write_text(json.dumps({
        "version": "v0.22c",
        "input_channels": [CHANNEL_NAMES[i] for i in RAW_INPUT_CHANNELS],
        "engineered_ict_channels_in_input": False,
        "target_masks": list(LOCALIZATION_LABELS),
        "config": cfg.__dict__,
        "causal_rule": "input and spatial targets use bars available by sample close; future bars are prohibited",
        "scientific_role": "tests whether raw candle/volume geometry can localize formalized ICT structure; not a trading-alpha claim",
    }, indent=2), encoding="utf-8")
