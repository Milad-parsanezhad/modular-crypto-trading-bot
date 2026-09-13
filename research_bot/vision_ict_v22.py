from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Iterable
import json

import numpy as np
import pandas as pd

from research_bot.multitimeframe_strategies_v19 import build_features

try:  # optional deep dependency
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover
    torch = None
    nn = None
    DataLoader = TensorDataset = None


@dataclass(frozen=True)
class VisionConfig:
    lookback: int = 96
    height: int = 128
    width: int = 192
    volume_fraction: float = 0.22
    line_radius: int = 1
    label_horizon: int = 1
    min_history: int = 220
    batch_size: int = 128
    epochs: int = 10
    learning_rate: float = 8e-4
    weight_decay: float = 1e-4
    patience: int = 3
    seed: int = 314


CHANNEL_NAMES = (
    "bull_body",
    "bear_body",
    "wick",
    "volume",
    "ichimoku",
    "liquidity_structure",
    "fvg_orderblock",
    "pd_regime",
)

WEAK_LABEL_NAMES = (
    "bull_candle",
    "bear_candle",
    "confirmed_swing_high_prevbar",
    "confirmed_swing_low_prevbar",
    "bos_up",
    "bos_down",
    "choch_up_proxy",
    "choch_down_proxy",
    "sweep_down",
    "sweep_up",
    "bull_fvg",
    "bear_fvg",
    "bull_orderblock_candidate",
    "bear_orderblock_candidate",
    "bull_mitigation_proxy",
    "bear_mitigation_proxy",
    "discount_location",
    "premium_location",
    "kumo_bull",
    "kumo_bear",
)


def require_torch() -> None:
    if torch is None:
        raise ImportError("PyTorch/torchvision are optional; install project extra: pip install -e '.[deep]'")


def _validate_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    cols = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = set(cols) - set(frame.columns)
    if missing:
        raise ValueError(f"missing OHLCV columns: {sorted(missing)}")
    x = frame[cols].copy()
    x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True, errors="coerce")
    for c in cols[1:]:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.dropna().drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    if len(x):
        bad = (x[["open", "high", "low", "close"]] <= 0).any(axis=1)
        bad |= x["high"] < x[["open", "close", "low"]].max(axis=1)
        bad |= x["low"] > x[["open", "close", "high"]].min(axis=1)
        if bad.any():
            raise ValueError(f"invalid OHLC rows: {int(bad.sum())}")
    return x


def _raw_source_hash(window: pd.DataFrame) -> str:
    cols = ["timestamp", "open", "high", "low", "close", "volume"]
    normalized = window[cols].copy()
    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], utc=True).astype("int64")
    payload = normalized.to_csv(index=False, float_format="%.12g").encode("utf-8")
    return sha256(payload).hexdigest()


def _clip_y(value: float, lo: float, hi: float, height: int) -> int:
    if not np.isfinite(value) or hi <= lo:
        return height // 2
    frac = (hi - value) / (hi - lo)
    return int(np.clip(round(frac * (height - 1)), 0, height - 1))


def _draw_vline(arr: np.ndarray, x: int, y0: int, y1: int, value: float = 1.0, radius: int = 0) -> None:
    a, b = sorted((int(y0), int(y1)))
    xl, xr = max(0, x - radius), min(arr.shape[1], x + radius + 1)
    arr[max(0, a):min(arr.shape[0], b + 1), xl:xr] = np.maximum(arr[max(0, a):min(arr.shape[0], b + 1), xl:xr], value)


def _draw_rect(arr: np.ndarray, x0: int, x1: int, y0: int, y1: int, value: float = 1.0) -> None:
    xa, xb = sorted((max(0, x0), min(arr.shape[1] - 1, x1)))
    ya, yb = sorted((max(0, y0), min(arr.shape[0] - 1, y1)))
    arr[ya:yb + 1, xa:xb + 1] = np.maximum(arr[ya:yb + 1, xa:xb + 1], value)


def _draw_polyline(arr: np.ndarray, xs: np.ndarray, ys: Iterable[float], lo: float, hi: float, value: float = 0.8, radius: int = 0) -> None:
    ys = list(ys)
    last = None
    for x, raw_y in zip(xs, ys):
        if not np.isfinite(raw_y):
            last = None
            continue
        y = _clip_y(float(raw_y), lo, hi, arr.shape[0])
        if last is not None:
            px, py = last
            steps = max(abs(int(x) - px), abs(y - py), 1)
            for k in range(steps + 1):
                xx = int(round(px + (int(x) - px) * k / steps))
                yy = int(round(py + (y - py) * k / steps))
                _draw_vline(arr, xx, yy, yy, value, radius)
        last = (int(x), y)


def ict_weak_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """Causal, rule-derived ICT/SMC structure labels.

    These labels are *weak structural labels*: they encode the project's formal
    proxies, not a claim that the named market concepts are ground truth.
    Every label at row t uses information available no later than the close of t.
    """
    x = _validate_ohlcv(frame)
    f = build_features(x)
    if f.empty:
        return pd.DataFrame(columns=["timestamp", *WEAK_LABEL_NAMES])

    out = pd.DataFrame({"timestamp": f["timestamp"]})
    out["bull_candle"] = (f["close"] > f["open"]).astype("int8")
    out["bear_candle"] = (f["close"] < f["open"]).astype("int8")

    sh = (f["high"].shift(1) > f["high"].shift(2)) & (f["high"].shift(1) >= f["high"])
    sl = (f["low"].shift(1) < f["low"].shift(2)) & (f["low"].shift(1) <= f["low"])
    out["confirmed_swing_high_prevbar"] = sh.fillna(False).astype("int8")
    out["confirmed_swing_low_prevbar"] = sl.fillna(False).astype("int8")
    out["bos_up"] = f["bos_up"].fillna(False).astype("int8")
    out["bos_down"] = f["bos_down"].fillna(False).astype("int8")

    structure = pd.Series(np.nan, index=f.index, dtype=float)
    structure.loc[f["bos_up"].fillna(False)] = 1.0
    structure.loc[f["bos_down"].fillna(False)] = -1.0
    prior_structure = structure.ffill().shift(1)
    out["choch_up_proxy"] = (f["bos_up"].fillna(False) & prior_structure.eq(-1)).astype("int8")
    out["choch_down_proxy"] = (f["bos_down"].fillna(False) & prior_structure.eq(1)).astype("int8")

    for c in ["sweep_down", "sweep_up", "bull_fvg", "bear_fvg"]:
        out[c] = f[c].fillna(False).astype("int8")

    prior_bear = (f["close"].shift(1) < f["open"].shift(1)).fillna(False)
    prior_bull = (f["close"].shift(1) > f["open"].shift(1)).fillna(False)
    out["bull_orderblock_candidate"] = (f["bos_up"].fillna(False) & prior_bear).astype("int8")
    out["bear_orderblock_candidate"] = (f["bos_down"].fillna(False) & prior_bull).astype("int8")

    bull_mit = f["bull_ob_mid"].notna() & (f["low"] <= f["bull_ob_mid"]) & (f["close"] > f["bull_ob_mid"])
    bear_mit = f["bear_ob_mid"].notna() & (f["high"] >= f["bear_ob_mid"]) & (f["close"] < f["bear_ob_mid"])
    out["bull_mitigation_proxy"] = bull_mit.fillna(False).astype("int8")
    out["bear_mitigation_proxy"] = bear_mit.fillna(False).astype("int8")

    # In the prior 20-bar dealing range: retracement > 0.5 means price is below midpoint (discount).
    out["discount_location"] = f["bull_retracement"].gt(0.5).fillna(False).astype("int8")
    out["premium_location"] = f["bull_retracement"].lt(0.5).fillna(False).astype("int8")
    out["kumo_bull"] = (f["close"] > f["cloud_top"]).fillna(False).astype("int8")
    out["kumo_bear"] = (f["close"] < f["cloud_bottom"]).fillna(False).astype("int8")
    return out[["timestamp", *WEAK_LABEL_NAMES]]


def render_multichannel_window(frame: pd.DataFrame, end_index: int, config: VisionConfig | None = None) -> tuple[np.ndarray, dict]:
    """Render a deterministic axes-free CxHxW tensor ending at end_index."""
    cfg = config or VisionConfig()
    x = _validate_ohlcv(frame)
    if end_index < cfg.lookback - 1 or end_index >= len(x):
        raise IndexError("end_index does not have the requested causal lookback")
    start = end_index - cfg.lookback + 1
    w = x.iloc[start:end_index + 1].reset_index(drop=True)
    f = build_features(x.iloc[:end_index + 1]).iloc[start:].reset_index(drop=True)

    h, width = cfg.height, cfg.width
    img = np.zeros((len(CHANNEL_NAMES), h, width), dtype=np.float32)
    price_h = max(16, int(round(h * (1.0 - cfg.volume_fraction))))
    price_hi = float(np.nanmax(w["high"].to_numpy(float)))
    price_lo = float(np.nanmin(w["low"].to_numpy(float)))
    pad = max((price_hi - price_lo) * 0.03, abs(price_hi) * 1e-8)
    price_hi += pad; price_lo -= pad
    xs = np.linspace(2, width - 3, len(w)).round().astype(int)
    body_half = max(1, int(round((width / max(1, len(w))) * 0.28)))

    # 0/1/2: bullish body, bearish body, common wicks.
    for i, row in w.iterrows():
        xx = int(xs[i])
        yo = _clip_y(float(row.open), price_lo, price_hi, price_h)
        yc = _clip_y(float(row.close), price_lo, price_hi, price_h)
        yh = _clip_y(float(row.high), price_lo, price_hi, price_h)
        yl = _clip_y(float(row.low), price_lo, price_hi, price_h)
        _draw_vline(img[2, :price_h], xx, yh, yl, 1.0, 0)
        ch = 0 if row.close >= row.open else 1
        _draw_rect(img[ch, :price_h], xx - body_half, xx + body_half, yo, yc, 1.0)

    # 3: volume; causal within-window normalization only.
    vol = w["volume"].to_numpy(float)
    vmax = float(np.nanmax(vol)) if np.isfinite(vol).any() else 1.0
    vmax = max(vmax, 1e-12)
    vol_top = price_h
    for i, value in enumerate(vol):
        bar_h = int(round((h - price_h - 1) * np.clip(value / vmax, 0, 1)))
        _draw_rect(img[3], int(xs[i]) - body_half, int(xs[i]) + body_half, h - 1 - bar_h, h - 1, 1.0)

    # 4: causal Ichimoku geometry. No visually forward-shifted cloud is used.
    _draw_polyline(img[4, :price_h], xs, f["tenkan"], price_lo, price_hi, 1.0, cfg.line_radius)
    _draw_polyline(img[4, :price_h], xs, f["kijun"], price_lo, price_hi, 0.75, cfg.line_radius)
    for i in range(len(f)):
        if np.isfinite(f["cloud_top"].iloc[i]) and np.isfinite(f["cloud_bottom"].iloc[i]):
            yt = _clip_y(float(f["cloud_top"].iloc[i]), price_lo, price_hi, price_h)
            yb = _clip_y(float(f["cloud_bottom"].iloc[i]), price_lo, price_hi, price_h)
            _draw_vline(img[4, :price_h], int(xs[i]), yt, yb, 0.30, 0)

    # 5: liquidity / BOS / sweep / swing reference channel.
    _draw_polyline(img[5, :price_h], xs, f["last_swing_high"], price_lo, price_hi, 0.35, 0)
    _draw_polyline(img[5, :price_h], xs, f["last_swing_low"], price_lo, price_hi, 0.35, 0)
    for i in range(len(f)):
        xx = int(xs[i])
        if bool(f["bos_up"].iloc[i]): _draw_vline(img[5, :price_h], xx, 0, price_h - 1, 0.85, 0)
        if bool(f["bos_down"].iloc[i]): _draw_vline(img[5, :price_h], xx, 0, price_h - 1, 0.65, 0)
        if bool(f["sweep_down"].iloc[i]): _draw_rect(img[5, :price_h], xx - body_half, xx + body_half, price_h - 5, price_h - 1, 1.0)
        if bool(f["sweep_up"].iloc[i]): _draw_rect(img[5, :price_h], xx - body_half, xx + body_half, 0, 4, 1.0)

    # 6: FVG and OB/mitigation context.
    for i in range(len(f)):
        xx = int(xs[i])
        if bool(f["bull_fvg"].iloc[i]): _draw_rect(img[6, :price_h], xx - body_half, xx + body_half, price_h - 8, price_h - 3, 1.0)
        if bool(f["bear_fvg"].iloc[i]): _draw_rect(img[6, :price_h], xx - body_half, xx + body_half, 3, 8, 1.0)
        for col, val in (("bull_ob_mid", 0.70), ("bear_ob_mid", 0.55), ("bull_fvg_mid", 0.45), ("bear_fvg_mid", 0.35)):
            raw = f[col].iloc[i]
            if np.isfinite(raw):
                yy = _clip_y(float(raw), price_lo, price_hi, price_h)
                _draw_vline(img[6, :price_h], xx, yy, yy, val, cfg.line_radius)

    # 7: premium/discount + coarse trend regime without text/color coding.
    mid = (price_hi + price_lo) / 2.0
    ym = _clip_y(mid, price_lo, price_hi, price_h)
    img[7, :ym, :] = 0.20  # premium half
    img[7, ym:price_h, :] = 0.40  # discount half
    above_cloud = (f["close"] > f["cloud_top"]).fillna(False).to_numpy()
    below_cloud = (f["close"] < f["cloud_bottom"]).fillna(False).to_numpy()
    for i, xx in enumerate(xs):
        if above_cloud[i]: _draw_vline(img[7, :price_h], int(xx), 0, 3, 0.9, 0)
        if below_cloud[i]: _draw_vline(img[7, :price_h], int(xx), price_h - 4, price_h - 1, 0.9, 0)

    img = np.clip(img, 0.0, 1.0).astype(np.float32, copy=False)
    source_hash = _raw_source_hash(w)
    render_hash = sha256(img.tobytes(order="C")).hexdigest()
    meta = {
        "signal_time": w["timestamp"].iloc[-1].isoformat(),
        "start_time": w["timestamp"].iloc[0].isoformat(),
        "lookback": cfg.lookback,
        "shape": list(img.shape),
        "channels": list(CHANNEL_NAMES),
        "source_bar_sha256": source_hash,
        "render_sha256": render_hash,
        "renderer_config": asdict(cfg),
        "axes_text_timestamps_drawn": False,
        "causal": True,
    }
    return img, meta


def build_vision_dataset(
    frame: pd.DataFrame,
    config: VisionConfig | None = None,
    stride: int = 1,
    max_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex, list[dict]]:
    """Build multi-channel tensors + weak ICT labels + future outcome label.

    The structural labels are causal at signal time. The outcome label is future
    information by design and is returned only as a target, never as a channel.
    """
    cfg = config or VisionConfig()
    x = _validate_ohlcv(frame)
    labels = ict_weak_labels(x).set_index("timestamp")
    start = max(cfg.lookback - 1, cfg.min_history - 1)
    end = len(x) - cfg.label_horizon
    indices = list(range(start, end, max(1, stride)))
    if max_samples is not None and len(indices) > max_samples:
        pick = np.linspace(0, len(indices) - 1, max_samples, dtype=int)
        indices = [indices[i] for i in pick]

    images: list[np.ndarray] = []
    weak: list[np.ndarray] = []
    outcome: list[float] = []
    times: list[pd.Timestamp] = []
    metas: list[dict] = []
    close = x["close"].to_numpy(float)
    for i in indices:
        img, meta = render_multichannel_window(x, i, cfg)
        t = x["timestamp"].iloc[i]
        if t not in labels.index:
            continue
        future_r = float(np.log(close[i + cfg.label_horizon] / close[i]))
        if not np.isfinite(future_r):
            continue
        images.append(img)
        weak.append(labels.loc[t, list(WEAK_LABEL_NAMES)].to_numpy(dtype=np.float32))
        outcome.append(float(future_r > 0))
        times.append(t)
        metas.append(meta)
    return (
        np.asarray(images, dtype=np.float32),
        np.asarray(weak, dtype=np.float32),
        np.asarray(outcome, dtype=np.float32),
        pd.DatetimeIndex(times),
        metas,
    )


def chronological_split(n: int, development: float = 0.60, validation: float = 0.20) -> tuple[slice, slice, slice]:
    a = int(n * development)
    b = int(n * (development + validation))
    return slice(0, a), slice(a, b), slice(b, n)


if nn is not None:
    class SmallMultiChannelCNN(nn.Module):
        def __init__(self, in_channels: int, weak_tasks: int):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(in_channels, 32, 5, stride=2, padding=2), nn.BatchNorm2d(32), nn.GELU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.GELU(),
                nn.Conv2d(64, 96, 3, stride=2, padding=1), nn.BatchNorm2d(96), nn.GELU(),
                nn.Conv2d(96, 128, 3, stride=2, padding=1), nn.GELU(), nn.AdaptiveAvgPool2d(1),
            )
            self.norm = nn.LayerNorm(128)
            self.weak_head = nn.Linear(128, weak_tasks)
            self.outcome_head = nn.Linear(128, 1)

        def forward(self, x):
            z = self.encoder(x).flatten(1)
            z = self.norm(z)
            return self.weak_head(z), self.outcome_head(z).squeeze(-1), z


    class SmallVisionTransformer(nn.Module):
        def __init__(self, in_channels: int, weak_tasks: int, image_size: tuple[int, int] = (128, 192), patch: int = 16, dim: int = 128, depth: int = 3, heads: int = 4):
            super().__init__()
            h, w = image_size
            if h % patch or w % patch:
                raise ValueError("image dimensions must be divisible by patch")
            self.patch = nn.Conv2d(in_channels, dim, patch, stride=patch)
            n = (h // patch) * (w // patch)
            self.cls = nn.Parameter(torch.zeros(1, 1, dim))
            self.pos = nn.Parameter(torch.zeros(1, n + 1, dim))
            layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout=0.1, batch_first=True, norm_first=True, activation="gelu")
            self.enc = nn.TransformerEncoder(layer, depth)
            self.norm = nn.LayerNorm(dim)
            self.weak_head = nn.Linear(dim, weak_tasks)
            self.outcome_head = nn.Linear(dim, 1)

        def forward(self, x):
            z = self.patch(x).flatten(2).transpose(1, 2)
            cls = self.cls.expand(len(x), -1, -1)
            z = torch.cat([cls, z], dim=1)
            z = self.enc(z + self.pos[:, :z.shape[1]])
            z = self.norm(z[:, 0])
            return self.weak_head(z), self.outcome_head(z).squeeze(-1), z


def make_vision_model(kind: str, in_channels: int = len(CHANNEL_NAMES), weak_tasks: int = len(WEAK_LABEL_NAMES), config: VisionConfig | None = None):
    require_torch()
    cfg = config or VisionConfig()
    if kind == "small_cnn":
        return SmallMultiChannelCNN(in_channels, weak_tasks)
    if kind == "small_vit":
        return SmallVisionTransformer(in_channels, weak_tasks, image_size=(cfg.height, cfg.width))
    if kind in {"resnet18", "efficientnet_b0"}:
        try:
            import torchvision.models as tvm
        except Exception as exc:  # pragma: no cover
            raise ImportError("torchvision is required for pretrained-style backbones") from exc
        if kind == "resnet18":
            net = tvm.resnet18(weights=None)
            net.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            dim = net.fc.in_features
            net.fc = nn.Identity()
        else:
            net = tvm.efficientnet_b0(weights=None)
            first = net.features[0][0]
            net.features[0][0] = nn.Conv2d(in_channels, first.out_channels, kernel_size=first.kernel_size, stride=first.stride, padding=first.padding, bias=False)
            dim = net.classifier[-1].in_features
            net.classifier = nn.Identity()

        class Wrapper(nn.Module):
            def __init__(self, backbone, dim, weak_tasks):
                super().__init__(); self.backbone = backbone; self.norm = nn.LayerNorm(dim); self.weak_head = nn.Linear(dim, weak_tasks); self.outcome_head = nn.Linear(dim, 1)
            def forward(self, x):
                z = self.norm(self.backbone(x)); return self.weak_head(z), self.outcome_head(z).squeeze(-1), z
        return Wrapper(net, dim, weak_tasks)
    raise ValueError(f"unknown vision model kind: {kind}")


def fit_multitask_model(model, x_train, weak_train, y_train, x_val, weak_val, y_val, config: VisionConfig | None = None, weak_weight: float = 0.35):
    require_torch()
    cfg = config or VisionConfig()
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    ds = TensorDataset(torch.tensor(x_train), torch.tensor(weak_train), torch.tensor(y_train))
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, generator=torch.Generator().manual_seed(cfg.seed))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    weak_loss = nn.BCEWithLogitsLoss()
    pos = max(1.0, float((y_train == 0).sum() / max(1, (y_train == 1).sum())))
    outcome_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos, device=device))
    xv = torch.tensor(x_val, device=device); wv = torch.tensor(weak_val, device=device); yv = torch.tensor(y_val, device=device)
    best, best_state, stale, history = np.inf, None, 0, []
    for epoch in range(cfg.epochs):
        model.train(); total = 0.0; n = 0
        for xb, wb, yb in loader:
            xb, wb, yb = xb.to(device), wb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            wl, ol, _ = model(xb)
            loss = weak_weight * weak_loss(wl, wb) + outcome_loss(ol, yb)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            total += float(loss.detach()) * len(xb); n += len(xb)
        model.eval()
        with torch.no_grad():
            wl, ol, _ = model(xv)
            vl = float((weak_weight * weak_loss(wl, wv) + outcome_loss(ol, yv)).cpu())
        history.append({"epoch": epoch + 1, "train_loss": total / max(1, n), "validation_loss": vl})
        if vl < best - 1e-5:
            best, stale = vl, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= cfg.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model.cpu(), pd.DataFrame(history)


def predict_multitask(model, x: np.ndarray, batch_size: int = 512) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    require_torch(); model.eval(); weak, outcome, emb = [], [], []
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


def save_vision_manifest(path: Path, config: VisionConfig, times: pd.DatetimeIndex, metas: list[dict], shape: tuple[int, ...]) -> dict:
    joined = "".join(m["render_sha256"] for m in metas).encode("ascii")
    manifest = {
        "version": "v0.22",
        "renderer": "deterministic_axes_free_multichannel_candlestick_ict",
        "config": asdict(config),
        "channels": list(CHANNEL_NAMES),
        "weak_structural_labels": list(WEAK_LABEL_NAMES),
        "dataset_shape": list(shape),
        "first_signal_time": times[0].isoformat() if len(times) else None,
        "last_signal_time": times[-1].isoformat() if len(times) else None,
        "sample_count": int(len(times)),
        "render_chain_sha256": sha256(joined).hexdigest(),
        "causality": "each image ends at signal_time; no axes/text/timestamp glyphs; structural labels are known by that close",
        "label_semantics": "ICT/SMC labels are weak rule-derived structural labels; future outcome is a separate supervised target",
    }
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
