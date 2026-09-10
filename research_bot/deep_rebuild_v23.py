from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover
    torch = None
    nn = None
    DataLoader = TensorDataset = None


@dataclass(frozen=True)
class DeepRebuildConfig:
    lookback: int = 64
    hidden_dim: int = 64
    layers: int = 2
    dropout: float = 0.15
    batch_size: int = 256
    epochs: int = 10
    learning_rate: float = 8e-4
    weight_decay: float = 1e-4
    patience: int = 3
    fee_bps_each_way: float = 10.0
    slippage_bps_each_way: float = 2.0
    min_active_validation: int = 100

    @property
    def one_way_cost(self) -> float:
        return (self.fee_bps_each_way + self.slippage_bps_each_way) / 10000.0

    @property
    def roundtrip_cost(self) -> float:
        return 2.0 * self.one_way_cost


def require_torch() -> None:
    if torch is None:
        raise ImportError("Install deep dependencies with pip install -e '.[deep]'")


def causal_bar_features(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.copy()
    x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True, errors="coerce")
    for c in ("open", "high", "low", "close", "volume"):
        x[c] = pd.to_numeric(x[c], errors="coerce")
    prev_close = x["close"].shift(1)
    tr = pd.concat([
        x["high"] - x["low"],
        (x["high"] - prev_close).abs(),
        (x["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean().replace(0, np.nan)
    rng = (x["high"] - x["low"]).replace(0, np.nan)
    body = x["close"] - x["open"]
    upper = x["high"] - np.maximum(x["open"], x["close"])
    lower = np.minimum(x["open"], x["close"]) - x["low"]
    ema20 = x["close"].ewm(span=20, adjust=False).mean()
    ema50 = x["close"].ewm(span=50, adjust=False).mean()
    ema200 = x["close"].ewm(span=200, adjust=False).mean()
    vol_med = x["volume"].shift(1).rolling(20, min_periods=5).median().replace(0, np.nan)
    hh20 = x["high"].shift(1).rolling(20, min_periods=5).max()
    ll20 = x["low"].shift(1).rolling(20, min_periods=5).min()

    out = pd.DataFrame({"timestamp": x["timestamp"]})
    out["f_ret1"] = np.log(x["close"] / x["close"].shift(1))
    out["f_ret3"] = np.log(x["close"] / x["close"].shift(3))
    out["f_ret12"] = np.log(x["close"] / x["close"].shift(12))
    out["f_range_atr"] = rng / atr
    out["f_body_range"] = body / rng
    out["f_upper_wick_range"] = upper / rng
    out["f_lower_wick_range"] = lower / rng
    out["f_close_ema20_atr"] = (x["close"] - ema20) / atr
    out["f_close_ema50_atr"] = (x["close"] - ema50) / atr
    out["f_close_ema200_atr"] = (x["close"] - ema200) / atr
    out["f_ema200_slope_atr"] = ema200.diff(5) / atr
    out["f_volume_ratio20"] = x["volume"] / vol_med
    out["f_break_high20"] = (x["close"] > hh20).astype(float)
    out["f_break_low20"] = (x["close"] < ll20).astype(float)
    out["f_atr_pct"] = atr / x["close"]
    return out.replace([np.inf, -np.inf], np.nan)


def build_causal_sequences(
    frame: pd.DataFrame,
    config: DeepRebuildConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex, list[str]]:
    """Past-only windows with execution-aligned next-bar target.

    Window ends at closed bar t. A hypothetical trade fills at open[t+1] and
    is marked to open[t+2]. The binary target is positive only when that gross
    move exceeds the configured round-trip friction hurdle.
    """
    cfg = config or DeepRebuildConfig()
    feat = causal_bar_features(frame)
    raw = frame.copy()
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True, errors="coerce")
    opens = pd.to_numeric(raw["open"], errors="coerce")
    gross = opens.shift(-2) / opens.shift(-1) - 1.0
    target = (gross > cfg.roundtrip_cost).astype(float)

    feature_cols = [c for c in feat.columns if c.startswith("f_")]
    fdf = feat[feature_cols].astype(float)
    mean = fdf.expanding(min_periods=30).mean().shift(1)
    std = fdf.expanding(min_periods=30).std(ddof=0).shift(1).replace(0, np.nan)
    norm = ((fdf - mean) / std).clip(-10, 10)

    xs, ys, rs, ts = [], [], [], []
    for i in range(cfg.lookback - 1, len(frame) - 2):
        window = norm.iloc[i - cfg.lookback + 1:i + 1].to_numpy(np.float32)
        g = float(gross.iloc[i])
        y = float(target.iloc[i])
        if not np.isfinite(window).all() or not np.isfinite(g):
            continue
        xs.append(window)
        ys.append(y)
        rs.append(g)
        ts.append(feat["timestamp"].iloc[i])
    return (
        np.asarray(xs, np.float32),
        np.asarray(ys, np.float32),
        np.asarray(rs, np.float32),
        pd.DatetimeIndex(ts),
        feature_cols,
    )


def chronological_split(
    x: np.ndarray,
    y: np.ndarray,
    gross: np.ndarray,
    ts: pd.DatetimeIndex,
    development: float = 0.60,
    validation: float = 0.20,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]]:
    n = len(x)
    if not (n == len(y) == len(gross) == len(ts)):
        raise ValueError("sequence arrays length mismatch")
    a = int(n * development)
    b = int(n * (development + validation))
    if a <= 0 or b <= a or b >= n:
        raise ValueError("invalid chronological split")
    return {
        "development": (x[:a], y[:a], gross[:a], ts[:a]),
        "validation": (x[a:b], y[a:b], gross[a:b], ts[a:b]),
        "test": (x[b:], y[b:], gross[b:], ts[b:]),
    }


if nn is not None:
    class RecurrentClassifier(nn.Module):
        def __init__(
            self,
            input_dim: int,
            kind: Literal["lstm", "gru"] = "lstm",
            hidden_dim: int = 64,
            layers: int = 2,
            dropout: float = 0.15,
        ):
            super().__init__()
            cls = nn.LSTM if kind == "lstm" else nn.GRU
            self.rnn = cls(
                input_dim, hidden_dim, num_layers=layers, batch_first=True,
                dropout=dropout if layers > 1 else 0.0,
            )
            self.norm = nn.LayerNorm(hidden_dim)
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1),
            )

        def forward(self, x):
            h, _ = self.rnn(x)
            return self.head(self.norm(h[:, -1])).squeeze(-1)


    class CNNLSTMClassifier(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.15):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(input_dim, hidden_dim, 5, padding=2), nn.GELU(),
                nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1), nn.GELU(),
            )
            self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
            self.head = nn.Sequential(
                nn.LayerNorm(hidden_dim), nn.Dropout(dropout), nn.Linear(hidden_dim, 1)
            )

        def forward(self, x):
            z = self.conv(x.transpose(1, 2)).transpose(1, 2)
            h, _ = self.lstm(z)
            return self.head(h[:, -1]).squeeze(-1)


    class TCNClassifier(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.15):
            super().__init__()
            blocks = []
            channels = input_dim
            for dilation in (1, 2, 4, 8):
                blocks.extend([
                    nn.Conv1d(channels, hidden_dim, 3, padding=dilation, dilation=dilation),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ])
                channels = hidden_dim
            self.net = nn.Sequential(*blocks)
            self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))

        def forward(self, x):
            z = self.net(x.transpose(1, 2))[:, :, -1]
            return self.head(z).squeeze(-1)


    class TransformerClassifier(nn.Module):
        def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 64,
            layers: int = 2,
            dropout: float = 0.15,
        ):
            super().__init__()
            self.proj = nn.Linear(input_dim, hidden_dim)
            enc = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim * 4,
                dropout=dropout, batch_first=True, norm_first=True, activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
            self.norm = nn.LayerNorm(hidden_dim)
            self.head = nn.Linear(hidden_dim, 1)

        def forward(self, x):
            z = self.encoder(self.proj(x))
            return self.head(self.norm(z[:, -1])).squeeze(-1)


def make_temporal_model(kind: str, input_dim: int, config: DeepRebuildConfig | None = None):
    require_torch()
    cfg = config or DeepRebuildConfig()
    if kind in {"lstm", "gru"}:
        return RecurrentClassifier(input_dim, kind=kind, hidden_dim=cfg.hidden_dim, layers=cfg.layers, dropout=cfg.dropout)
    if kind == "cnn_lstm":
        return CNNLSTMClassifier(input_dim, cfg.hidden_dim, cfg.dropout)
    if kind == "tcn":
        return TCNClassifier(input_dim, cfg.hidden_dim, cfg.dropout)
    if kind == "transformer":
        return TransformerClassifier(input_dim, cfg.hidden_dim, cfg.layers, cfg.dropout)
    raise ValueError(f"unknown temporal model: {kind}")


def fit_temporal_model(
    model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    config: DeepRebuildConfig | None = None,
    seed: int = 314,
):
    require_torch()
    cfg = config or DeepRebuildConfig()
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(min(4, max(1, torch.get_num_threads())))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train.astype(np.float32)))
    loader = DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    positives = max(float((y_train == 1).sum()), 1.0)
    negatives = max(float((y_train == 0).sum()), 1.0)
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(negatives / positives, dtype=torch.float32, device=device)
    )
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    xv = torch.from_numpy(x_val).to(device)
    yv = torch.from_numpy(y_val.astype(np.float32)).to(device)
    best_loss, best_state, stale, history = np.inf, None, 0, []
    for epoch in range(cfg.epochs):
        model.train()
        total, count = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach().cpu()) * len(xb)
            count += len(xb)
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(xv), yv).detach().cpu())
        history.append({
            "epoch": epoch + 1,
            "train_loss": total / max(count, 1),
            "validation_loss": val_loss,
        })
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= cfg.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model.cpu(), pd.DataFrame(history)


def predict_probability(model, x: np.ndarray, batch_size: int = 2048) -> np.ndarray:
    require_torch()
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            logits = model(torch.from_numpy(x[i:i + batch_size]))
            out.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(out).astype(np.float32) if out else np.empty(0, np.float32)


def economic_bar_metrics(
    probability: np.ndarray,
    gross_return: np.ndarray,
    threshold: float,
    config: DeepRebuildConfig | None = None,
) -> dict:
    cfg = config or DeepRebuildConfig()
    p = np.asarray(probability, float)
    r = np.asarray(gross_return, float)
    if len(p) != len(r):
        raise ValueError("probability/return length mismatch")
    lower = 1.0 - threshold
    position = np.where(p >= threshold, 1.0, np.where(p <= lower, -1.0, 0.0))
    turnover = np.abs(np.diff(np.r_[0.0, position]))
    net = position * r - turnover * cfg.one_way_cost
    equity = np.cumprod(1.0 + net)
    dd = equity / np.maximum.accumulate(equity) - 1.0 if len(equity) else np.empty(0)
    active = position != 0
    active_net = net[active]
    wins = active_net[active_net > 0].sum()
    losses = -active_net[active_net < 0].sum()
    pf = float(wins / losses) if losses > 0 else (np.inf if wins > 0 else np.nan)
    return {
        "bars": int(len(r)),
        "active_bars": int(active.sum()),
        "coverage": float(active.mean()) if len(active) else 0.0,
        "turnover_units": float(turnover.sum()),
        "mean_active_net_return": float(active_net.mean()) if len(active_net) else np.nan,
        "profit_factor": pf,
        "total_return": float(equity[-1] - 1.0) if len(equity) else 0.0,
        "max_drawdown": float(dd.min()) if len(dd) else 0.0,
    }


def choose_temporal_validation_threshold(
    probability: np.ndarray,
    gross_return: np.ndarray,
    config: DeepRebuildConfig | None = None,
) -> tuple[float, dict]:
    cfg = config or DeepRebuildConfig()
    best_threshold, best_metrics, best_objective = 0.99, {}, -np.inf
    for t in np.arange(0.55, 0.951, 0.025):
        metrics = economic_bar_metrics(probability, gross_return, float(t), cfg)
        if metrics["active_bars"] < cfg.min_active_validation:
            continue
        objective = (
            np.log1p(max(metrics["total_return"], -0.999999))
            - 1.5 * abs(min(0.0, metrics["max_drawdown"]))
            - 0.00002 * metrics["turnover_units"]
        )
        if objective > best_objective:
            best_objective = float(objective)
            best_threshold = float(t)
            best_metrics = metrics
    return best_threshold, {**best_metrics, "validation_objective": best_objective}
