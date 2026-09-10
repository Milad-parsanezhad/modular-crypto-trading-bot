from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover - optional dependency
    torch = None
    nn = None
    DataLoader = TensorDataset = None


@dataclass(frozen=True)
class TemporalConfig:
    lookback: int = 64
    horizon: int = 1
    hidden_dim: int = 64
    layers: int = 2
    dropout: float = 0.15
    batch_size: int = 256
    epochs: int = 12
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 3
    fee_bps_each_way: float = 10.0
    slippage_bps_each_way: float = 2.0


def require_torch() -> None:
    if torch is None:
        raise ImportError("PyTorch is optional. Install project extra: pip install -e '.[deep]'")


def causal_bar_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Causal numeric features from OHLCV; every row uses data at or before t."""
    x = frame.copy()
    x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    prev = x["close"].shift(1)
    tr = pd.concat([(x["high"] - x["low"]), (x["high"] - prev).abs(), (x["low"] - prev).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean().replace(0, np.nan)
    body = x["close"] - x["open"]
    rng = (x["high"] - x["low"]).replace(0, np.nan)
    upper = x["high"] - np.maximum(x["open"], x["close"])
    lower = np.minimum(x["open"], x["close"]) - x["low"]
    ema20 = x["close"].ewm(span=20, adjust=False).mean()
    ema50 = x["close"].ewm(span=50, adjust=False).mean()
    ema200 = x["close"].ewm(span=200, adjust=False).mean()
    vol_med = x["volume"].shift(1).rolling(20, min_periods=5).median().replace(0, np.nan)
    hh20 = x["high"].shift(1).rolling(20, min_periods=5).max()
    ll20 = x["low"].shift(1).rolling(20, min_periods=5).min()

    out = pd.DataFrame({"timestamp": x["timestamp"]})
    out["ret1"] = np.log(x["close"] / x["close"].shift(1))
    out["ret3"] = np.log(x["close"] / x["close"].shift(3))
    out["ret12"] = np.log(x["close"] / x["close"].shift(12))
    out["range_atr"] = rng / atr
    out["body_range"] = body / rng
    out["upper_wick_range"] = upper / rng
    out["lower_wick_range"] = lower / rng
    out["close_ema20_atr"] = (x["close"] - ema20) / atr
    out["close_ema50_atr"] = (x["close"] - ema50) / atr
    out["close_ema200_atr"] = (x["close"] - ema200) / atr
    out["ema200_slope_atr"] = ema200.diff(5) / atr
    out["volume_ratio20"] = x["volume"] / vol_med
    out["break_high20"] = (x["close"] > hh20).astype(float)
    out["break_low20"] = (x["close"] < ll20).astype(float)
    out["atr_pct"] = atr / x["close"]
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def build_sequences(frame: pd.DataFrame, config: TemporalConfig | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex, list[str]]:
    cfg = config or TemporalConfig()
    feat = causal_bar_features(frame)
    raw = frame.copy()
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
    future_return = np.log(raw["close"].shift(-cfg.horizon) / raw["close"])
    feature_cols = [c for c in feat.columns if c != "timestamp"]
    matrix = feat[feature_cols].to_numpy(dtype=np.float32)
    # Expanding-only standardization: statistics for row t use rows <= t-1.
    fdf = feat[feature_cols].astype(float)
    mean = fdf.expanding(min_periods=30).mean().shift(1)
    std = fdf.expanding(min_periods=30).std(ddof=0).shift(1).replace(0, np.nan)
    matrix = ((fdf - mean) / std).clip(-10, 10).to_numpy(dtype=np.float32)
    ret = future_return.to_numpy(dtype=np.float32)
    xs, ys, rs, ts = [], [], [], []
    for i in range(cfg.lookback - 1, len(frame) - cfg.horizon):
        w = matrix[i - cfg.lookback + 1:i + 1]
        r = ret[i]
        if not np.isfinite(w).all() or not np.isfinite(r):
            continue
        xs.append(w)
        ys.append(float(r > 0))
        rs.append(float(r))
        ts.append(feat["timestamp"].iloc[i])
    return np.asarray(xs, np.float32), np.asarray(ys, np.float32), np.asarray(rs, np.float32), pd.DatetimeIndex(ts), feature_cols


if nn is not None:
    class RecurrentClassifier(nn.Module):
        def __init__(self, input_dim: int, kind: Literal["lstm", "gru"] = "lstm", hidden_dim: int = 64, layers: int = 2, dropout: float = 0.15):
            super().__init__()
            cls = nn.LSTM if kind == "lstm" else nn.GRU
            self.rnn = cls(input_dim, hidden_dim, num_layers=layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
            self.norm = nn.LayerNorm(hidden_dim)
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, 1))
        def forward(self, x):
            h, _ = self.rnn(x)
            return self.head(self.norm(h[:, -1])).squeeze(-1)


    class CNNLSTMClassifier(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.15):
            super().__init__()
            self.conv = nn.Sequential(nn.Conv1d(input_dim, hidden_dim, 5, padding=2), nn.GELU(), nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1), nn.GELU())
            self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
            self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
        def forward(self, x):
            z = self.conv(x.transpose(1, 2)).transpose(1, 2)
            h, _ = self.lstm(z)
            return self.head(h[:, -1]).squeeze(-1)


    class TCNClassifier(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.15):
            super().__init__()
            blocks = []
            c = input_dim
            for dilation in [1, 2, 4, 8]:
                blocks += [nn.Conv1d(c, hidden_dim, 3, padding=dilation, dilation=dilation), nn.GELU(), nn.Dropout(dropout)]
                c = hidden_dim
            self.net = nn.Sequential(*blocks)
            self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))
        def forward(self, x):
            z = self.net(x.transpose(1, 2))[:, :, -1]
            return self.head(z).squeeze(-1)


    class TransformerClassifier(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int = 64, layers: int = 2, dropout: float = 0.15):
            super().__init__()
            self.proj = nn.Linear(input_dim, hidden_dim)
            enc = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim * 4, dropout=dropout, batch_first=True, norm_first=True, activation="gelu")
            self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
            self.norm = nn.LayerNorm(hidden_dim)
            self.head = nn.Linear(hidden_dim, 1)
        def forward(self, x):
            z = self.encoder(self.proj(x))
            return self.head(self.norm(z[:, -1])).squeeze(-1)


def make_model(kind: str, input_dim: int, cfg: TemporalConfig | None = None):
    require_torch()
    c = cfg or TemporalConfig()
    if kind in {"lstm", "gru"}:
        return RecurrentClassifier(input_dim, kind=kind, hidden_dim=c.hidden_dim, layers=c.layers, dropout=c.dropout)
    if kind == "cnn_lstm":
        return CNNLSTMClassifier(input_dim, c.hidden_dim, c.dropout)
    if kind == "tcn":
        return TCNClassifier(input_dim, c.hidden_dim, c.dropout)
    if kind == "transformer":
        return TransformerClassifier(input_dim, c.hidden_dim, c.layers, c.dropout)
    raise ValueError(f"unknown temporal model: {kind}")


def fit_binary_model(model, x_train, y_train, x_val, y_val, cfg: TemporalConfig | None = None, seed: int = 314):
    require_torch()
    c = cfg or TemporalConfig()
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    train = TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
    loader = DataLoader(train, batch_size=c.batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
    pos = max(1.0, float((y_train == 0).sum() / max(1, (y_train == 1).sum())))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=c.learning_rate, weight_decay=c.weight_decay)
    best, best_state, stale, history = np.inf, None, 0, []
    xv = torch.tensor(x_val, device=device); yv = torch.tensor(y_val, device=device)
    for epoch in range(c.epochs):
        model.train(); train_loss = 0.0; n = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            train_loss += float(loss.detach()) * len(xb); n += len(xb)
        model.eval()
        with torch.no_grad(): val_loss = float(loss_fn(model(xv), yv).cpu())
        history.append({"epoch": epoch + 1, "train_loss": train_loss / max(1, n), "validation_loss": val_loss})
        if val_loss < best - 1e-5:
            best, stale = val_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= c.patience: break
    if best_state is not None: model.load_state_dict(best_state)
    return model.cpu(), pd.DataFrame(history)


def predict_probability(model, x: np.ndarray, batch_size: int = 2048) -> np.ndarray:
    require_torch(); model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            logits = model(torch.tensor(x[i:i + batch_size]))
            out.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(out) if out else np.empty(0)


def economic_backtest(prob: np.ndarray, future_return: np.ndarray, upper: float, lower: float | None = None, cost_bps_each_way: float = 12.0) -> dict:
    lower = 1.0 - upper if lower is None else lower
    pos = np.where(prob >= upper, 1.0, np.where(prob <= lower, -1.0, 0.0))
    turnover = np.abs(np.diff(np.r_[0.0, pos]))
    net = pos * future_return - turnover * (cost_bps_each_way / 10000.0)
    equity = np.exp(np.cumsum(net))
    dd = equity / np.maximum.accumulate(equity) - 1.0
    active = pos != 0
    gross_p = (pos * future_return)[active]
    net_p = net[active]
    wins = net_p[net_p > 0].sum(); losses = -net_p[net_p < 0].sum()
    pf = float(wins / losses) if losses > 0 else (np.inf if wins > 0 else np.nan)
    return {
        "bars": int(len(net)), "active_bars": int(active.sum()), "coverage": float(active.mean()) if len(active) else 0.0,
        "turnover_units": float(turnover.sum()), "mean_active_net_return": float(net_p.mean()) if len(net_p) else np.nan,
        "profit_factor": pf, "total_log_return": float(net.sum()), "total_return": float(equity[-1] - 1.0) if len(equity) else 0.0,
        "max_drawdown": float(dd.min()) if len(dd) else 0.0,
        "direction_accuracy_active": float(((gross_p > 0)).mean()) if len(gross_p) else np.nan,
    }


def choose_validation_threshold(prob: np.ndarray, future_return: np.ndarray, min_active: int = 200, cost_bps_each_way: float = 12.0) -> tuple[float, dict]:
    best_t, best_m, best = 0.99, {}, -np.inf
    for t in np.arange(0.55, 0.951, 0.025):
        m = economic_backtest(prob, future_return, float(t), cost_bps_each_way=cost_bps_each_way)
        if m["active_bars"] < min_active: continue
        score = m["total_log_return"] - 1.5 * abs(min(0.0, m["max_drawdown"])) - 0.00002 * m["turnover_units"]
        if score > best:
            best, best_t, best_m = score, float(t), m
    return best_t, {**best_m, "validation_objective": float(best)}
