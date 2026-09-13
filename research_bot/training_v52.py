from __future__ import annotations

from dataclasses import dataclass
import copy
import random

import numpy as np

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


@dataclass(frozen=True)
class TrainingConfigV52:
    sequence_length: int = 48
    epochs: int = 50
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 8

    def __post_init__(self) -> None:
        if self.sequence_length < 2 or self.epochs < 1 or self.batch_size < 1 or self.learning_rate <= 0 or self.patience < 1:
            raise ValueError("invalid training configuration")


def require_torch_v52() -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required; install the v0.52 'deep' extra")


def set_seed_v52(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)


def make_sequences_v52(features: np.ndarray, targets: np.ndarray, *, sequence_length: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(targets, dtype=np.float32)
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("features must be 2D and targets aligned 1D")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("features/targets must be finite")
    if sequence_length < 2 or len(x) <= sequence_length:
        raise ValueError("insufficient rows for requested sequence length")
    windows = np.stack([x[i - sequence_length:i] for i in range(sequence_length, len(x))])
    labels = y[sequence_length:]
    return windows, labels


def fit_torch_regressor_v52(model, train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray, *, seed: int, config: TrainingConfigV52 | None = None, device: str | None = None):
    require_torch_v52()
    cfg = config or TrainingConfigV52()
    set_seed_v52(seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(dev)
    train_ds = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    vx = torch.from_numpy(val_x).to(dev)
    vy = torch.from_numpy(val_y).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    loss_fn = nn.HuberLoss()
    best_state = None
    best_val = float("inf")
    stale = 0
    history = []
    for epoch in range(cfg.epochs):
        model.train()
        losses = []
        for bx, by in loader:
            bx, by = bx.to(dev), by.to(dev)
            optimizer.zero_grad(set_to_none=True)
            pred = model(bx)
            loss = loss_fn(pred, by)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(vx), vy).detach().cpu())
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "val_loss": val_loss})
        if val_loss < best_val - 1e-8:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= cfg.patience:
                break
    if best_state is None:
        raise RuntimeError("training produced no valid checkpoint")
    model.load_state_dict(best_state)
    return model, {"best_val_loss": best_val, "epochs_ran": len(history), "history": history, "device": str(dev)}


def predict_torch_v52(model, x: np.ndarray, *, device: str | None = None) -> np.ndarray:
    require_torch_v52()
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(dev)
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(np.asarray(x, dtype=np.float32)).to(dev)).detach().cpu().numpy()
    pred = np.asarray(pred, dtype=float).reshape(-1)
    if not np.isfinite(pred).all():
        raise RuntimeError("non-finite model predictions")
    return pred
