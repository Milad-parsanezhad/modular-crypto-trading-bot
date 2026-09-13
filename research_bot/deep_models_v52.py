from __future__ import annotations

"""Optional deep sequence backends for v0.52 research.

The module imports without torch/mamba installed. Model construction fails closed
with a clear dependency error rather than silently substituting another model.
"""

from dataclasses import dataclass

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - dependency optional
    torch = None
    nn = None


@dataclass(frozen=True)
class DeepModelConfigV52:
    input_dim: int
    hidden_dim: int = 64
    layers: int = 2
    dropout: float = 0.1
    patch_len: int = 8
    d_model: int = 64
    nhead: int = 4

    def __post_init__(self) -> None:
        if self.input_dim <= 0 or self.hidden_dim <= 0 or self.layers <= 0:
            raise ValueError("model dimensions must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        if self.patch_len <= 0 or self.d_model <= 0 or self.nhead <= 0:
            raise ValueError("patch/transformer dimensions must be positive")
        if self.d_model % self.nhead:
            raise ValueError("d_model must be divisible by nhead")


def _require_torch() -> None:
    if torch is None or nn is None:
        raise RuntimeError("PyTorch is required for v0.52 deep models; install the 'deep' extra")


if nn is not None:
    class LSTMForecasterV52(nn.Module):
        def __init__(self, config: DeepModelConfigV52):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=config.input_dim,
                hidden_size=config.hidden_dim,
                num_layers=config.layers,
                dropout=config.dropout if config.layers > 1 else 0.0,
                batch_first=True,
            )
            self.head = nn.Linear(config.hidden_dim, 1)

        def forward(self, x):
            y, _ = self.lstm(x)
            return self.head(y[:, -1]).squeeze(-1)


    class PatchTSTStyleForecasterV52(nn.Module):
        """Compact PatchTST-style research baseline, not a claim of exact paper reproduction."""

        def __init__(self, config: DeepModelConfigV52):
            super().__init__()
            self.patch_len = config.patch_len
            self.patch_proj = nn.Linear(config.input_dim * config.patch_len, config.d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.nhead,
                dim_feedforward=config.d_model * 4,
                dropout=config.dropout,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.layers)
            self.norm = nn.LayerNorm(config.d_model)
            self.head = nn.Linear(config.d_model, 1)

        def forward(self, x):
            batch, steps, features = x.shape
            n_patches = steps // self.patch_len
            if n_patches < 1:
                raise ValueError("sequence shorter than patch_len")
            x = x[:, -n_patches * self.patch_len :, :]
            x = x.reshape(batch, n_patches, self.patch_len * features)
            z = self.patch_proj(x)
            z = self.encoder(z)
            z = self.norm(z.mean(dim=1))
            return self.head(z).squeeze(-1)
else:  # pragma: no cover
    LSTMForecasterV52 = None
    PatchTSTStyleForecasterV52 = None


def build_lstm_v52(config: DeepModelConfigV52):
    _require_torch()
    return LSTMForecasterV52(config)


def build_patchtst_v52(config: DeepModelConfigV52):
    _require_torch()
    return PatchTSTStyleForecasterV52(config)


def build_mamba_v52(config: DeepModelConfigV52):
    _require_torch()
    try:
        from mamba_ssm import Mamba2
    except Exception as exc:  # pragma: no cover - optional CUDA/native dependency
        raise RuntimeError("mamba-ssm is required for the real v0.52 Mamba backend; install the 'mamba' extra") from exc

    class MambaForecasterV52(nn.Module):
        def __init__(self):
            super().__init__()
            self.in_proj = nn.Linear(config.input_dim, config.d_model)
            self.blocks = nn.ModuleList([Mamba2(d_model=config.d_model, d_state=64, d_conv=4, expand=2) for _ in range(config.layers)])
            self.norm = nn.LayerNorm(config.d_model)
            self.head = nn.Linear(config.d_model, 1)

        def forward(self, x):
            z = self.in_proj(x)
            for block in self.blocks:
                z = z + block(z)
            z = self.norm(z[:, -1])
            return self.head(z).squeeze(-1)

    return MambaForecasterV52()
