"""Explicit, serializable experiment assumptions."""
from dataclasses import dataclass, field, asdict
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class RiskConfig:
    initial_cash: float = 10000.0
    fee: float = 0.001
    slippage_bps: float = 5.0
    allocation: float = 0.95
    risk_per_trade: float = 0.02
    stop_atr: float = 2.0
    take_atr: float = 4.0
    max_drawdown: float = 0.05
    daily_loss: float = 0.03

    def __post_init__(self):
        import math
        if not all(math.isfinite(v) for v in asdict(self).values()):
            raise ValueError("Risk parameters must be finite")
        if self.initial_cash <= 0 or not 0 <= self.fee < 0.05:
            raise ValueError("Invalid cash or fee")
        if not 0 <= self.slippage_bps < 1000:
            raise ValueError("Invalid slippage")
        for name in ("allocation", "risk_per_trade", "max_drawdown", "daily_loss"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        if self.stop_atr <= 0 or self.take_atr <= 0:
            raise ValueError("ATR multiples must be positive")


@dataclass(frozen=True)
class ExperimentConfig:
    symbol: str = "BTC/USDT"
    timeframe: str = "4h"
    lookback: int = 24
    horizon: int = 1
    train_bars: int = 1800
    validation_bars: int = 360
    test_bars: int = 360
    folds: int = 3
    seeds: list[int] = field(default_factory=lambda: [7, 42, 314])
    models: list[str] = field(default_factory=lambda: ["random_forest", "xgboost", "lstm", "ppo"])
    thresholds: list[float] = field(default_factory=lambda: [0.5, 0.55, 0.6])
    epochs: int = 20
    ppo_steps: int = 20000
    ablation: bool = True
    risk: RiskConfig = field(default_factory=RiskConfig)

    def __post_init__(self):
        if self.timeframe not in ("1h", "4h"):
            raise ValueError("Supported research timeframes: 1h, 4h")
        for n in ("lookback", "horizon", "train_bars", "validation_bars", "test_bars", "folds", "epochs", "ppo_steps"):
            if getattr(self, n) < 1:
                raise ValueError(f"{n} must be positive")
        if self.train_bars < self.lookback + 10:
            raise ValueError("Training segment too short")
        if not self.seeds or not self.thresholds or not all(0 < x < 1 for x in self.thresholds):
            raise ValueError("Specify seeds and probability thresholds in (0,1)")
        allowed = {"random_forest", "xgboost", "lstm", "cnn", "transformer", "ppo"}
        if not set(self.models) <= allowed:
            raise ValueError("Unknown model")

    @property
    def bars_per_year(self):
        return 365 * (24 if self.timeframe == "1h" else 6)

    def to_dict(self):
        return asdict(self)


def load_config(path):
    raw = tomllib.loads(Path(path).read_text())
    raw["risk"] = RiskConfig(**raw.get("risk", {}))
    return ExperimentConfig(**raw)
