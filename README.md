# Modular Crypto Trading Bot

Research software by Milad Parsanezhad for studying **Ichimoku + ML/DL/RL** on cryptocurrency OHLCV data.

**Status: v0.1.0 research implementation.** Training, walk-forward backtesting and checkpointed paper simulation work end to end. Validation in this workspace uses explicitly synthetic data. No historical profitability, live-exchange reliability or publication-ready result is claimed.

راهنمای فارسی: [شروع و اجرای ربات](docs/guide_fa.md) · [مرور علمی و طراحی](docs/research_review_fa.md) · [بررسی کدهای قبلی](docs/input_audit.md)

## Implemented

- BTC/USDT and ETH/USDT through separate experiments; 4h reference and 1h sensitivity profiles.
- Strict UTC/closed-candle validation, paginated public CCXT downloader, dataset checksums.
- Causal Ichimoku 9/26/52/26, RSI, ATR, MACD and price/volume features.
- Cash, passive buy-and-hold, risk-matched always-long and Ichimoku baselines.
- Random Forest, XGBoost, LSTM, CNN and a compact patch Transformer classifier.
- PPO using Stable-Baselines3 and the same execution/risk engine as the classifiers.
- LSTM/XGBoost agreement as a separate backtest strategy.
- Expanding walk-forward splits, a label-horizon purge, train-only scaling, validation-only threshold selection.
- With/without-Ichimoku ablations, fixed friction stress scenarios, regime diagnostics and paired block intervals.
- Next-open fills, two-sided fees/slippage, ATR stop/take, equity-based position sizing, drawdown/daily-loss circuit breakers.
- Paper account checkpointing, duplicate-candle protection, model/history binding and public-data polling.

These are independent implementations inspired by the literature, **not exact reproductions of Ghadiri's model, CLSTM-PPO, PatchTST, DTQL, Decision Transformer or DreamerV3**. The Transformer here predicts a binary label; it is not a world model or a trading language model.

## Install and test

Python 3.11+ (tested locally with Python 3.12; see run manifests for exact dependencies).

```bash
python -m venv .venv
```

Windows PowerShell: `.venv\Scripts\Activate.ps1`. Linux/macOS: `source .venv/bin/activate`.

```bash
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[research,test]"
python -m pytest -q
milad-bot demo --config configs/smoke.toml --output outputs/demo-001
```

The demo creates a synthetic fixture, trains all six models in two folds with both feature sets, and writes metrics, equity curves, fills, model bundles and manifests. Reusing an output directory is rejected to preserve previous experiments.

## Research on real candles

```bash
milad-bot fetch --exchange coinex --symbol BTC/USDT --timeframe 4h --since 2024-01-01T00:00:00Z --until 2026-01-01T00:00:00Z --output data/raw/coinex-btc-4h.csv
milad-bot research --config configs/research_4h.toml --data data/raw/coinex-btc-4h.csv --output outputs/btc-development-001
```

These commands require internet access to the exchange. The CoinEx download could not be verified here because DNS/network access failed. The adapter rejects unavailable start dates, incomplete ranges and candle gaps. Do not replace a failed historical download with synthetic candles under the same result label.

For ETH use `configs/research_eth_4h.toml` and download `ETH/USDT`. Use `configs/research_1h.toml` only with 1h input. The 1h profile preserves calendar lengths for lookback, label horizon and split sizes, but retains standard Ichimoku bar counts: this is a timeframe sensitivity experiment, not an identical indicator timescale.

## Forward paper simulation

Choose a model using development/validation evidence, freeze it, and start with a fresh simulated account. For example:

```bash
milad-bot paper-watch --bundle outputs/btc-development-001/fold2_seed42_random_forest_ichi.joblib --exchange coinex --state outputs/paper-btc/account.json --poll-seconds 60
```

No API key is needed. The first call commits an intention for the next candle; it does not invent historical fills. The next completed candle settles that intention using an assumed open fill and OHLC stops. This is forward **paper simulation**, not an exchange testnet or an authenticated order system. Missed candles stop processing for reconciliation. A lock avoids concurrent writers; after an interrupted process, inspect state before removing a stale `.lock` file. Full history is fetched to preserve exact feature initialization; this has overhead.

Only load model bundles you generated or otherwise trust: joblib and pickle formats can execute code when loaded.

## Research record

- [Scientific evidence and quality checks](docs/research_review_fa.md)
- [Reference registry and access levels](docs/references.json)
- [Experiment protocol and limitations](docs/protocol.md)
- [Architecture and timing](docs/architecture.md)
- [Validation evidence](reports/validation.md)
- [Roadmap and unresolved research gates](docs/roadmap.md)

The original `Gym-Trading-Env` fork and its [reliability PR](https://github.com/parsa314/Gym-Trading-Env/pull/1) remain separate. This repository uses its own broker/Gymnasium adapter for consistent next-open accounting and risk semantics.

No secret keys, original thesis documents, trained binary weights or exchange datasets are included in the public source tree. A redistribution license is intentionally not assigned automatically; see [provenance](docs/input_audit.md).
