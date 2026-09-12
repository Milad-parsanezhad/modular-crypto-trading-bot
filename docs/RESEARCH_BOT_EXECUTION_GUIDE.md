# Modular Crypto Research Bot — Execution Guide

This repository is a fail-closed academic research system. It can build causal strategy features, run frozen development characterizations, report manifests/status and test governance. It does **not** expose an order-placement, PAPER or LIVE command.

## Install

```bash
python -m pip install -e .
```

## Health / governance check

```bash
modular-crypto-bot doctor
```

Expected safety state:

- mode: RESEARCH_ONLY
- Kraken: SEALED
- PAPER: false
- LIVE: false

## Inspect a frozen research manifest

```bash
modular-crypto-bot manifest --version v41
```

Available versions: `v39`, `v40`, `v41`.

## Current status

```bash
modular-crypto-bot status
```

## Build causal mother-strategy features from OHLCV CSV

Input columns must include at least timestamp/open/high/low/close/volume.

```bash
modular-crypto-bot features \
  --input data/btc_4h.csv \
  --output results/btc_4h_mother_features.csv
```

## Run the frozen v0.41 development characterization

```bash
modular-crypto-bot characterize \
  --version v41 \
  --output-dir results/v41_characterization
```

The characterization fetches only the consumed development venues CoinEx, OKX and KuCoin. Kraken is never instantiated by the frozen v0.39-v0.41 development runners.

## Scientific workflow

The executable path is:

`OHLCV -> causal v0.39 mother features -> semantic event family -> v0.41 cause-specific target/stop hazards -> cumulative incidence -> timeout-value head -> expected post-cost R -> conformal lower bound -> non-overlap realization -> financial risk governor -> frozen robustness gates`

A technically successful run does not imply profitable alpha. Scientific promotion requires every preregistered development gate to pass before the sealed external holdout can even be considered.

## Risk contract

The financial governor remains independent of the predictive model:

- base stop risk/trade: 0.25% equity;
- max stop risk/trade: 0.50%;
- aggregate open stop risk: 2.00%;
- same-direction risk: 1.50%;
- max asset nominal weight: 35%;
- max gross: 70%;
- drawdown throttles: 2% and 3.5%;
- hard firewall: 5%;
- cooldown after three consecutive losses;
- no martingale, averaging down or discretionary override.

## Why there is no LIVE command

LIVE execution is not a software-completeness criterion for this thesis. It is a promotion state that remains disabled until development robustness and a separately authorized untouched holdout succeed. The absence of a LIVE/order command is intentional fail-closed design, not an unfinished feature.
