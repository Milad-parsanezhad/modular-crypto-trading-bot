# Validation record — 2026-09-06

**Synthetic software validation only. No real-market profitability result.**

Executed locally with Python 3.12.13:

```bash
python -m pytest -q
PYTHONPATH=src python -m milad_trader.cli demo --config configs/smoke.toml --output outputs/smoke-20260906
```

- Tests: **27 passed**, exit status 0 (5.74 seconds).
- End-to-end smoke run: completed, exit status 0.
- Six model families, two walk-forward folds, one seed, with/without Ichimoku: **24 model training/evaluation conditions**.
- Four ensemble evaluations and eight baseline evaluations bring the metrics table to 36 rows.
- 1,600 synthetic candles; 1,523 feature rows after historical warm-up.
- The manifest records exact package versions, configuration, data digest, source hashes and fold boundaries. Source hashes were checked against the delivered source.

Tests cover future-perturbation invariance, historical cloud alignment, unknown target tails, purged boundaries, invalid candle rejection, train-only scaling, equity/fee accounting, next-open fills, conservative stop/gap handling, risk latching, duplicate rejection, bootstrap mechanics, deep model serialization, the Gymnasium/SB3 interface, and paper checkpoint restart/history binding.

## Reviewable run evidence

- [Manifest](synthetic-smoke/manifest.json)
- [Metrics](synthetic-smoke/metrics.csv)

All returns in these files concern the synthetic fixture. They must not be used as thesis market-performance results. Trained binary weights and large generated outputs are intentionally excluded; the command above reproduces the run. This short run checks integration, not convergence or superiority of any model.

## Remaining validation

A read-only CoinEx historical-data attempt failed with a network/DNS error. No authenticated API or real order was sent. Public download reliability, real historical out-of-sample performance, multi-seed convergence, a frozen final holdout and extended forward paper operation remain unverified. GitHub Actions is configured for Python 3.11 and 3.12; local success does not imply hosted CI success.
