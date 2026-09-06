# Colab Notebook Audit — 2026-09-06

All 11 notebooks in the connected `Colab Notebooks` folder were reviewed as historical research artifacts, not validated strategies.

## Lineage
- `Untitled1` → `Untitled2` → `Untitled4`/`Untitled6`: early Ichimoku + Q-learning/DQN.
- `Untitled8` → `Untitled9`: modular CoinEx + LSTM/attention/RL.
- `Untitled0`: large merged notebook with multiple duplicated generations.
- `Untitled10`: later modular rewrite with Random Forest, feature selection, risk/executor/vault concepts; much of the demonstrated market data/execution is synthetic.
- `Untitled3` and `Untitled5`: effectively empty.
- `Untitled7`: Word-document utility, unrelated to trading-model research.

## Critical findings
1. **Future leakage:** several notebooks create Chikou/lagging span with `close.shift(-26)` and include it as a current-row feature. This exposes future close values. Banned in the new research pipeline.
2. **Temporal leakage:** `Untitled10` uses `bfill()` after indicators. Backfill can move later information into earlier rows. Warm-up rows must be dropped instead.
3. **No proper OOS gate in the main Untitled10 pipeline:** feature selection and Random Forest are fitted on the aligned full sample, then the same sample is iterated for decisions.
4. **Synthetic performance is not evidence:** Untitled10 prints many BUY executions on randomly generated weekly prices near 10,000. Its demonstrated loop does not connect realized PnL into capital, so zero drawdown is not meaningful.
5. **Early RL environments are economically weak:** immediate next-close rewards, incomplete inventory/accounting, weak/no friction models, and optimization on the same sample. They will not be migrated.
6. Stored outputs show dependency/runtime failures across generations: TA-Lib build errors, missing packages, outdated Keras arguments, unavailable `PrioritizedReplayBuffer`, syntax errors, and interrupted infinite loops.

## Disposition
| Notebook | Decision |
|---|---|
| Untitled0 | Archive; mine ideas only |
| Untitled1 | Archive; keep orchestration ideas only |
| Untitled2 | Archive; do not reuse RL environment |
| Untitled3 | Ignore (empty) |
| Untitled4 | Archive; notification pattern only |
| Untitled5 | Ignore (empty) |
| Untitled6 | Ignore as near-duplicate of Untitled4 |
| Untitled7 | Separate non-trading utility |
| Untitled8 | Concept source only |
| Untitled9 | Selective migration: data retry/risk-state ideas |
| Untitled10 | Rewrite, do not patch in place |

## Keep/rewrite
Public CoinEx ingestion with retries; multi-symbol/timeframe configuration; risk-based position sizing; audit logging/state persistence later; notification hooks later; tree/LSTM/Transformer/RL only as challengers.

## Reject
Future-shift features; temporal backfill; scaler/feature selection on all data before split; synthetic performance claims; pseudo-RL/random actions presented as learning; endless live loops before paper-trading gates; notebook-local credentials.

## v0.2 gate
Point-in-time CUSUM event filtering, triple-barrier labeling with ambiguity/no-overlap controls, purged walk-forward validation, real OHLCV component tests, explicit abstention and round-trip friction assumptions. Next: CPCV, PBO/Deflated Sharpe, funding/carry, point-in-time on-chain, order flow/LOB, capacity/impact.
