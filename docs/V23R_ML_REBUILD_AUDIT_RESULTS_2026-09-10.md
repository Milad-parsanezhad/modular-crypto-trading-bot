# v0.23r Clean ML Rebuild + Audit Results — 2026-09-10

Baseline: v0.22d commit `2a5b62c84354b9cbffd35c0bdd6591d4219bf216`

Branch: `research/v23-ml-rebuild-audit2`

Workflow run: `34525227033`

## CI status

Both workflow jobs completed successfully:

- `deterministic-audit-tests`: SUCCESS
- `real-data-4h-smoke`: SUCCESS

Deterministic clean ML/audit suite: **20 passed**.

Relevant v0.22d regression suite: **11 passed, 2 warnings**. The two warnings are expected `All-NaN slice encountered` warnings from the robustness test for missing-value scaling; the test itself passes and verifies finite output handling.

## Real-data audit

The real-data experiment used closed CoinEx 4h spot bars for BTC/USDT, ETH/USDT and SOL/USDT under the frozen 24 bps round-trip cost contract.

Audit decision: **AUDIT_PASS**

- rows: 7,767
- columns: 23
- numeric model features: 13
- categorical context features: 2 (`timeframe`, `side`)
- positive-label fraction: 0.3680957899
- duplicate observation-key fraction (`symbol`, `signal_time`): 0.0
- expected repeated-timestamp fraction across the three-symbol panel: 0.6666666667
- worst feature missingness: 0.0092699884
- constant features: none
- fatal audit findings: none
- audit warnings: none
- dataframe SHA-256: `c27e4cc1038420c13ffa1e236aa7e7c47b02f86d3d4af1a31a3ee624b91bd1f8`
- audit-report SHA-256: `e7660f46e8bbe9ade6c06cfd487abe9b1041fb997670b5e080572814a9d26689`

## Bugs discovered and corrected during the rebuild

1. The first clean real-data smoke failed because an overly broad leakage-name rule treated the causal feature `f_realized_vol20` as an outcome merely because its name contained `realized_`. The deny-list was narrowed to genuinely post-trade realized-return/PnL/label/target fields while preserving causal realized-volatility features.
2. The first audit also treated repeated timestamps in a multi-asset panel as duplicates. This was methodologically incorrect because BTC, ETH and SOL are distinct observations at the same market timestamp. Duplicate detection now uses the observation key (`symbol`, `signal_time`) while repeated panel timestamps are reported separately as an informational diagnostic.
3. Regression tests were added for both bugs so they cannot silently return.

## Supervised smoke result

Seven model families were evaluated across three frozen seeds. Model-family selection used validation objective only; the test segment was read only after the validation champion was frozen.

Validation-selected champion: **Logistic Regression**, seed 314, frozen threshold `0.5221713792`.

Champion test result:

- selected observations: 214
- test AUC: 0.5567455
- test mean net return per selected observation: -0.00141423
- test profit factor: 0.7427094
- unit-exposure test total return: -27.4542%
- unit-exposure test maximum drawdown: -37.2663%

Decision: **NO_ML_MODEL_PROMOTED**.

This is a correct negative result: prediction discrimination above random on this test does not translate into positive post-cost economics.

### Important post-hoc observation

Extra Trees had a positive mean test net return (~+0.0003044) and mean test PF (~1.0563) across seeds, with mean test AUC ~0.5602. It was **not** the validation-selected champion. Therefore it is not promoted after observing the test result; doing so would contaminate the holdout. It may only motivate a newly pre-registered future experiment on a fresh period/venue.

## Unsupervised diagnostics

All three development-only unsupervised diagnostics executed successfully:

- Isolation Forest: 2 anomaly states
- GMM-3: 3 latent states
- GMM-5: 5 latent states

These states have `alpha_authorized = false`. They are regime/anomaly descriptors only until a separately frozen out-of-sample conditional-performance test demonstrates incremental value.

## Final v0.23r state

- ML data/audit plumbing: TESTED / PASS
- causal panel splitting + embargo + label-window purge: TESTED / PASS
- leakage/name guards: TESTED / PASS
- supervised 4h smoke alpha: REJECTED
- unsupervised regime discovery: DIAGNOSTIC ONLY
- v0.22d regression compatibility: PASS
- forward PAPER authorization: false
- PAPER replacement authorization: false
- LIVE execution authorization: false

The next research layer should build on this green audit foundation: strategy-event meta-labeling, development-only unsupervised regime features with OOS transform, expanded boosted-tree ablations, then a separately frozen deep temporal track (LSTM/GRU/TCN/Transformer). Vision/multimodal outputs remain separate representations and RL remains locked until upstream economic evidence is stable.
