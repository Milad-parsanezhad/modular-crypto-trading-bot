# Project Status — 2026-09-11

## Purpose

This document is the defense-oriented status map for the current MSc research program. It separates engineering success from scientific promotion and records negative/blocking evidence without rewriting history.

## Claim boundary

- `LIVE_EXECUTION = false`
- no guaranteed-profit claim is authorized
- a green GitHub Actions run means the frozen experiment executed according to its engineering contract
- model/strategy promotion requires the scientific gate, not CI success alone
- previously observed terminal data cannot be re-labeled as fresh untouched evidence after redesign

## Evidence ladder

| Stage | Status | Scientific interpretation |
|---|---|---|
| v0.17 Ichimoku strategy lab | TESTED | S6 showed locally promising 2025 evidence, but remained research/forward-observation only |
| v0.18 cost/regime replication | TESTED / NOT PROMOTED | some cost-aware/regime improvements; confidence intervals did not justify promotion |
| v0.19 multi-timeframe tournament | TESTED / NOT PROMOTED | lower timeframes were heavily damaged by modeled friction; no robust candidate promoted |
| v0.20 42-candidate global-source tournament | TESTED / EXTERNAL FAIL | H4 candidates passed internal gates; external winner exceeded frozen 5% DD ceiling |
| v0.23r ML rebuild | TESTED | leakage/provenance/split contracts hardened; prediction layer remains research-only |
| v0.24 pooled strategy-aware meta-labeling | REJECTED | `NO_META_MODEL_PROMOTED` |
| v0.24b family-specific meta-labeling | REJECTED | `NO_FAMILY_META_PROMOTION`; filtered portfolio underperformed shadow base and hit DD kill |
| v0.24c external fresh-evidence route | BLOCKED | exact frozen-model reconstruction differed by one validation selection; external outcome was not read |
| v0.24c temporal challengers | CHALLENGER | validation-only champions frozen; terminal v0.24b test deliberately not scored |

## v0.24 result summary

Workflow: `34577448117`

Decision: `NO_META_MODEL_PROMOTED`.

Untouched-test summary recorded in PR #22:

- total strategy events: 4,979
- test events: 953
- frozen pooled champion: Ridge Classifier, seed 314, threshold `0.5187396`
- selected test events: 246
- mean R: `+0.03394R → +0.04438R`
- profit factor: `1.0482 → 1.0605`
- win rate: `34.94% → 36.99%`
- event-level MDD: approximately `-39.21% → -17.78%`
- total return: `+7.56% → +2.54%`
- symbol breadth: 7/12 positive uplift
- supported strategy breadth: 1/6
- paired bootstrap uplift interval crossed zero
- validation PBO: `0.65`
- DSR probability: `0.05098`
- 36 bps stress: negative

Interpretation: local classification/filtering metrics improved, but incremental portfolio/economic evidence was insufficient.

## v0.24b result summary

Workflow: `34578494059`

Artifact: `v24b-family-portfolio-34578494059`  
Artifact ID: `10190676664`  
Artifact digest: `sha256:0384391db85922309e7b67e0e0481f2cbf8795cf069ee48f146e36ba857525db`

Decision: `NO_FAMILY_META_PROMOTION`.

Recorded shadow findings:

- five strategy families were sample-eligible;
- Kumo Triangle and Supply/Demand were `DATA_INSUFFICIENT`;
- `H4_D1_OB_BOS_RISK` shadow base return `-0.69%` → filtered `+1.38%`, PF `1.2536`;
- `H4_S6_BREAKOUT` improved mean-R/PF but reduced total return because coverage fell;
- overlap-aware base portfolio: `+10.84%`, PF `1.3875`, realized DD `-3.70%`, 161 accepted trades;
- family-meta-filtered portfolio: `-2.33%`, PF `0.7957`, realized DD `-5.44%`, 60 accepted trades;
- hard drawdown kill triggered in the filtered portfolio.

The terminal v0.24b period was already observed and therefore remains SHADOW only.

## v0.24c temporal challenger result

Workflow: `34582340510` — SUCCESS  
Artifact: `v24c-temporal-challenger-34582340510`  
Artifact ID: `10192231684`  
Artifact digest: `sha256:f5eef115839c5c1cec3419ad4e35fe47ffd4674d099997abdd1278ae022b8cb4`

The run searched `LSTM`, `GRU`, `TCN`, `CNN-LSTM`, and `Transformer` across seeds `314`, `2718`, and `1618`, using development fit and validation-only selection.

Frozen challengers:

### H4_S6_BREAKOUT

- model: `TCN`
- seed: `2718`
- threshold: `0.4991191626`
- validation objective: `2.6366626924`
- validation selected: `95`
- validation mean R: `+0.025791R`
- validation PF: `1.033631`
- validation total return: `+0.523%`
- validation MDD: `-11.497%`

This is a **challenger**, not a promotion. The DD is already a material warning and fresh evidence is mandatory.

### H4_D1_OB_BOS_RISK

- model: `LSTM`
- seed: `1618`
- threshold: `0.4864120185`
- validation objective: `2.7853651507`
- validation selected: `93`
- validation mean R: `+0.251931R`
- validation PF: `1.412762`
- validation total return: `+5.941%`
- validation MDD: `-8.584%`

This is also a **challenger**, not a promotion. The v0.24b terminal test was deliberately not scored.

## v0.24c external-validation blocker

Primary and OKX fallback external routes both stopped before reading an external economic outcome because exact frozen-model reproduction failed:

`FROZEN_THRESHOLD_REPRODUCTION_MISMATCH H4_D1_OB_BOS_RISK: got=46 expected=47`

This one-event difference is treated as a reproducibility defect. It is not rounded away and is not used as a reason to retune the model.

A separate important reproducibility observation is that the v0.24b serialized champion bundle was created under:

- `scikit-learn==1.9.1`
- `numpy==2.5.3`
- `pandas==3.0.5`
- `joblib==1.6.0`

The frozen model artifact file `family_validation_frozen_champions.joblib` has SHA-256:

`ec4a81b7d708b0ffc7a80238668fa1bb34cccdac0408c51e4aff082075a5a22a`

The next scientific route must consume this exact serialized snapshot under an environment compatible with its persisted estimator state instead of recreating the winner from model name/seed alone.

## Next evidence gate

The next stage is pre-registered as:

1. recover the exact v0.24b frozen serialized model artifact by immutable run/artifact identity and verify its SHA-256;
2. load it under the frozen dependency environment;
3. verify feature schema and frozen thresholds without refit;
4. evaluate on a disjoint external cross-sectional universe or genuinely future-time bars;
5. run mark-to-market portfolio simulation with correlation/CVaR/open-risk controls;
6. run 24/36/60 bps cost stress;
7. evaluate paired block-bootstrap uplift, breadth and sample sufficiency;
8. run CPCV/PBO/DSR/search-aware audit before any PAPER promotion;
9. keep LIVE fail-closed.

## Defense interpretation

The scientifically important contribution is not that every increasingly complex model becomes profitable. The defensible contribution is a reproducible pipeline that detects leakage, overfitting, cost fragility, regime dependence, portfolio-risk failure and reproducibility defects before allowing promotion.
