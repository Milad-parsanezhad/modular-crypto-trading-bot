# Project Status — 2026-09-11

## Purpose

This document is the defense-oriented status map for the current MSc research program. It separates engineering success from scientific promotion and records negative/blocking evidence without rewriting history.

## Claim boundary

- `LIVE_EXECUTION = false`
- no guaranteed-profit claim is authorized
- a green GitHub Actions run means the frozen experiment executed according to its engineering contract
- model/strategy promotion requires the scientific gate, not CI success alone
- previously observed terminal/external data cannot be re-labeled as fresh untouched evidence after redesign

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
| v0.24c external fresh-evidence route | BLOCKED | reconstructed model identity differed by one validation selection; external outcome was not read |
| v0.24c temporal challengers | CHALLENGER | validation-only TCN/LSTM champions frozen; both exceeded 5% validation DD ceiling |
| v0.24d exact frozen-snapshot replay | TESTED / REPRODUCIBILITY PASS | exact persisted binaries reproduced 118/47 archived validation selections under the original environment |
| v0.24d OKX+KuCoin external triangulation | REJECTED | exact event filter improved event-level economics broadly, but portfolio PF/DD/bootstrap gates failed on both venues |

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

Frozen challengers:

### H4_S6_BREAKOUT

- model: `TCN`
- seed: `2718`
- threshold: `0.4991191626`
- validation selected: `95`
- validation mean R: `+0.025791R`
- validation PF: `1.033631`
- validation total return: `+0.523%`
- validation MDD: `-11.497%`

### H4_D1_OB_BOS_RISK

- model: `LSTM`
- seed: `1618`
- threshold: `0.4864120185`
- validation selected: `93`
- validation mean R: `+0.251931R`
- validation PF: `1.412762`
- validation total return: `+5.941%`
- validation MDD: `-8.584%`

Neither is promoted; the previously observed v0.24b terminal test was deliberately not scored.

## v0.24c reproducibility blocker

Primary and fallback external routes stopped before reading an external economic outcome because model reconstruction from algorithm/seed/threshold produced:

`FROZEN_THRESHOLD_REPRODUCTION_MISMATCH H4_D1_OB_BOS_RISK: got=46 expected=47`

Instead of relaxing the threshold, v0.24d switched scientific model identity to the exact persisted binary artifact.

## v0.24d exact-replay + external result

Workflow: `34585325516` — SUCCESS  
Artifact: `v24d-frozen-snapshot-external-34585325516`  
Artifact ID: `10193425439`  
Artifact digest: `sha256:e75423d67476172c78b24c03b5a253257fa0a76b94996fc0557e4d2b85fc8acb`

### Exact archived replay

- S6: `118/236` selected — exact PASS
- D1-OB: `47/185` selected — exact PASS

This resolves the v0.24c 46-vs-47 reproducibility blocker without refit or threshold retuning.

### OKX

- 1,868 external events / 638 frozen-selected
- event-level 24 bps PF `1.1817`, mean `+0.1220R`
- 36 bps PF `1.1306`; 60 bps PF `1.0370`
- symbol uplift breadth `10/11`
- MTM-filtered portfolio: 165 accepted, return `-0.861%`, PF `0.9621`
- max intrabar-stress DD `-5.335%`
- block-uplift CI lower bound below zero
- hard MTM kill triggered

### KuCoin

- 1,869 external events / 641 frozen-selected
- event-level 24 bps PF `1.1975`, mean `+0.1327R`
- 36 bps PF `1.1447`; 60 bps PF `1.0481`
- symbol uplift breadth `9/11`
- MTM-filtered portfolio: 156 accepted, return `+0.315%`, PF `1.0147`
- max intrabar-stress DD `-5.703%`
- block-uplift CI lower bound below zero
- hard MTM kill triggered

Both venues failed the frozen portfolio PF, MTM-DD and paired-bootstrap gates. Decision: **`EXTERNAL_REPLICATION_FAILED`**.

## New Plan-D discovery

The v0.24d result exposes a new architectural distinction:

> **event alpha / abstention and portfolio admission/arbitration are not the same problem.**

The frozen model improved event-level economics broadly on both venues, yet the overlap-aware portfolio did not preserve that edge. The current allocator uses the model score as a binary gate but, when simultaneous selected events compete for scarce risk budget, deterministic ordering is `entry_time → strategy → symbol` rather than economic priority. Portfolio/risk caps and the 5% kill therefore choose/truncate a subset that can differ materially from the broad positive event population.

This is a post-outcome diagnostic discovery, not a rescue of v0.24d. The OKX/KuCoin 2025–2026 sample is now development/diagnostic evidence for the redesigned hypothesis and cannot be called untouched again.

## Next evidence gate

Plan D now requires a **new portfolio-arbitration research cycle**. Before implementation, the next design is being grounded in current literature on:

- learning-to-rank for portfolio selection;
- conformal / uncertainty-aware abstention;
- survival analysis for trading-signal duration;
- score-to-net-R calibration;
- CVaR/correlation-aware marginal risk utility;
- RL only after simpler deterministic/ML allocator baselines survive fresh evidence.

The redesigned allocator must then be frozen and evaluated on genuinely future-time evidence. No same-sample retuning or promotion is allowed.

## Defense interpretation

The strongest contribution is not that increasing model complexity always produces higher returns. The defensible contribution is a falsifiable, reproducible architecture that distinguishes prediction from portfolio economics and stops promotion when external portfolio evidence contradicts encouraging event-level metrics.
