# v0.22d Multimodal Fusion — Verified Results (2026-09-10)

## Evidence status

- GitHub Actions run: `34493123996`
- Artifact: `v22d-multimodal-fusion`
- Artifact ID: `10158775017`
- Artifact SHA-256: `7e0ba5cfdc6189fee54ff2f1c81b9270fa550453158a587f0eda6b390c037762`
- Workflow conclusion: `success`
- Scientific decision: `MULTIMODAL_REPRESENTATION_CANDIDATE`
- Representation gate passed: `true`
- Paper replacement authorized: `false`
- Live execution authorized: `false`
- Vision-to-RL state connected: `false`

This result is a representation/predictive-ablation result. It is not evidence of stable trading alpha and is not a live-trading authorization.

## Data and target

- Spot 4h: BTC/USDT, ETH/USDT, SOL/USDT.
- Development: 864 sampled decision events.
- Validation: 288 events.
- Untouched test: 288 events.
- Signal after `close[t]`; hypothetical fill at `open[t+1]`; exit at `open[t+2]`.
- Positive target only when the gross move exceeds the configured 24 bps round-trip cost hurdle.

## Primary ablation

| Arm | Validation AUC | Validation balanced accuracy | Test AUC | Test balanced accuracy | Test selected-event diagnostic |
|---|---:|---:|---:|---:|---:|
| Numeric core | 0.5637 | 0.5079 | 0.6158 | 0.5649 | +13.35% compounded sampled-event diagnostic |
| Numeric + supported liquidity | 0.5410 | 0.5393 | 0.6055 | 0.5845 | -8.67% diagnostic |
| Raw chart image | 0.5068 | 0.5000 | 0.4976 | 0.5000 | no events at p>=0.60 |
| Fusion: raw chart + supported numeric | **0.5768** | 0.5295 | **0.6388** | **0.6099** | +2.43% diagnostic |
| Fusion + course-only Wyckoff/ICT proxies | 0.5666 | 0.5363 | 0.6230 | 0.5676 | +0.90% diagnostic |

The selected-event returns above are diagnostics on sampled independent decision events, not an annualized portfolio backtest.

## Gate interpretation

Primary comparison: `fusion_supported` versus the stronger of `numeric_supported` and `image_raw`.

- Validation AUC gain: `+0.0358439201`.
- Test AUC regret: `-0.0332719318` (the fusion model outperformed the unimodal comparator on test rather than merely staying within the allowed regret).
- Minimum validation AUC gain gate: 0.015 — passed.
- Minimum test AUC gate: 0.53 — passed.
- Maximum test AUC regret gate: 0.005 — passed.
- Minimum test balanced-accuracy gate: 0.51 — passed.

Decision: `MULTIMODAL_REPRESENTATION_CANDIDATE`.

## Negative evidence retained

The scientifically supported liquidity feature family did **not** improve the numeric-only AUC in this run:

- validation AUC increment versus numeric core: `-0.0227364388`;
- test AUC increment versus numeric core: `-0.0102463735`.

The course-only Wyckoff/ICT proxy family also failed to add incremental value to the supported fusion model:

- validation increment: `-0.0101835047`;
- test increment: `-0.0157725075`;
- status: `NO_COURSE_PROXY_INCREMENTAL_EVIDENCE`.

Therefore course proxies are not eligible for strategy promotion.

## Consequence for v0.23

v0.22d justifies carrying the multimodal representation forward as a **candidate representation**, but it does not justify connecting it to reinforcement learning or replacing the current PAPER strategy. v0.23 therefore focuses on stricter machine-learning selection, probability calibration, economic backtesting, cost stress, risk sizing, external-venue replication, and a single frozen final-test decision. RL remains gated until these economic and prospective-evidence requirements are met.
