# MSc Defense Evidence Index

**Purpose:** provide a compact map from thesis/defense claims to immutable code, workflows and negative/positive empirical evidence.

## 1. What the project can defend today

The strongest defensible claim is methodological and engineering-scientific:

> A modular cryptocurrency financial-ML research system was built that can formulate strategy hypotheses, construct causal/point-in-time features, model realistic costs, separate model selection from final evidence, detect leakage/overfitting/reproducibility defects, enforce independent portfolio risk, preserve negative results, and generate reproducible CI artifacts for later forward validation.

The repository does **not** currently support a claim of guaranteed profitability or LIVE readiness.

## 2. Defense question → evidence map

| Likely defense question | Evidence to show |
|---|---|
| How did you prevent look-ahead bias? | closed-bar/next-open contracts, purged chronological splits, future-mutation tests, completed-HTF availability, v0.23r/v0.24c anti-leak tests |
| How did you account for fees/slippage? | baseline 24 bps round trip plus 36/60 bps stress, cost-aware execution modules and v0.18-v0.24 experiments |
| Did ML beat a simple strategy? | v0.24 pooled meta result shows local metric improvement but failed incremental breadth/bootstrap/PBO/DSR/cost gates; no false promotion |
| How did you address overfitting from many strategies/models? | trial registry, multiple seeds, block bootstrap, FDR, PBO/DSR diagnostics, CPCV planned/frozen gate |
| Why not use only one train/test split? | non-stationarity/regime evidence and search-aware validation ladder; future-time/external replication required |
| Did deep learning help? | v0.24c LSTM/GRU/TCN/CNN-LSTM/Transformer validation tournament; two frozen challengers, neither promoted because fresh evidence/risk gate absent |
| How is portfolio risk different from event accuracy? | overlap-aware realized portfolio in v0.24b; MTM/correlation/CVaR engine in v0.24c/v0.24d |
| What happened when experiments failed? | v0.12, v0.18, v0.19, v0.20, v0.24, v0.24b negative/borderline evidence is retained and explicitly labeled |
| How is reproducibility guaranteed? | source SHA, data/artifact hashes, `pip freeze`, immutable workflow artifacts, v0.24d exact frozen-binary replay |
| Why is LIVE disabled? | independent evidence contract requires fresh replication → PAPER → testnet → explicit live-readiness audit |

## 3. Key empirical milestones to cite

### v0.17 — Ichimoku strategy lab

S6 produced promising local 2025 evidence on BTC/ETH under the frozen causal/cost contract, while IRGC-S was rejected. The result was deliberately limited to research/forward observation.

### v0.18 — cost/regime experiments

Cost-aware/regime conditioning improved some comparisons, but the bootstrap confidence intervals crossed zero. No promotion.

### v0.19/v0.20 — broad strategy discovery

Thirty then forty-two multi-timeframe candidates were evaluated under realistic friction and risk gates. Lower-timeframe strategies were especially cost-fragile. In v0.20, the internal S6 winner replicated on OKX with positive expectancy/PF but breached the frozen 5% drawdown ceiling; the gate was not relaxed.

### v0.24 — pooled strategy-aware meta-labeling

- 4,979 strategy events;
- 953 test events;
- frozen Ridge meta-filter selected 246 test events;
- some mean-R/PF/win-rate metrics improved;
- total return fell;
- breadth/bootstrap/PBO/DSR/36bps stress did not support promotion.

Decision: `NO_META_MODEL_PROMOTED`.

### v0.24b — family meta + overlap-aware portfolio

Workflow `34578494059`; artifact `10190676664`.

- base shadow portfolio: `+10.84%`, PF `1.3875`, realized DD `-3.70%`;
- family-filtered portfolio: `-2.33%`, PF `0.7957`, DD `-5.44%`;
- hard drawdown kill triggered.

Decision: `NO_FAMILY_META_PROMOTION`.

### v0.24c — temporal challengers + reproducibility blocker

Temporal workflow `34582340510`; artifact `10192231684`.

- S6 frozen challenger: TCN / seed 2718 / PF `1.0336` / validation MDD about `-11.50%`;
- D1-OB frozen challenger: LSTM / seed 1618 / PF `1.4128` / validation MDD about `-8.58%`.

Both remain challengers because validation DD exceeded the 5% ceiling and no fresh terminal evidence was consumed.

External validation was stopped before outcome inspection when reconstructed D1-OB validation selection differed by one event (`46` vs frozen `47`). This became a new reproducibility research question instead of being ignored.

## 4. Current next experiment

`v0.24d — Frozen Snapshot External Triangulation`

Scientific novelty of this stage:

- the exact persisted v0.24b model binary is the model identity;
- artifact files are verified by SHA-256 before loading;
- the original persistence environment is pinned;
- archived validation must reproduce exact selections;
- the model is then transported without refit to a fixed disjoint external symbol universe on pre-registered venues;
- MTM, correlation, CVaR, exposure, drawdown and cost-stress gates are applied;
- even a positive external result still requires genuinely future-time validation before PAPER promotion.

## 5. Evidence labels for slides and Chapter 4

Use these labels consistently:

- **TESTED** — executed under a frozen protocol;
- **REJECTED** — a valid scientific gate failed;
- **BLOCKED** — infrastructure/data/reproducibility prevented a valid outcome read;
- **CHALLENGER** — promising local validation evidence but not eligible for promotion;
- **DATA_UNAVAILABLE** — required information is unavailable and is not fabricated;
- **FORWARD-PAPER CANDIDATE** — only after a fresh evidence gate explicitly authorizes it;
- **LIVE** — prohibited until the complete readiness ladder passes.

## 6. Files an examiner can inspect

- `README.md` — project contract/current state
- `docs/RESEARCH_TRACEABILITY_MATRIX.md` — research-to-code map
- `docs/PROJECT_STATUS_2026-09-11.md` — current evidence snapshot
- `docs/V17_ICHIMOKU_STRATEGY_LAB_PROTOCOL.md` — strategy-lab causal contract
- dated v0.17–v0.24 result documents — immutable result narrative
- `research_bot/` — implementation
- `tests/` — leakage/risk/provenance regression tests
- `.github/workflows/` — executable evidence pipelines
- `CITATION.cff` — academic citation metadata

## 7. What should not be said in the defense

Do not present a validation metric as an untouched test result; do not call a blocked external experiment a strategy failure; do not call a green workflow proof of alpha; do not cite conceptual/earlier placeholder performance as empirical thesis evidence; and do not claim LIVE readiness while the explicit evidence contract remains false.
