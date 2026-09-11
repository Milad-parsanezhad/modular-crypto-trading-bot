# MSc Defense Evidence Index

**Purpose:** provide a compact map from thesis/defense claims to immutable code, workflows and positive, negative, blocked and prospective evidence.

## 1. Strongest defensible claim today

The strongest defensible contribution is methodological and engineering-scientific:

> A modular cryptocurrency financial-ML research system was built that formulates strategy hypotheses, constructs causal/point-in-time features, models realistic friction, separates model selection from final evidence, audits leakage and model-search bias, distinguishes event selection from portfolio admission, enforces independent MTM/CVaR/correlation risk, preserves failed experiments, freezes model/artifact identity, and now collects genuinely prospective evidence through a blinded, hash-linked, schedule-only ledger.

The repository does **not** currently support a claim of guaranteed profitability, PAPER replacement or LIVE readiness.

## 2. Defense question → evidence map

| Likely defense question | Evidence to show |
|---|---|
| How did you prevent look-ahead bias? | closed-bar/next-open contracts, purged chronological splits, future-mutation tests, completed-HTF availability, `future_event_slice`, completed-4h boundary rule |
| How did you account for fees/slippage? | 24 bps base round trip; 36/60 bps stress; cost-aware execution and prospective portfolio stress |
| Did ML beat a simple strategy? | validation-only v0.25 HGB ranker beat frozen-score allocator locally, but spent OKX/KuCoin diagnostics did not confirm transfer; therefore no promotion |
| How did you address overfitting from many strategies/models? | compact model registries, multiple seeds, block bootstrap, FDR where applicable, PBO/DSR diagnostics, CPCV/search-aware gate, prospective first-look rule |
| Why not use only one train/test split? | regime/non-stationarity risk plus external and genuinely future-time replication requirements |
| Did deep learning help? | v0.24c LSTM/GRU/TCN/CNN-LSTM/Transformer validation tournament; challengers remained unpromoted because risk/fresh-evidence gates failed or were absent |
| How is portfolio risk different from event accuracy? | v0.24d: positive event-level economics on OKX/KuCoin but portfolio PF/DD/bootstrap failure; this directly motivated v0.25 ranking/arbitration |
| How is reproducibility guaranteed? | source SHA, exact persisted binaries, artifact hashes, dependency lock, `pip freeze`, exact archived replay, complete SHA-256 manifests and cross-artifact chain links |
| How was optional stopping controlled? | economics remain blinded until frozen maturity; first mature PASS/FAIL spends the sample; prospective workflow is schedule-only after pre-boundary validation |
| What if GitHub skips a scheduled collection? | frozen scheduler-continuity audit; post-boundary accepted-chain gap >5.5 h aborts before reading new evidence |
| Why is LIVE disabled? | fresh prospective evidence → CPCV/PBO/DSR → prospective PAPER → later readiness review are still required |

## 3. Key empirical milestones

### v0.17 — Ichimoku strategy lab — `TESTED`

S6 produced locally promising 2025 evidence on BTC/ETH under a frozen causal/cost contract; IRGC-S was rejected. The result was deliberately limited to research/forward observation.

### v0.18 — cost/regime experiments — `TESTED / NOT PROMOTED`

Cost/regime conditioning improved some comparisons, but confidence intervals crossed zero.

### v0.19/v0.20 — broad strategy discovery — `TESTED / EXTERNAL FAIL`

Thirty then forty-two multi-timeframe candidates were evaluated. Lower-timeframe candidates were especially friction-sensitive. v0.20's internal S6 winner retained positive OKX expectancy/PF but exceeded the frozen 5% drawdown ceiling; the rule was not relaxed.

### v0.24 — pooled strategy-aware meta-labeling — `REJECTED`

- 4,979 strategy events;
- 953 test events;
- frozen Ridge meta-filter selected 246 test events;
- mean-R/PF/win-rate improved locally;
- total return fell;
- breadth/bootstrap/PBO/DSR/36bps gates did not support promotion.

Decision: `NO_META_MODEL_PROMOTED`.

### v0.24b — family meta + overlap-aware portfolio — `REJECTED`

Workflow `34578494059`; artifact `10190676664`.

- base shadow portfolio: `+10.84%`, PF `1.3875`, realized DD `-3.70%`;
- family-filtered portfolio: `-2.33%`, PF `0.7957`, DD `-5.44%`;
- hard drawdown kill triggered.

Decision: `NO_FAMILY_META_PROMOTION`.

### v0.24c — temporal challengers / reproducibility defect — `CHALLENGER + BLOCKED`

Temporal workflow `34582340510`; artifact `10192231684`.

- S6: TCN, seed 2718, validation PF `1.0336`, MDD about `-11.50%`;
- D1-OB: LSTM, seed 1618, validation PF `1.4128`, MDD about `-8.58%`.

Neither passed the 5% risk ceiling. External validation was stopped when reconstructed D1-OB selection was `46` instead of frozen `47`; this was treated as a reproducibility defect rather than rounded away.

### v0.24d — exact frozen binary + dual-venue external triangulation — `TESTED / REJECTED`

Workflow `34585325516`; artifact `10193425439`.

Exact replay:

- S6: `118/236` exact;
- D1-OB: `47/185` exact.

External event-level economics were positive on both OKX and KuCoin, but overlap-aware MTM portfolios failed frozen PF/drawdown/bootstrap gates and triggered the hard-DD contract. Decision: `EXTERNAL_REPLICATION_FAILED`.

**Defense lesson:** event-level edge does not imply portfolio-level validity.

### v0.25 — risk-aware portfolio ranking — `CHALLENGER_NOT_PROMOTED`

Workflow `34590214474`; artifact `10195326409`; digest `sha256:3380bc537b551fd7f62ddbd37f250c9eaacb52d5e1dabd6799b39eac78d4a6bc`.

Validation-only frozen champion: `hgb_expected_r`.

- HGB accepted `69` vs frozen-score baseline `55`;
- return `+3.922%` vs `+1.874%`;
- PF `1.3427` vs `1.1948`;
- mean R `+0.2295R` vs `+0.1426R`;
- realized DD `-4.672%` vs `-5.201%`.

After freezing, the ranker was inspected on **already-spent** v0.24d data only as a transfer diagnostic. It underperformed the frozen-score baseline on both OKX and KuCoin. No retuning followed.

**Defense lesson:** validation improvement does not imply external generalization.

## 4. Current prospective experiment — v0.25 Future Evidence Ledger

Scientific state: `WAITING_FOR_FUTURE_BOUNDARY / BLINDED_NOT_STARTED`.

Frozen prospective boundary: `2026-09-11T16:00:00Z`.

Frozen collector: `fac456ccc0eb445ce7f2d8554840f37da9142ac7`.

Venues: OKX + KuCoin; timeframe: 4h; scheduler: `23 0,4,8,12,16,20 * * *` UTC.

The first mature economic read is prohibited until at least:

- `168 h` elapsed;
- `200` future events per venue;
- minimum usable symbol coverage.

Before maturity, economics are blinded. At maturity, the first PASS or FAIL is terminal and spends the prospective sample. The same sample cannot be extended and reread to rescue a failure.

The portfolio gate requires sufficient accepted trades, positive/incremental return over the frozen-score baseline, PF threshold, <=5% MTM drawdown, CVaR budget, no hard-DD kill, 36bps stress survival and a positive paired moving-block-bootstrap lower bound. Even a pass advances only to CPCV/PBO/DSR review, not PAPER replacement.

### Governance hardening before the clock

A final pre-boundary dry run `34611629491` succeeded. Artifact `10267919110`, digest `sha256:2dbf88c90c853cb3ada64e365dcb0861fbdae3dbaf2c9541a28f0beb49d4e121`.

Two failure modes were explicitly closed before the boundary:

1. **Manual first-look timing:** the production-like ledger is now schedule-only; push/manual dispatch are not normal evidence triggers.
2. **Scheduler discontinuity:** a post-boundary scheduled gap > `5.5 h` causes fail-closed abortion before new evidence is read.

Trigger governance and scheduler continuity records are SHA-256 linked into prospective chain-link v2.

## 5. Evidence labels for slides and Chapter 4

- **TESTED** — executed under a frozen protocol;
- **REJECTED** — a valid scientific gate failed;
- **BLOCKED** — infrastructure/data/reproducibility prevented a valid outcome read;
- **CHALLENGER** — promising local evidence but not eligible for promotion;
- **DATA_UNAVAILABLE** — required information is unavailable and is not fabricated;
- **FORWARD-PAPER CANDIDATE** — only after a fresh evidence gate explicitly authorizes it;
- **LIVE** — prohibited until the complete readiness ladder passes.

## 6. Files an examiner can inspect

- `README.md` — project contract/current state;
- `docs/PROJECT_STATUS_2026-09-11.md` — dated scientific status;
- `docs/RESEARCH_TRACEABILITY_MATRIX.md` — research-to-code map;
- `docs/V25_RANKING_RESULTS_2026-09-11.md` — validation ranker and spent transfer diagnostic;
- `docs/V25_PROSPECTIVE_LEDGER_ACTIVATION_2026-09-11.md` — future-evidence governance and chain identity;
- `research_bot/future_evidence_v25.py` — future boundary, maturity and portfolio gate;
- `scripts/run_v25_future_snapshot.py` — append-only/blinded collector and first-look executor;
- `.github/workflows/v25-prospective-evidence-ledger.yml` — schedule-only production-like evidence ledger;
- `tests/` — leakage/risk/provenance regression tests;
- `CITATION.cff` — academic citation metadata.

## 7. What should not be said in the defense

Do not present validation as untouched test evidence. Do not call a blocked experiment a strategy failure. Do not call a green CI run proof of alpha. Do not use previously spent OKX/KuCoin data as future evidence. Do not describe `hgb_expected_r` as externally validated. Do not claim PAPER replacement or LIVE readiness while their explicit authorization flags remain false.
