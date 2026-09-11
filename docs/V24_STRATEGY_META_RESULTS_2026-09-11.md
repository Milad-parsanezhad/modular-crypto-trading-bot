# v0.24 Strategy-Aware Meta-Labeling Results — 2026-09-11

Workflow run: `34577448117`

Source commit: `ec90ec5fd2adcf5216db974b4732128f4b2d0b80`

Artifact: `v24-strategy-meta-34577448117` (artifact ID `10190273190`)

Artifact SHA-256: `42239a39ac4fa6cff5afd6338f47171a77565035e113d6bbbe8b468e8dcc6d64`

## Frozen decision

`NO_META_MODEL_PROMOTED`

The v0.24 strategy-aware ML layer improved some per-trade quality statistics, but it did not establish robust incremental economic value over the frozen base strategy pool. Forward PAPER replacement and LIVE execution remain unauthorized.

## Dataset audit

- Audit: `AUDIT_PASS`
- Strategy-event rows: **4,979**
- Frozen strategies: **7**
- Symbols: **12**
- Positive meta-label fraction: **32.07%**
- Duplicate strategy/symbol/signal events: **0%**
- Temporal event ordering: PASS
- Purged panel split: PASS
- Feature whitelist: PASS
- Worst model-feature missingness: **2.47%**
- Development events: **3,036**
- Validation events: **990**
- Untouched test events: **953**
- Dataset SHA-256: `6e942c8d5c38ca076b9457beb7d99c9126ca4ec37d6ee41f68a6456ceb90937e`

The seven frozen 4h strategies were `H4_S6_BREAKOUT`, `H4_KUMO_TRIANGLE`, `H4_OB_BOS_RETEST`, `H4_SUPPLY_DEMAND`, `H4_CORRELATION_DIVERGENCE`, `H4_D1_S6_VOL_RISK`, and `H4_D1_OB_BOS_RISK`.

`H4_KUMO_TRIANGLE` generated only three events in the fetched panel (ADA, BCH and DOT each contributed one; the other symbols generated zero). It is therefore considered `DATA_INSUFFICIENT` for strategy-level inference in this experiment and did not appear as a supported strategy group in the untouched-test breadth statistic.

## Frozen ML champion

- Model family: **Ridge Classifier**
- Seed: **314**
- Frozen score threshold: **0.5187395995**
- Model fit: development only
- Family/seed/threshold selection: validation only
- Test used for selection/refit: **false**
- Search configurations represented in multiple-testing diagnostics: **189**

Strategy and symbol identity were excluded from the primary model input. The model received causal `f_*` market/structure/context features plus `timeframe` and `side`, including the new causal CUSUM state and frozen strategy-parameter descriptors.

## Untouched test: base vs ML abstention filter

| Metric | Frozen base strategy events | Base + ML filter |
|---|---:|---:|
| Events selected | 953 | 246 |
| Coverage | 100.00% | 25.81% |
| Mean R | +0.03394R | **+0.04438R** |
| Profit factor | 1.0482 | **1.0605** |
| Win rate | 34.94% | **36.99%** |
| Compound return under the fixed event-level research sizing | **+7.56%** | +2.54% |
| Maximum drawdown under the same event-level accounting | -39.21% | **-17.78%** |
| Mean account return per selected event | +0.00849% | **+0.01109%** |

The ML filter therefore improved local trade quality and reduced the event-level drawdown substantially, but it rejected roughly three quarters of all events. The remaining sample did not generate greater total compounded return than the unfiltered pool and still violated the pre-registered 5% maximum-drawdown gate.

The drawdown figures above are from the experiment's simple event-level sequential compounding and are **not** a portfolio-overlap-aware execution backtest. A dedicated portfolio concurrency/correlation risk simulation is required before risk conclusions can be promoted.

## Breadth

- Symbols with positive total-return uplift: **7 / 12 = 58.33%**; gate = 60%, FAIL.
- Supported test strategy groups with positive uplift: **1 / 6 = 16.67%**; gate = 50%, FAIL.

The only strategy with positive total-return uplift was `H4_CORRELATION_DIVERGENCE`. `H4_S6_BREAKOUT` showed materially better per-trade quality after filtering but lower total return because of the lower event count.

### Strategy-level untouched-test ablation

| Strategy | Base mean R | Filtered mean R | Base PF | Filtered PF | Base return | Filtered return | Total-return uplift |
|---|---:|---:|---:|---:|---:|---:|---:|
| H4_CORRELATION_DIVERGENCE | -0.08941 | +0.06482 | 0.8760 | 1.0822 | -2.67% | +0.30% | **+2.98%** |
| H4_D1_OB_BOS_RISK | -0.01245 | -0.14382 | 0.9823 | 0.8133 | -0.69% | -1.75% | -1.06% |
| H4_D1_S6_VOL_RISK | +0.09185 | +0.05013 | 1.1362 | 1.0700 | +2.92% | +0.45% | -2.47% |
| H4_OB_BOS_RETEST | -0.00860 | -0.08282 | 0.9880 | 0.8919 | -0.79% | -1.49% | -0.71% |
| H4_S6_BREAKOUT | +0.15576 | **+0.36179** | 1.2251 | **1.5609** | +8.60% | +5.13% | -3.46% |
| H4_SUPPLY_DEMAND | +0.03590 | +0.01111 | 1.0499 | 1.0147 | +0.36% | +0.03% | -0.33% |

`H4_S6_BREAKOUT` is especially important for the next experiment: the filter increased mean R from +0.1558R to +0.3618R and PF from 1.2251 to 1.5609 while reducing its simple event-level MDD from -15.23% to -4.13%. However, it selected only 56 of 217 S6 test events, so total return declined from +8.60% to +5.13%. This is a strong reason to test a strategy-family-specific filter rather than the current pooled model, but it is not sufficient evidence for promotion.

## Multiple-testing and robustness diagnostics

### Paired moving-block uplift CI

- Lower bound: **-0.06470%** mean account-return uplift per event
- Upper bound: **+0.04059%**

The interval crosses zero, so incremental ML uplift is not statistically supported by this diagnostic.

### Validation CSCV/PBO diagnostic

- Paths: **20**
- PBO: **0.65**
- Median OOS rank percentile: **0.3553**
- Frozen maximum PBO gate: **0.50**

Result: FAIL. The model/threshold search is too selection-unstable to promote.

### Deflated-Sharpe-style diagnostic

- Selected untouched-test observations: **246**
- Search trials represented: **189**
- Per-trade Sharpe: **0.02626**
- Multiple-testing benchmark Sharpe: **0.12961**
- Deflated-Sharpe probability: **0.05098**
- Frozen gate: **0.95**

Result: FAIL.

## Transaction-cost stress

| Round-trip friction | Base mean R | Base PF | Base return | Filtered mean R | Filtered PF | Filtered return |
|---|---:|---:|---:|---:|---:|---:|
| 24 bps | +0.03394 | 1.0482 | +7.56% | +0.04438 | 1.0605 | +2.54% |
| 36 bps | -0.02713 | 0.9636 | -7.01% | -0.03503 | 0.9555 | -2.35% |
| 60 bps | -0.14929 | 0.8195 | -30.50% | -0.19384 | 0.7836 | -11.44% |

The edge is not robust to the 36 bps stress condition; the pre-registered stress gate therefore fails.

## Gate table

PASS:

- minimum untouched-test selected observations;
- positive filtered expectancy at the frozen 24 bps baseline;
- filtered PF >= 1.05 at the frozen 24 bps baseline.

FAIL:

- incremental total return;
- maximum drawdown <=5%;
- paired block-bootstrap uplift lower bound >0;
- symbol breadth >=60%;
- strategy breadth >=50%;
- Deflated-Sharpe probability >=0.95;
- validation PBO <=0.50;
- positive/PF>=1.0 at 36 bps stress.

## Scientific interpretation

v0.24 provides a useful negative/partial-positive result rather than a promotion. The pooled strategy-aware ML layer can rank some events better — most visibly for `H4_S6_BREAKOUT` and `H4_CORRELATION_DIVERGENCE` — but the improvement is not broad, stable or cost-robust enough to justify external replication or Forward PAPER.

The next justified experiment is a **strategy-family-specific meta-labeling and portfolio-overlap-aware risk study**, with unsupported sparse strategies such as the current `H4_KUMO_TRIANGLE` arm marked `DATA_INSUFFICIENT` rather than allowed to distort breadth/search statistics. Candidate families should be frozen before their new untouched test is generated.

RL remains disconnected. No v0.24 result authorizes Forward PAPER replacement or LIVE execution.
