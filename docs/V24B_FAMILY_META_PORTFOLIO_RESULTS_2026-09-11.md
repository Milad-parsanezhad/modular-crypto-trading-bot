# v0.24b Family-Specific Meta-Labeling + Portfolio Results — 2026-09-11

Workflow run: `34578494059`

Source commit: `1ac8cb34791ab9fa589cbd1b3004586204183b56`

Artifact: `v24b-family-portfolio-34578494059` (artifact ID `10190676664`)

Artifact SHA-256: `0384391db85922309e7b67e0e0481f2cbf8795cf069ee48f146e36ba857525db`

## Frozen scientific status

`EXPLORATORY_REUSED_SHADOW_NO_PROMOTION`

The terminal CoinEx segment was already observed during v0.24 before this v0.24b design existed. It is therefore a **reused SHADOW segment**, not a fresh untouched test. No v0.24b number can authorize Forward PAPER replacement or LIVE execution.

## Data and family eligibility

The run reproduced **4,979 strategy events across 12 symbols**.

Eligibility used sample counts only, never outcome quality:

| Strategy | Development | Validation | Shadow | Status |
|---|---:|---:|---:|---|
| H4_CORRELATION_DIVERGENCE | 363 | 99 | 117 | FAMILY_MODEL_ELIGIBLE |
| H4_D1_OB_BOS_RISK | 566 | 185 | 178 | FAMILY_MODEL_ELIGIBLE |
| H4_D1_S6_VOL_RISK | 466 | 159 | 130 | FAMILY_MODEL_ELIGIBLE |
| H4_OB_BOS_RETEST | 797 | 271 | 267 | FAMILY_MODEL_ELIGIBLE |
| H4_S6_BREAKOUT | 695 | 236 | 217 | FAMILY_MODEL_ELIGIBLE |
| H4_SUPPLY_DEMAND | 146 | 40 | 44 | DATA_INSUFFICIENT |
| H4_KUMO_TRIANGLE | 3 | 0 | 0 | DATA_INSUFFICIENT |

Five family-specific models were therefore fit. Model family, seed and threshold were frozen using validation only; the shadow segment was not used for model/threshold selection or refit.

## Family-specific shadow ablation

| Strategy | Champion | Shadow selected | Base mean R | Filtered mean R | Base PF | Filtered PF | Base return | Filtered return | Filtered MDD |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| H4_CORRELATION_DIVERGENCE | SGD Logistic | 40/117 | -0.0894R | -0.1243R | 0.8760 | 0.8362 | -2.67% | -1.27% | -2.02% |
| H4_D1_OB_BOS_RISK | Logistic | 30/178 | -0.0124R | **+0.1866R** | 0.9823 | **1.2536** | -0.69% | **+1.38%** | -3.88% |
| H4_D1_S6_VOL_RISK | SGD Logistic | 24/130 | +0.0919R | -0.0238R | 1.1362 | 0.9690 | +2.92% | -0.16% | -1.63% |
| H4_OB_BOS_RETEST | Gradient Boosting | 38/267 | -0.0086R | -0.3230R | 0.9880 | 0.6090 | -0.79% | -3.05% | -4.02% |
| H4_S6_BREAKOUT | SGD Logistic | 91/217 | +0.1558R | **+0.1924R** | 1.2251 | **1.2794** | +8.60% | +4.38% | **-6.94%** |

Interpretation:

- `H4_D1_OB_BOS_RISK` is the strongest family-specific **shadow improvement**: it moved from negative base return to positive filtered return, with positive mean R and PF > 1.25.
- `H4_S6_BREAKOUT` again improved per-trade quality and drawdown after filtering, but the lower coverage reduced total return materially.
- `H4_CORRELATION_DIVERGENCE` had positive *relative* total-return uplift because the loss became smaller (`-2.67%` to `-1.27%`); it did **not** become profitable in this family-specific run.
- `H4_D1_S6_VOL_RISK` and `H4_OB_BOS_RETEST` were damaged by the family-specific filters.

For every family, the paired moving-block uplift confidence interval crossed zero. No family has statistically supported incremental uplift in this reused shadow experiment.

## Overlap-aware portfolio result

The new portfolio engine enforces one active position per symbol, 1.00% maximum open portfolio risk, 0.50% maximum open risk per strategy, 0.75% same-direction risk, maximum 5 concurrent positions, and a 5% realized-equity drawdown kill.

### Base strategy pool under portfolio constraints

- Shadow events presented: **953**
- Trades accepted after overlap/risk controls: **161**
- Acceptance fraction: **16.89%**
- Total realized-equity return: **+10.84%**
- Profit factor: **1.3875**
- Mean accepted trade: **+0.2624R**
- Maximum realized-equity drawdown: **-3.70%**
- Maximum concurrent positions: **4**
- Maximum observed open risk: **1.00%**
- Hard drawdown kill: **not triggered**

### Family-meta-filtered pool under the same portfolio constraints

- Shadow events presented: **953**
- Family filter selected: **223**
- Trades accepted after overlap/risk controls: **60**
- Total realized-equity return: **-2.33%**
- Profit factor: **0.7957**
- Mean accepted trade: **-0.1533R**
- Maximum realized-equity drawdown: **-5.44%**
- Hard drawdown kill: **triggered**

The family-specific ML layer therefore **failed at portfolio level**. The incremental return versus the constrained base portfolio was **-13.17 percentage points**.

This is a key result: optimizing event-level trade quality inside individual strategy families did not preserve value once simultaneous trades competed for symbol, strategy, directional and total-risk budget. Portfolio interaction must be part of model selection rather than an after-the-fact filter.

## What the portfolio engine changed

The unconstrained event pool contained many simultaneous candidates. The base portfolio rejected events mainly because of:

- directional open-risk cap: 256;
- portfolio open-risk cap: 235;
- same-symbol overlap: 176;
- strategy open-risk cap: 125.

This explains why event-level compounding from v0.24 overstated the practical relevance of many simultaneous signals. Under the explicit portfolio contract, only 161 of 953 shadow events were admitted.

## Limitation

Portfolio drawdown is currently **realized-equity drawdown**, not intrabar mark-to-market drawdown. Open positions are settled at their recorded causal bracket exits. A later stage must replay the open book against point-in-time OHLC to measure true MTM exposure, portfolio VaR/CVaR and correlation shocks.

## Decision

`NO_FAMILY_META_PROMOTION`

The family-specific filter is rejected as a portfolio-level replacement in this experiment.

- `forward_paper_authorized = false`
- `paper_replacement_authorized = false`
- `live_execution_authorized = false`

## Next justified experiment

The next research target should **not** tune the reused CoinEx shadow further. The justified path is:

1. freeze the current promising hypotheses (`H4_S6_BREAKOUT` and `H4_D1_OB_BOS_RISK`) without reading another CoinEx pseudo-test;
2. obtain untouched external-venue or future-time data;
3. evaluate strategy + allocator jointly with mark-to-market portfolio accounting;
4. incorporate rolling cross-asset correlation and CVaR constraints;
5. run CPCV / multiple-testing-aware validation before any Forward PAPER decision.

RL remains disconnected and LIVE execution remains fail-closed.
