# v0.25 Risk-Aware Ranking — Validation Freeze Results

Date: 2026-09-11  
Workflow: `34590214474` — **SUCCESS**  
Artifact: `v25-risk-aware-ranking-34590214474`  
Artifact ID: `10195326409`  
Artifact digest: `sha256:3380bc537b551fd7f62ddbd37f250c9eaacb52d5e1dabd6799b39eac78d4a6bc`

## Scientific status

**`CHALLENGER_NOT_PROMOTED`**

The v0.25 ranker was fit on archived development data and selected on archived validation data only. The v0.24d OKX/KuCoin sample was used only after the ranker was frozen as a **SPENT diagnostic** and cannot authorize promotion.

`forward_paper_authorized = false`  
`paper_replacement_authorized = false`  
`live_execution_authorized = false`

## Engineering / reproducibility checks

- exact v0.24b model artifact and dataset hashes verified;
- original scikit-learn / NumPy / pandas / joblib persistence environment restored;
- exact archived frozen replay passed;
- Colab JSON validated;
- v0.24d/v0.24c regression and v0.25 allocator tests passed;
- fail-closed decision contract passed.

## Validation-only ranking tournament

| Ranker | Accepted | Return | PF | Mean R | Max realized DD | Hard kill |
|---|---:|---:|---:|---:|---:|---|
| **HGB expected-R** | **69** | **+3.922%** | **1.3427** | **+0.2295R** | **-4.672%** | **No** |
| HGB lower-quartile R | 55 | +2.785% | 1.2921 | +0.2083R | -5.197% | Yes |
| Pairwise logistic ranker | 57 | +2.258% | 1.2256 | +0.1641R | -5.172% | Yes |
| Ridge expected-R | 55 | +1.878% | 1.1952 | +0.1428R | -5.198% | Yes |
| Frozen score margin | 55 | +1.874% | 1.1948 | +0.1426R | -5.201% | Yes |
| Frozen score baseline | 55 | +1.874% | 1.1948 | +0.1426R | -5.201% | Yes |

Frozen validation champion: **`hgb_expected_r`**.

Relative to the frozen-score admission baseline on the selection sample, HGB expected-R increased accepted trades from 55 to 69, validation return from +1.874% to +3.922%, PF from 1.1948 to 1.3427, mean R from +0.1426R to +0.2295R, and kept realized drawdown inside the 5% research ceiling. These are **selection-sample results** and are not fresh evidence.

## Spent v0.24d diagnostic — deliberately non-promotional

The frozen HGB ranker was then inspected on the already-consumed v0.24d external rows only to diagnose transfer behavior.

### OKX spent diagnostic

- frozen-score allocator return: **+4.175%**
- HGB ranker diagnostic return: **+3.496%**
- uplift: **-0.679 percentage points**
- PF: `1.0812 → 1.0678`
- mean R: `+0.0606R → +0.0517R`
- DD: `-5.196% → -5.378%`

### KuCoin spent diagnostic

- frozen-score allocator return: **+7.567%**
- HGB ranker diagnostic return: **+5.020%**
- uplift: **-2.547 percentage points**
- PF: `1.1296 → 1.0891`
- mean R: `+0.0951R → +0.0671R`
- DD: `-5.143% → -5.386%`

This diagnostic is scientifically useful because it shows the validation gain **did not transfer automatically** to the previously observed external regimes. It strengthens the need for genuinely future-time evidence and argues against immediate complexity escalation.

## Interpretation

v0.25 provides a better validation allocator but **not evidence of generalized portfolio alpha**. The spent diagnostic is directionally negative for the learned ranker. Therefore:

- do not tune HGB against OKX/KuCoin v0.24d;
- do not replace the frozen event-score baseline post-hoc;
- freeze the v0.25 validation champion exactly as produced;
- begin prospective future-time collection from the pre-registered boundary;
- compare the frozen HGB ranker and frozen-score baseline only on new future outcomes;
- require MTM/CVaR/correlation/cost/search-aware gates before PAPER replacement.

## Next gate

`PROSPECTIVE_FUTURE_TIME_COLLECTION_THEN_MTM_CVAR_PORTFOLIO_REPLAY`

Future evidence boundary: `2026-09-11T12:00:00Z`.
