# v0.17 Search-Aware Statistical Audit Protocol

Date: 2026-09-09

Status: **Methodological upgrade for future candidate promotion**

## 1. Motivation

Repeated model, feature, horizon, threshold and regime search can create a strong-looking backtest even when no persistent edge exists. Therefore future promotion decisions must account not only for sampling uncertainty, but also for the search process that produced the selected candidate.

This protocol does not retroactively invent statistics that cannot be computed from archived return paths. Instead it defines what must be recorded and tested from the next candidate-search cycle onward.

## 2. Trial registry

Every candidate evaluation must receive an immutable trial record containing at least:

- trial ID;
- git commit;
- dataset fingerprint;
- feature family;
- model family;
- hyperparameters;
- random seed;
- forecast horizon;
- decision threshold;
- transaction-cost assumptions;
- regime filter, if any;
- OOS/holdout boundaries;
- gross and net return path artifact;
- promotion decision.

Failed and unattractive trials must remain in the registry. Deleting failed trials would invalidate search-aware inference.

## 3. Statistical layers

### Layer A — dependence-aware uncertainty

Use paired moving-block or stationary bootstrap for return differences when strategies are evaluated on matched time periods. The block length must reflect serial dependence and should be sensitivity-tested rather than chosen solely to maximize significance.

### Layer B — multiple-hypothesis control

Benjamini-Hochberg FDR remains useful when the family of planned pairwise hypotheses is explicit. It does not, by itself, solve the broader problem of selecting the best strategy from a large adaptive search.

### Layer C — Reality Check / SPA

White’s Reality Check tests whether the best strategy found in a specification search has predictive superiority over a benchmark after accounting for data snooping.

Hansen’s Superior Predictive Ability test improves power relative to the original Reality Check by reducing the influence of poor or irrelevant alternatives. For a future large strategy registry, SPA is preferred as a primary search-aware comparison, with Reality Check retained as a robustness companion where feasible.

### Layer D — Probabilistic and Deflated Sharpe

A raw Sharpe ratio is insufficient because it ignores finite-sample uncertainty, skewness, kurtosis and strategy selection.

- Probabilistic Sharpe Ratio asks how confident we can be that the population Sharpe exceeds a benchmark.
- Deflated Sharpe Ratio adjusts the apparent Sharpe for selection bias/multiple trials and non-normal return characteristics.

The number of trials used for DSR must reflect the genuine search history, not only the final short list.

### Layer E — PBO / CPCV

Probability of Backtest Overfitting is used when the dataset and experiment permit combinatorial purged cross-validation. The purpose is to estimate how often a strategy selected as best in-sample degrades materially out of sample.

PBO must not be computed on tiny or structurally inappropriate samples merely to populate a thesis table.

### Layer F — Minimum track record

Even a high observed Sharpe can be statistically weak when the track record is short or returns are highly non-normal/autocorrelated. Minimum Track Record Length reasoning therefore becomes a separate gate before risk-adjusted skill language is allowed.

## 4. Promotion hierarchy

A future candidate may move through these labels only in order:

`HYPOTHESIS`
→ `DISCOVERY_CANDIDATE`
→ `VALIDATED_OOS`
→ `SEARCH_AWARE_SURVIVOR`
→ `FORWARD_REPLICATED`
→ `TESTNET_READY`
→ `LIVE_REVIEW_CANDIDATE`

No label transition is automatic.

A strong backtest is not enough for `SEARCH_AWARE_SURVIVOR`. A search-aware survivor must satisfy, where applicable:

- positive net OOS performance after costs;
- risk-adjusted superiority over strong simple baselines;
- dependence-aware confidence interval supporting positive edge;
- planned multiple-testing family not rejected by FDR control;
- SPA/Reality-Check evidence against the benchmark after accounting for search;
- acceptable Deflated Sharpe / Probabilistic Sharpe evidence;
- acceptable PBO when CPCV is applicable;
- no single-regime concentration that invalidates the claimed scope;
- adequate track-record length.

## 5. Current project mapping

| Requirement | Current state |
|---|---|
| Purged OOS | implemented and used in v0.10-v0.11 |
| Explicit costs | implemented and used |
| Paired block bootstrap | implemented in v0.11-v0.12 |
| FDR | implemented in v0.11-v0.12 |
| Regime diagnostics | implemented |
| Untouched holdout | implemented in v0.12 |
| Prospective paper evidence | active in v0.15-v0.16 |
| Trial registry across all searched variants | partial — strengthen next |
| White Reality Check | not yet run on a complete search registry |
| Hansen SPA | not yet run on a complete search registry |
| DSR/PSR | infrastructure planned; do not report until valid trial count and return path exist |
| CPCV/PBO | existing research infrastructure references it; future candidate must supply a valid application artifact |
| Minimum track record gate | introduced formally in v0.17 |

## 6. Recent external research implication

Two recent crypto findings are especially relevant to future design:

1. Bysik & Ślepaczuk (2026) report that positive gross BTC forecast performance can disappear under 10-bp transaction costs, while cost-aware execution filters can materially change economic outcomes. This supports the project’s Net-Alpha / abstention architecture rather than naive sign trading.
2. Anastasopoulos et al. (2026), Journal of Financial Markets, find predictive value in cross-sectional world order flow across cryptocurrencies and report economically meaningful OOS performance for non-linear models. This motivates a future higher-quality order-flow experiment, but does **not** validate the project’s existing v0.12 proxy implementation.

This distinction is important: literature informs hypotheses; project-specific holdout evidence decides promotion.

## 7. Core references

- White, H. (2000). A Reality Check for Data Snooping. *Econometrica*, 68(5), 1097–1126. DOI `10.1111/1468-0262.00152`.
- Hansen, P. R. (2005). A Test for Superior Predictive Ability. *Journal of Business & Economic Statistics*, 23(4), 365–380. DOI `10.1198/073500105000000063`.
- Sullivan, R., Timmermann, A., & White, H. (1999). Data-Snooping, Technical Trading Rule Performance, and the Bootstrap. *Journal of Finance*, 54(5), 1647–1691. DOI `10.1111/0022-1082.00163`.
- Bailey, D. H., & López de Prado, M. (2014). The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality. *Journal of Portfolio Management*, 40(5), 94–107. DOI `10.3905/jpm.2014.40.5.094`.
- Anastasopoulos, A., Gradojevic, N., Liu, F., Maynard, A., & Tsiakas, I. (2026). Order flow and cryptocurrency returns. *Journal of Financial Markets*, 79, 101047. DOI `10.1016/j.finmar.2026.101047`.

## 8. Fail-closed rule

If the trial registry is incomplete, the search intensity is unknown, the return path is unavailable, or the sample is too short for a requested statistic, the output must be `DATA_INSUFFICIENT_FOR_SEARCH_AWARE_INFERENCE` rather than an imputed or optimistic score.
