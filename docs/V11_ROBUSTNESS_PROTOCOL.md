# v0.11 Robustness & Statistical Inference Protocol

## Status

**Research-only. LIVE execution remains disabled.**

v0.10 produced a purged expanding OOS tournament over a live-discovered eligible universe, but no learned model passed promotion. Aggregate performance was regime-sensitive and the provisional learned winner was not fold-stable. v0.11 therefore tests robustness before any paper-execution promotion.

## Research question

Does any learned cross-sectional model retain a **net, coverage-matched, risk-adjusted edge** over simple baselines after accounting for model randomness, serial dependence, multiple testing, market regime and transaction costs?

## Required diagnostics

1. **Multi-seed stability**
   - Logistic Regression, HistGradientBoosting and Random Forest are re-run across independent seeds.
   - Report median/mean/std/min/max net return, median Sharpe, Sharpe dispersion, positive-seed fraction and median maximum drawdown.
   - A single hero seed cannot authorize promotion.

2. **Paired moving-block bootstrap**
   - Candidate and baseline returns are matched on the exact same OOS timestamps.
   - Moving blocks preserve local serial dependence better than i.i.d. resampling.
   - Report confidence intervals for mean net-return difference, total-return difference and Sharpe difference.

3. **Multiple-testing control**
   - All 3 learned models are tested against Momentum, Ichimoku and Equal-Weight Market baselines.
   - Benjamini-Hochberg FDR correction is applied across these nine planned pairwise tests.

4. **Coverage-matched diagnostics**
   - Pairwise inference uses inner timestamp matching only.
   - Coverage-match fraction must be reported; missing periods cannot be silently treated as zero return.

5. **Regime-conditional diagnostics**
   - Point-in-time regimes: `HIGH_VOL`, `TREND_UP`, `TREND_DOWN`, `RANGE`.
   - Report return, Sharpe, Sortino, maximum drawdown and CVaR where enough OOS periods exist.

## Fail-closed promotion gate

A learned candidate can only reach:

`PROVISIONAL_ROBUSTNESS_WINNER_STILL_NOT_EXECUTION_APPROVED`

when all current v0.11 gates pass:

- positive median seed return;
- positive-return seed fraction >= 80%;
- median max drawdown not worse than -35%;
- paired bootstrap lower confidence bound for mean edge over the strongest baseline > 0;
- FDR-adjusted q-value <= 0.10 versus the strongest baseline;
- coverage-match fraction >= 95%;
- no obvious regime-concentration failure when at least two regimes have adequate observations.

Otherwise status remains `NO_MODEL_PROMOTED` with explicit reasons.

Passing v0.11 **still does not authorize PAPER, TESTNET or LIVE trading**. The full Great Filter in the research specification also requires cost/capacity stress, final untouched testing, independent risk checks and execution readiness.

## Scientific interpretation rule

- `p < 0.05` or `q < 0.10` is not proof of future profit.
- Confidence intervals and effect size must be read with costs, drawdown, coverage and regime stability.
- Negative results are retained.
- Model complexity is not rewarded unless OOS evidence justifies it.
