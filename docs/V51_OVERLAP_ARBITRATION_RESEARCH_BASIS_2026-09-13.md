# v0.51 Overlap-Conflict Arbitration — Research Basis

Date: 2026-09-13  
Status: **LITERATURE / MECHANISM REVIEW — NO POST-RESULT DESIGN CHANGE**

## Research question

v0.50 isolated the earliest-first non-overlap transformation as the only pipeline
stage satisfying its preregistered broad-harm attribution rule. v0.51 therefore asks
one bounded question: when several already-admitted candidate trades are executable at
the same time while the symbol slot is flat, does a causal priority rule based on the
already-frozen Expected-R improve the economic result relative to earliest-first?

The literature below supports treating concurrency, transaction cost and scarce-risk
allocation explicitly. It does **not** establish that v0.51 A1 will outperform A0.

## 1. Overlapping financial labels are not IID observations

López de Prado's *Advances in Financial Machine Learning*, Chapter 4, treats overlapping
outcomes, concurrent labels and average uniqueness as explicit financial-ML problems.
For this thesis, the implication is operational: simultaneous/overlapping candidate
trades cannot be treated as independent opportunities when they compete for the same
capital/risk slot.

v0.51 therefore keeps one `venue × symbol` slot and compares two causal arbitration
rules rather than pretending every overlapping event can be executed independently.

## 2. Portfolio selection must include transaction costs and risk constraints

Guo, Gu, Fok & Ching, *European Journal of Operational Research* (2023), DOI
`10.1016/j.ejor.2023.05.001`, formulate online portfolio selection with explicit
transaction costs and a risk-parity constraint. Their result is not directly portable
to this crypto system, but it supports the broader design principle that allocation
quality should be judged **after** costs and under risk constraints.

Guo et al., *Quantitative Finance* (2024), DOI
`10.1080/14697688.2023.2287134`, similarly study adaptive online mean-variance
portfolio selection with transaction costs and peer/market effects.

Accordingly, v0.51 leaves the frozen transaction-cost assumptions and Financial
Governor unchanged. Arbitration is not allowed to claim improvement merely by
increasing turnover, leverage or drawdown.

## 3. Crypto order flow can be predictive, but prediction and portfolio economics differ

Anastasopoulos, Gradojevic, Liu, Maynard & Tsiakas (2026), *Journal of Financial
Markets*, DOI `10.1016/j.finmar.2026.101047`, report economically meaningful
out-of-sample information in world cryptocurrency order flow, including nonlinear-ML
portfolio applications and transaction-cost analysis.

This supports the thesis-wide distinction between:

1. predictive/event information; and
2. portfolio admission/arbitration under execution constraints.

The repository's own v0.24d/v0.50 evidence independently points in the same direction:
event-level or pre-admission economics can look better than the final scarce-risk
portfolio after overlap and governor transformations.

## 4. Why v0.51 remains deliberately simple

The current hypothesis is intentionally **not** a reinforcement-learning allocator,
optimizer search, Transformer or multi-objective hyperparameter sweep. A complex policy
would make it impossible to attribute any improvement specifically to correction of
the v0.50 overlap bottleneck.

The two-arm design is therefore:

- A0: earliest-first control;
- A1: maximum already-frozen Expected-R among simultaneous candidates only, with no
  pre-emption.

If A1 fails, a more complex allocator is not justified by this experiment. If A1 passes
its full prospective gate, it advances only to a separately preregistered confirmation
protocol.

## 5. Future hypotheses explicitly deferred

The following may be studied only after v0.51 is spent and under new preregistration:

- Expected-R minus incremental CVaR penalty;
- correlation-aware capital locking / diversification penalty;
- constrained mean-variance or risk-parity arbitration;
- uncertainty-adjusted Expected-R lower confidence bound;
- contextual bandit or reinforcement-learning allocation.

These are not hidden v0.51 rescue options.

## 6. Scientific consequence of the predictor-identity audit

A literature-supported allocation rule is still scientifically invalid if its input
model is not uniquely identified. The v0.47 C1 experiment produced a family of
fold×symbol models rather than one persisted production predictor. Therefore the
v0.51 predictor-identity amendment freezes one production object per allowed asset
using the latest canonical fold by chronology and requires artifact-level verification
before future scoring.

This is a reproducibility prerequisite, not a performance optimization.

## Safety / claim boundary

`RESEARCH_ONLY / KRAKEN_SEALED / PAPER_OFF / LIVE_OFF`

No article cited here is treated as proof that the repository has alpha. Repository
promotion decisions remain governed by its own fresh prospective evidence.
