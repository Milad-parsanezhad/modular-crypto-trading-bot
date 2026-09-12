# v0.39R Test Protocol

v0.39R separates **architecture verification** from **empirical performance evaluation**.

## Phase A — reconstruction CI (required now)

The dedicated workflow `.github/workflows/v39r-reconstruction.yml` must pass:

- component-manifest governance checks;
- discrete decision-domain checks (`-1/0/+1`);
- causal future-perturbation/no-lookahead test;
- risk-weight upper-bound test;
- explicit verification that paper/live execution remain disabled;
- deterministic synthetic smoke run.

A Phase A pass means only that the canonical strategy is internally consistent and leakage-safe under the tested invariants. It is **not** evidence of profitability.

## Phase B — development-only empirical evaluation (next, separately preregistered)

After Phase A passes, v0.39R may be evaluated on already-consumed development venues only (CoinEx, OKX, KuCoin) using the frozen research constraints inherited from the post-v0.38 protocol.

Required outputs must include at minimum:

- selected event count;
- net expectancy in R;
- profit factor;
- positive-asset breadth;
- moving-block bootstrap confidence interval;
- positive-quarter fraction;
- transaction-cost stress including 36 bps round-trip;
- venue-by-venue and quarter-by-quarter attribution;
- ablation of each canonical layer.

## Holdout rule

Kraken remains sealed throughout v0.39R reconstruction and development evaluation. It can be touched only after a preregistered development candidate passes every frozen gate and an explicit holdout authorization is recorded.

## Promotion rule

No synthetic test, unit test, single-venue result, or average-profit result can authorize PAPER or LIVE execution.
