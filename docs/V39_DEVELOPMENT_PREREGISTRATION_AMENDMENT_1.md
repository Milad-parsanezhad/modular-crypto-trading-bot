# v0.39 Development Preregistration — Amendment 1

Status: **FROZEN BEFORE FIRST EMPIRICAL v0.39 RUN**

This amendment clarifies implementation details before any v0.39 development characterization result is observed. It does not change the mother-strategy features, venues, symbols, sample window, model order, or profitability gates.

## A. Shallow MLP optimizer clarification

The current executable characterization uses `sklearn.neural_network.MLPRegressor`, therefore the frozen shallow-MLP implementation is:

- optimizer: Adam as implemented by scikit-learn;
- L2 regularization: `alpha = 1e-4`;
- hidden layers: `(64, 32)`;
- ReLU;
- learning-rate init: `3e-4`;
- max iterations: `80`;
- early stopping enabled;
- validation fraction `0.15`;
- no-improvement patience `8`;
- seeds `314`, `1618`, `2718`;
- final prediction = median of all three seeds.

No claim is made that sklearn Adam is AdamW. A true PyTorch AdamW MLP is outside this frozen run and requires a new experiment unless implementation-equivalence is established prospectively.

## B. Financial drawdown qualification gate

In addition to the preregistered venue gates, each venue must satisfy:

- maximum realized account drawdown under the v0.39 financial allocator must be `<= 5.0%`.

This makes the existing 5% hard drawdown firewall an explicit development qualification criterion rather than only an entry-time governance rule.

## C. Loss-streak psychology cooldown

For this 4h characterization:

- after the third consecutive realized losing trade, new entries are blocked for `2 calendar days`;
- no risk increase after a loss is permitted;
- no martingale or averaging down is permitted;
- maximum new entries per calendar day = `4`.

The two-day value follows the existing 4h psychology/risk lineage and is frozen before the run.

## D. Same-bar / settlement causality

Open-position P/L is settled for capital-allocation purposes only when `exit_time < new_entry_time`. If an exit and a possible new entry share the same timestamp, the old position is not treated as known before the new entry. This is intentionally conservative.

## E. Final decision semantics

Passing development gates still does **not** authorize Kraken, paper execution, or live execution. It can only produce a development winner eligible for a separate holdout-authorization decision.
