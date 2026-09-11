# v0.37 — Marginal-Utility Portfolio Arbitration

**Preregistered before v0.36 result inspection. Research only. Kraken remains untouched.**

## Scientific question

If v0.36 identifies economically useful events, can scarce portfolio risk be assigned by a causal ranking rule that accounts for uncertainty, capital-lock duration, tail loss and correlation rather than arbitrary symbol/strategy ordering?

This directly tests the post-v0.24d hypothesis that **event alpha and portfolio admission are distinct learning problems**.

## Literature basis frozen before result inspection

- Barak, Mousavi & Hosseini (2025), SSRN 5494646: dynamic learning-to-rank for crypto with explicit risk-aware allocation.
- Burdorf (2025), SSRN 5255258: learning-to-rank can improve momentum portfolio construction by modelling relative relationships rather than treating assets independently.
- *Portfolio constructions in cryptocurrency market: A CVaR-based deep reinforcement learning approach*, Economic Modelling 119 (2023), DOI 10.1016/j.econmod.2022.106078: motivates CVaR for crypto tail risk.
- *Asset correlation based deep reinforcement learning for portfolio selection*, Expert Systems with Applications (2023): motivates explicit cross-asset correlation features in allocation.
- Schmitt (2026), arXiv:2602.03903: motivates uncertainty-aware tail-risk control under nonstationarity.

## Frozen parent relationship

The three v0.36 model families are parents. Each is combined with each of three fixed admission policies, creating **9** v0.37 trials.

A v0.37 portfolio result can only be a joint candidate if its parent v0.36 model independently passes the v0.36 event-alpha gate. Therefore v0.37 cannot rescue a failed event model after outcomes are observed.

## Policies

All policies rank only already-selected v0.36 events.

### BALANCED
- correlation penalty weight: 0.20
- normalized CVaR penalty: 0.25
- predicted-duration penalty: 0.10
- conformal-uncertainty penalty: 0.20

### TAIL_DEFENSIVE
- correlation: 0.10
- CVaR: 0.40
- duration: 0.10
- uncertainty: 0.20

### DIVERSITY_FIRST
- correlation: 0.35
- CVaR: 0.15
- duration: 0.10
- uncertainty: 0.20

Common controls:
- maximum 4 new positions per entry batch;
- one open position per symbol;
- correlation/CVaR use trailing 60 Daily returns ending no later than signal time;
- open positions are settled only when `exit_time < new_entry_time`; same-time exits remain open;
- deterministic tie breaking by marginal score, symbol, then strategy;
- v0.31 pre-entry hard-DD firewall remains unchanged after admission;
- 2.0% aggregate open-risk cap, 1.5% same-direction cap, 5% hard drawdown cap, 24 bps round-trip friction.

## Marginal score

Within each simultaneous entry batch, v0.36 utility is converted to a percentile rank. Penalties are then applied for:

1. maximum absolute trailing correlation to currently open symbols;
2. normalized empirical lower-tail CVaR;
3. predicted holding duration;
4. conformal uncertainty buffer relative to point prediction.

No realized future outcome from the candidate itself enters the admission score.

## Development gate

Every consumed development venue (CoinEx, OKX, KuCoin) must satisfy:

- executed trades >= 200;
- PF >= 1.05;
- expectancy > 0 R;
- positive-symbol breadth >= 60%;
- |MDD| <= 5%;
- moving-block CI lower bound > 0.

And the parent v0.36 candidate must have passed its event-alpha gate.

## Multiplicity

Prior trials after v0.36: 123. New v0.37 policy combinations: 9. Total effective history: **132**.

## Outcomes

- `NO_V37_JOINT_EVENT_PORTFOLIO_CANDIDATE`
- `V37_JOINT_CANDIDATE_LOCKED_FOR_UNTOUCHED_KRAKEN`

Even the latter only locks one candidate for a **subsequent separately executed untouched Kraken holdout**. v0.37 itself never fetches Kraken, and PAPER/LIVE remain unauthorized.