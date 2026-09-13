# v0.51 — Prospective Overlap-Conflict Arbitration — Preregistration

Status: **DESIGN FROZEN BEFORE v0.51 PROSPECTIVE OUTCOMES**  
Parent scientific result: `V50_NONOVERLAP_FAILURE_SUPPORTED`  
Engineering parent: `deploy/research-v50@7cf4410ba5de2b5e4feb2224b53b5ddccadb7af5`
Immutable preregistration commit: `d8ee4576aaf55750dd5910cc0d3b2efcbba3f5b2`  
Immutable preregistration time: `2026-09-13T04:49:28Z`

## Question

Can a causal, expected-R-aware rule resolve simultaneously executable
candidate conflicts more effectively than the frozen earliest-first rule on
new post-registration development evidence, without changing the predictor,
admission threshold, costs, exits, Financial Governor, venues or assets?

This is a prospective hypothesis. The consumed v0.47–v0.50 outcomes may explain
why the question exists, but they may not confirm v0.51.

## Frozen scope and firewall

- development venues: CoinEx, OKX and KuCoin only;
- universe: BTC, ETH, SOL, XRP and DOGE only;
- exact canonical v0.47 `C1_TEMPERATURE_R1` predictor and FIT-derived state means;
- admission remains `expected_r_v47 > 0`;
- transaction costs, 36 bps stress costs, exits and Financial Governor remain unchanged;
- no venue, asset, side, regime or event-family pruning;
- no refit, threshold search or arbitration hyperparameter search;
- evidence must have `signal_time` strictly after the immutable preregistration commit time;
- Kraken remains sealed; `PAPER=false`; `LIVE=false`.

## Arms

### A0 — frozen control

The v0.39/v0.50 earliest-first rule: within `venue × symbol`, order by
`entry_time, signal_time`, admit the first candidate, and block later candidates
through its exit.

### A1 — primary v0.51 rule

Policy identifier: `SIMULTANEOUS_MAX_EXPECTED_R_NO_PREEMPTION`.

Within each `venue × symbol` stream:

1. process entry times chronologically;
2. if a position is active, reject every newly arriving overlapping candidate;
3. when flat and exactly one candidate is executable, admit it;
4. when flat and multiple candidates share the same `entry_time`, admit the
   candidate with greatest frozen `expected_r_v47`;
5. ties use earliest `signal_time`, then lexical `series_id`;
6. never pre-empt an active trade and never use `net_r`, outcome, future path,
   realized exit quality or candidates arriving after the decision time.

The winner's observed `exit_time` is used only to reproduce when the simulated
slot becomes flat; it is not part of the priority key.

## Required causal tests

- mutation of realized `net_r` and outcome cannot change admission;
- a later high-scored signal cannot retroactively pre-empt an active trade;
- input row order cannot change the winner;
- non-finite scores, non-positive/non-admitted candidates, invalid timestamps
  and duplicate decision identities fail closed;
- venue and symbol streams remain isolated.

## Prospective support and decision rule

Evaluation is deferred until all support conditions are met:

- five consecutive 30-day UTC blocks beginning `2026-09-13T08:00:00Z`;
- at least 100 simultaneous conflict cohorts overall;
- at least 10 simultaneous conflict cohorts in every block;
- exact common candidate input for A0 and A1;
- both arms pass through the unchanged Financial Governor;
- missing venue/block support is recorded as failure, never silently dropped.

A1 is **supported for advancement to a separate confirmation protocol** only if:

1. post-governor expectancy delta `A1 - A0` is positive in at least 3/5 blocks
   and its five-block median is positive;
2. aggregate post-cost expectancy is strictly greater than A0;
3. aggregate profit factor is not lower than A0;
4. 36 bps stress profit factor is not lower than A0;
5. worst drawdown remains within the frozen 5% firewall;
6. all causal, provenance and support checks pass.

Otherwise the result is `V51_ARBITRATION_NOT_SUPPORTED` or
`V51_INSUFFICIENT_PROSPECTIVE_SUPPORT`. Passing does not authorize candidate
promotion, Kraken, PAPER or LIVE; it only permits a separately preregistered
confirmation stage.

## Forbidden interpretations

This implementation is not evidence of profitability. Historical replay on
already consumed v0.47–v0.50 outcomes may be used only as a software
reproduction check and must be labeled exploratory. No decision threshold or
policy detail may be changed after observing prospective v0.51 outcomes.
