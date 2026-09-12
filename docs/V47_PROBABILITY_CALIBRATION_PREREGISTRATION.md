# v0.47 — Probability Calibration Ablation — Preregistration

Status: **PREREGISTERED BEFORE ANY v0.47 EMPIRICAL EXECUTION**

Parent result commit: `d3d7003b735ae72ca1be9b0b4d5d020916bf4d7c`
Research-note commit: `50736352b0c1e6c3dac3b22df45a118a90a938ea`
Source v0.44 run: `34700944062`
Source v0.46 canonical run: `34704267477`

## Question

Can post-hoc probability recalibration of the frozen v0.46 `R1_FOUR_STATE_TIMEOUT_SIGN` model improve common-space OOS probability calibration and downstream economic robustness without touching test labels or changing the trading system?

## Frozen invariants

The following are unchanged from the consumed lineage:

- Mother Strategy and ICT / SMC / Ichimoku / Al Brooks features;
- v0.41 causal feature representation;
- R1 four-state terminal encoding: TARGET / STOP / TIME_POSITIVE / TIME_NONPOSITIVE;
- asset-specific cross-venue multinomial logistic base learner;
- exact five chronological/purged v0.44 fold boundaries;
- CoinEx / OKX / KuCoin development venues;
- BTC / ETH / SOL / XRP / DOGE universe;
- transaction costs and 36 bps stress costs;
- Financial Governor and 5% drawdown firewall;
- non-overlap realization;
- no asset, venue, regime or event-family pruning;
- Kraken sealed;
- PAPER=false;
- LIVE=false.

## Frozen arms

### C0 — `C0_IDENTITY_R1`

No probability transformation. This is the control and must reproduce the R1 probability/economic semantics under the v0.47 runner.

### C1 — `C1_TEMPERATURE_R1`

A single scalar temperature is fit separately for each asset-fold using **CALIBRATION data only**.

- optimization target: four-state multiclass negative log likelihood;
- optimization variable: `log(T)`;
- fixed bound: `[-3.0, +3.0]`;
- probabilities: `softmax(log(p_base_clipped) / T)`;
- probability clipping epsilon: `1e-6`;
- TEST labels are forbidden from fitting T.

### C2 — `C2_DIRICHLET_R1`

Native multiclass calibration on log probabilities, fit separately for each asset-fold using **CALIBRATION data only**.

- input: `log(clip(p_base, 1e-6, 1.0))`;
- calibrator: multinomial logistic regression;
- L2 regularization: fixed `C=1.0`;
- solver: `lbfgs`;
- max iterations: `2000`;
- no hyperparameter search;
- TEST labels are forbidden from fitting the calibrator.

No fourth calibrator may be added after observing results.

## Support rule

For an asset-fold to be admissible for calibrated arms:

- FIT events >= 600;
- CAL events >= 50;
- TEST events >= 1;
- all four R1 states must occur in FIT;
- all four R1 states must occur in CAL for C2;
- C1 requires at least two CAL states and finite base probabilities;
- unsupported units fail closed and are recorded; they are never silently dropped from gate denominators.

## Data partition firewall

For every fold and asset:

1. base R1 model is fit on FIT only;
2. base probabilities are generated for CAL and TEST;
3. C1/C2 are fit on CAL only;
4. calibrated probabilities are generated for TEST;
5. forecast metrics, event selection, Financial Governor and economics are evaluated on TEST only.

No TEST label can affect fitting, arm choice, threshold, state mean or calibration parameter.

## Expected-R mapping

State economic means are estimated from FIT only and remain identical across C0/C1/C2 within the same asset-fold. Calibrated probabilities change only probability weights; realized labels or state means are not changed.

Selected event rule remains the frozen simple rule:

`expected_r > 0`

No post-result threshold change is allowed.

## Forecast scoring

Primary cross-arm space is the common three-state space:

`TARGET / STOP / TIME`

with:

`P(TIME) = P(TIME_POSITIVE) + P(TIME_NONPOSITIVE)`

Primary forecast metrics:

- median common-space multiclass Brier;
- median common-space mean reliability error;
- common-space macro one-vs-rest AUC.

Native four-state Brier/log-loss remain diagnostics only.

## Frozen promotion gates

A calibrated arm must pass every gate:

1. median common-space multiclass Brier strictly lower than C0;
2. median common-space mean reliability error strictly lower than C0;
3. median common-space macro OVR AUC >= C0 minus `0.01`;
4. positive OOS fold expectancy fraction >= `0.60` (>=3/5);
5. aggregate post-cost expectancy > `0`;
6. aggregate PF >= `1.05`;
7. aggregate 36 bps stress PF >= `1.00`;
8. worst Financial-Governor drawdown >= `-0.05`;
9. Kraken remains sealed;
10. no post-result threshold/asset/regime/event-family pruning.

## Deterministic selection rule

Evaluate complexity in the frozen order:

`C1_TEMPERATURE_R1 -> C2_DIRICHLET_R1`

The first/simplest calibrated arm passing **all** gates is the v0.47 development candidate. If C1 passes, C2 cannot replace it merely because C2 has better point metrics. If neither passes, decision is:

`V47_CALIBRATION_REJECT_OR_INSUFFICIENT_EVIDENCE`

If a calibrated arm passes, decision is:

`V47_DEVELOPMENT_CANDIDATE`

This is not permission to access Kraken, PAPER trade or LIVE trade. A subsequent prospectively preregistered stage is still required.

## Bug policy

If execution fails before a canonical scientific decision is produced:

- identify whether the defect is engineering or scientific;
- research the defect using authoritative documentation/literature when needed;
- patch only the defect;
- add a regression test reproducing it;
- record the superseded run and repair commit;
- rerun the same frozen contract;
- do not change arms, thresholds, gates, features, assets or costs in response to observed partial results.

## Prohibited rescue actions

- threshold relaxation;
- outcome-based asset deletion;
- venue deletion;
- regime/event-family pruning;
- calibrator hyperparameter sweep;
- blending C1 and C2 after results;
- Transformer / Decision Transformer / Mamba;
- PPO / DQN / other RL;
- on-chain / order-book / sentiment feature additions;
- Kraken access.

Final preregistered state:

`V47_PREREGISTERED / R1_FROZEN / CALIBRATION_ONLY / KRAKEN_SEALED / PAPER_OFF / LIVE_OFF`
