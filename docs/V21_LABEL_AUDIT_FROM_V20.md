# v0.21 Pre-ML Label Audit from v0.20 Ledger

Source artifact: v0.20 workflow run `34469292583`, `trade_attempt_ledger.csv`.

This audit is descriptive only. It is performed before model fitting so class imbalance and sample structure are explicit rather than discovered after choosing a learner.

## Dataset scale

- Total source-derived strategy attempts: **232,910**
- Development attempts: **139,693**
- Validation attempts: **46,170**
- Test attempts: **47,047**
- Attempts actually permitted by the v0.20 deterministic risk/psychology overlay: **15,115**
- Risk-policy execution rate: **6.49%**

The ML meta-label dataset uses strategy attempts as observations. Risk-rejected attempts are not erased: their hypothetical frozen-bracket outcomes remain useful labels for learning whether a candidate signal had edge, while the separate risk engine remains authoritative for execution permission.

## Label balance before ML

Using the post-cost R-multiple already recorded by the frozen v0.20 bracket simulator:

- `label_profitable_net = 1`: **24.34%**
- `label_target_hit = 1`: **24.51%**
- `label_ge_1r = 1`: **19.35%**
- `label_ge_2r = 1`: **12.48%**
- Mean R across all attempts: **-1.2335R**
- Median R across all attempts: **-1.4220R**

Therefore plain accuracy is a dangerous objective: a trivial classifier can obtain a high raw accuracy by rejecting most events. v0.21 consequently reports balanced accuracy/AUC but selects its meta-filter primarily on validation-set trading economics and abstention quality.

## Label distribution by timeframe

| TF | Attempts | Profitable rate | Mean R | >=1R rate | >=2R rate |
|---|---:|---:|---:|---:|---:|
| 1m | 86,329 | 18.55% | -2.3602R | 11.83% | 2.22% |
| 5m | 35,125 | 24.58% | -1.3819R | 20.60% | 9.14% |
| 15m | 46,848 | 26.85% | -0.5760R | 24.80% | 21.49% |
| 1h | 41,077 | 27.95% | -0.1865R | 25.05% | 23.25% |
| 4h | 15,577 | 31.35% | -0.0271R | 24.97% | 20.77% |
| 1d | 7,954 | 38.87% | +0.0067R | 23.01% | 13.53% |

This is strong empirical evidence that the current fee/slippage/stop structure is especially hostile to the shortest timeframes. The meta-model is allowed to learn this structure, but timeframe is a declared categorical feature and not a hidden post-hoc filter.

## Strategy families with >=1,000 recorded attempts and strongest full-ledger mean R

The following are descriptive full-ledger statistics, **not selection results**:

| Strategy | TF | Attempts | Win rate | Mean R |
|---|---:|---:|---:|---:|
| H4_S6_BREAKOUT | 4h | 3,564 | 32.60% | +0.0455R |
| D1_OB_BOS_RETEST | 1d | 1,679 | 39.19% | +0.0361R |
| H4_D1_S6_VOL_RISK | 4h | 2,343 | 32.22% | +0.0197R |
| D1_WEEKLY_TSMOM_VOL_RISK | 1d | 1,613 | 35.83% | +0.0015R |
| D1_ICT_SWING_PD | 1d | 1,722 | 40.42% | +0.0009R |

These figures mix development, validation and test and therefore must never be used to select the final model or strategy. They are included only as an audit of label geometry.

## ML consequence

v0.21 trains on development only, chooses the abstention threshold and classifier champion on validation only, then evaluates the frozen champion on test. Future/outcome columns such as entry/exit/target/stop, realized return, R-multiple, exit reason and post-trade equity are banned from the model feature list. Only `f_*` signal-time snapshots plus declared strategy/timeframe/symbol/side metadata can be inputs.
