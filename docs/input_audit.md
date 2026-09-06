# Supplied-input audit and provenance

The user supplied a proposal PDF, a research-summary PDF, a thesis Word draft and two Python-source drafts. The original documents and any personal administrative information are not redistributed here. They informed the requirements; they are not treated as independently verified empirical evidence.

## Source-code identities

| Input | SHA-256 |
|---|---|
| `406 robo.py` | `2030778983c8a251f360c176988474eab7bb5d5aac0c233178bebcffba7f55be` |
| `نسخه اخر.txt` | `cd7311e619395858745a5727d24519fed0e22561222e08d72ee13e8b1cb4cea2` |

Both parse as Python. Syntactic validity does not mean their full runtime is functional.

## Concrete findings

1. `406 robo.py` includes empty exchange, environment, risk, persistence and predictor methods. The latest text still has empty `DeepPredictor`, `FeatureExtractor` and `AR_DNN_Combination` classes. These are not working trained models.
2. Latest indicator construction creates `close.shift(-26)` for Chikou, then drops null rows. That series contains future prices at earlier rows and discards the latest 26 rows. The inspected selected RF feature list does not directly include Chikou, so direct RF-feature leakage through that column is not claimed; including it later would leak, and the stale-row behavior already exists.
3. The latest `AdvancedTradingEnvironment.step` computes drawdown from cash balance. Buying converts cash into an asset, which must still be included in marked equity. The new broker uses cash plus inventory times price.
4. The legacy step uses current close for buying but next close for selling, mixing execution clocks. The new implementation fixes intention at close t and applies signal fills at open t+1.
5. The latest code references `extra_indicators_integrated`, `gru_price_predictor` and `state_engineering` as separate modules, although only the uploaded scripts were supplied. In-file class definitions do not satisfy missing module imports.
6. The classifier creates its final binary label after a negative shift comparison, which can turn an unknown future return into a negative class. New labels explicitly preserve missing targets.
7. Predictor training and RL training in the inspected legacy flow reuse the historical data without a clear chronological held-out protocol. The rewrite uses explicit training, validation, purge and test intervals with manifests.

These findings motivated a modular implementation rather than treating the old script as production-ready. Only targeted source inspection and parsing were performed on the legacy scripts; no original exchange or notification code was executed.

## Requirements reconciled

- Proposal: ML/DL/RL comparison, risk control, backtesting and paper trading.
- Research summary: causal Ichimoku, BTC/ETH, ablation, chronological evaluation and realistic friction.
- Word draft: LSTM/XGBoost agreement and independent risk management; some sections remain placeholders.
- Earlier user preference: BTC 4h. The reference profile retains 4h; a separate 1h profile represents the newer report's recommendation. This design choice can be changed through a new versioned configuration.
- Conflicting triangle/crossover descriptions and ambitious DT/world-model components are not silently asserted as completed; see the roadmap.

## Rights and research integrity

The proposal contains an institutional intellectual-property notice. This repository does not assign a permissive redistribution license to the supplied thesis/code automatically. License selection should follow the user's applicable university arrangements. This is a provenance note, not a legal determination. Original upstream `Gym-Trading-Env` authorship and its separate license must remain intact when that code is used.

The new source has no authenticated trading, withdrawal or notification path. Scientific citations identify the original contributions. Placeholder results and literature-reported profits are not relabeled as results of this software.
