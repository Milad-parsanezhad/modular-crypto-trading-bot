# Experiment protocol — v0.1.0

Protocol snapshot: 2026-09-06. A runnable development protocol is implemented; confirm the final preregistered thesis protocol with the supervisor before interpreting any final evaluation.

## Questions

1. Does adding causal Ichimoku features improve the same model under the same friction and risk constraints?
2. How do classical, sequence and PPO policies compare to passive and rule baselines across chronological windows?
3. How sensitive are results to fees, slippage and trailing market regimes?

All are hypotheses, not established positive results. A negative or flat result must remain in the record.

## Data and partitions

- BTC/USDT and ETH/USDT: independent spot experiments. Their account balances are separate; no joint portfolio or cross-asset allocation is implemented.
- 4h reference, 1h sensitivity. The user's earlier 4h preference is the reference; the supplied research report also recommends 1h, so both are represented.
- Validate ascending unique UTC candle-open times, exact spacing, finite positive OHLC, nonnegative volume and completed candles. Reject gaps rather than synthesize executable prices.
- Feature warmup removes leading rows only for valid inputs. Current cloud uses values computed 26 bars earlier. Chikou uses current close versus close 26 bars ago; no future shift is used as a predictor.
- Binary target: open(t+1+h)/open(t+1)-1 exceeds estimated round-trip fee plus slippage. It is a coarse opportunity label, not a calibrated profit expectation. LSTM here is a classifier, whereas some uploaded drafts propose regression; this choice is explicit.
- Expanding folds use train -> h+1-bar purge -> validation -> h+1-bar purge -> test. Label information must end before the next partition. Tests do not overlap. Later folds may learn from observations available by then.
- Historical context before a partition is allowed in its input window. Scalers fit only training history. Val labels support early stopping and threshold selection. Test labels are only used for post-hoc forecast diagnostics.
- Current configs consume the requested number of folds from the beginning of the supplied feature frame; they do **not** automatically use all remaining observations. Unused observations are not automatically declared a pristine final holdout.

## Baselines and selection

Cash and passive full-allocation buy-and-hold accompany risk-matched always-long and Ichimoku baselines. Passive buy-and-hold pays fees/slippage but does not have strategy risk overlays; its difference must be disclosed. Ichimoku long means close above current cloud and Tenkan above Kijun. Triangle-under-cloud logic from earlier conversations is not included because its exact crossover rules conflict across earlier descriptions and are not fixed in the uploaded protocol.

RF, XGBoost, LSTM, CNN and patch Transformer use identical trailing windows within a feature condition. Seed, capacity and early-stopping policy are fixed in source/config. Probability cutoffs are selected by validation net return from a small declared grid. Probabilities are **not** calibrated confidence guarantees. PPO uses a fixed training budget and no test-based tuning. LSTM/XGBoost agreement takes a long position only if both individual thresholds agree; it is a separate experimental policy, not a claimed universal improvement.

Each seed is reported. Do not treat multiple seeds on the same market returns as independent observations. Compute budgets differ by model and by timeframe: record wall time and actual PPO steps before any equal-budget claim.

## Friction, risk and reporting

Reference fee = 0.1% per side; slippage = 5 bps per side. These are experimental assumptions, not verified account-specific CoinEx fees. Long-only position sizing respects cash including fees, a maximum allocation and an estimated stop-risk budget. ATR stop/take and total/daily drawdown triggers apply equally to active policies. No leverage, funding or borrowing is used.

Metrics: net total return, annualized return, Sharpe (zero risk-free rate; 24/7 annualization), Sortino (zero target), MDD, Calmar, trade-level profit factor, win rate, completed round trips, fees, turnover notional and empirical 95% CVaR loss. Undefined ratios are null, not infinity. Annualized short-run figures are unstable and should not headline results. Report time out of market alongside any policy that stays in cash.

With/without-Ichimoku uses the same warmed-up rows, splits and risk data. ATR stays in both because it controls risk for every policy. Friction stress uses 1x/2x/3x fee and slippage for supervised policies with a frozen intent sequence. It is not a retrained cost-aware policy comparison. PPO stress retraining/re-evaluation is not yet implemented. Regime diagnostics use the preceding decision candle's trailing 48-bar return relative to realized volatility; these are heuristic regimes, not economic ground truth.

Circular moving-block bootstrap intervals compare mean net per-bar returns against passive buy-and-hold within each contiguous test fold. They are **descriptive and not adjusted for trying multiple models, seeds, thresholds or hypotheses**. This implementation does not claim to compute Deflated Sharpe Ratio, PBO or a confirmatory significance test. Do not infer statistical superiority from interval exclusion alone after model selection.

## Final research gates

1. Retrieve and version actual BTC and ETH histories; inspect gaps and exchange-specific coverage.
2. Freeze data dates, model/threshold search budget, risk assumptions and primary endpoint.
3. Complete multi-seed development comparisons and record every attempted experiment.
4. Reserve a genuinely untouched later period; use a dedicated final configuration after freezing the design. Do not repeatedly reuse it for development.
5. Evaluate costs, realistic execution assumptions and multiple comparisons with the supervisor.
6. Run forward paper simulation long enough to observe adverse conditions and operational failures.

The current software validation is synthetic and does not satisfy these empirical gates. Live order execution, market-depth modeling, on-chain/sentiment alignment, Decision Transformer, DreamerV3, CVaR-constrained optimization and hyperparameter search services remain separate extensions.
