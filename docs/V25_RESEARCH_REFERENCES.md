# v0.25 Research References and Comparable-System Audit

Frozen before v0.25 prospective economic evidence is consumed.

## Scientific references

1. Hu, W., Zhou, J. (2024). *Trading Signal Survival Analysis: A Framework for Enhancing Technical Analysis Strategies in Stock Markets*. Computational Economics 64, 3473–3507. DOI: `10.1007/s10614-024-10567-8`.
2. *Momentum portfolio selection based on learning-to-rank algorithms with heterogeneous knowledge graphs*. Applied Intelligence 54 (2024). DOI: `10.1007/s10489-024-05377-2`.
3. Arian, H., Norouzi Mobarekeh, D., Seco, L. (2024). *Backtest overfitting in the machine learning era: A comparison of out-of-sample testing methods in a synthetic controlled environment*. Knowledge-Based Systems 305, 112477. DOI: `10.1016/j.knosys.2024.112477`.
4. Kruthof, G., Müller, S. (2025). *Can deep reinforcement learning beat 1/N?* Finance Research Letters 75, 106866. DOI: `10.1016/j.frl.2025.106866`.
5. Jiang, Y., Olmo, J., Atwi, M. (2025). *High-dimensional multi-period portfolio allocation using deep reinforcement learning*. International Review of Economics & Finance 98, 103996. DOI: `10.1016/j.iref.2025.103996`.
6. Kato, M. (2024). *Conformal Predictive Portfolio Selection*. arXiv:2410.16333. Used as a hypothesis source for later uncertainty/lower-bound portfolio admission; not treated as proof of trading alpha.

## Primary implementation references

### CCXT OHLCV semantics

CCXT Manual: `https://github.com/ccxt/ccxt/wiki/manual`

Relevant documented constraints translated into v0.25 safeguards:

- unified timestamps are UTC milliseconds;
- `since` is a UTC millisecond timestamp and exchange behavior is exchange-specific;
- returned OHLCV series can contain legitimate gaps;
- historical depth is limited on many exchanges;
- continuous polling/storage is recommended when a reproducible history is required;
- the last/current candle may be incomplete until the next candle starts;
- secondary OHLCV data can have exchange-specific calculation latency.

Design response:

- timezone-aware UTC everywhere;
- completed-bar boundary before eligibility;
- deterministic bounded pagination;
- cursor advancement by a full timeframe rather than `+1ms`;
- gap/coverage diagnostics;
- append-only first-observed archive;
- historical-restatement detection instead of silent overwrite.

CCXT's sequential-OHLCV example also advances its pagination cursor by elapsed candle intervals rather than arbitrary sub-candle offsets: `https://github.com/ccxt/ccxt/wiki/fetch-ohlcv-sequentially`.

### GitHub Actions scheduling semantics

GitHub Docs: `https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows`

GitHub documents two operational facts important to scientific automation:

- a `schedule` workflow triggers only when the workflow file exists on the default branch and scheduled workflows run on the default branch;
- scheduled jobs can be delayed under high load, especially at the start of an hour, and moving the cron away from minute 0 reduces that risk.

Design response:

- the persistent scheduler is placed on `main`;
- it checks out an exact frozen scientific collector SHA rather than following branch drift;
- cron runs at minute 23 after each 4h boundary;
- scientific time eligibility is determined from exchange/bar timestamps, never from assuming the workflow fired at an exact wall-clock second.

## Comparable-system bug audit

### FinRL CCXT timezone/index drift

Public FinRL issue #1440 documents a CCXT processor path in which naive local-time conversion can shift exchange UTC timestamps and corrupt cross-asset alignment. v0.25 therefore retains timezone-aware UTC normalization and treats timestamp/index consistency as a testable data contract.

Reference: `https://github.com/AI4Finance-Foundation/FinRL/issues/1440`

### Qlib chained dependency/index failures

Public Qlib issue #2233 documents a sequence of compatibility failures involving LightGBM 4+, unhashable segment lists and pandas MultiIndex behavior. The key engineering lesson is that one blocked dependency bug can mask later failures. v0.25 therefore uses fail-fast dependency checks, exact persisted-model environment replay and separate regression gates instead of continuing after partial incompatibility.

Reference: `https://github.com/microsoft/qlib/issues/2233`

### Exchange pagination / boundary ambiguity

Across exchange APIs, `since` semantics, candle alignment and pagination inclusivity are not perfectly uniform. A superficially harmless `last_ts + 1ms` cursor can therefore create first-candle omission/alignment behavior on some venues. v0.25 treats the candle timeframe itself as the pagination unit and now has a regression test that requires a 4h cursor to move by exactly 14,400,000 ms.

This is an engineering lesson, not a claim that every exchange has the same defect.

## Statistical-design audit discovered during v0.25

The first draft of the future collector would have emitted PF/return/DD on every scheduled run while the sample accumulated. That design was rejected because repeated inspection creates an optional-stopping / research-flexibility channel even if the model weights are unchanged.

The repaired design is **blind until an outcome-independent maturity gate is satisfied**. Maturity uses only elapsed time, data coverage and event count. The first mature economic read is terminal for that test window.

The first draft also stated that 36 bps stress was required but did not make the stress result an explicit programmatic promotion gate. That contract/code mismatch was corrected. The frozen first mature gate now checks 36 bps portfolio PF/mean-R, rolling CVaR and paired time-series uplift uncertainty in addition to PF/return/DD.

These defects and repairs are intentionally preserved as part of the thesis software-validation trail rather than hidden from the final record.

## Project-specific lessons carried forward

- exact model binary + feature schema + threshold + environment + dataset hash is the scientific model identity;
- a successful classifier is not automatically a successful portfolio allocator;
- validation improvement does not imply cross-venue generalization;
- spent external evidence cannot be rebranded as a fresh test after redesign;
- future evidence must be blinded until a pre-registered first look to reduce optional stopping;
- an exchange's later historical restatement must not silently rewrite the evidence used at the time of decision;
- future RL must beat simple allocation baselines after costs and risk, not merely raw-return reward curves;
- all promotion claims require portfolio-level economic evidence, not accuracy/AUC alone;
- a green CI run is engineering evidence, not alpha evidence.
