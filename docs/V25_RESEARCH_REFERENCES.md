# v0.25 Research References and Comparable-System Audit

Frozen before v0.25 future-time evidence is consumed.

## Scientific references

1. Hu, W., Zhou, J. (2024). *Trading Signal Survival Analysis: A Framework for Enhancing Technical Analysis Strategies in Stock Markets*. Computational Economics 64, 3473–3507. DOI: `10.1007/s10614-024-10567-8`.
2. *Momentum portfolio selection based on learning-to-rank algorithms with heterogeneous knowledge graphs*. Applied Intelligence 54 (2024). DOI: `10.1007/s10489-024-05377-2`.
3. Arian, H., Norouzi Mobarekeh, D., Seco, L. (2024). *Backtest overfitting in the machine learning era: A comparison of out-of-sample testing methods in a synthetic controlled environment*. Knowledge-Based Systems 305, 112477. DOI: `10.1016/j.knosys.2024.112477`.
4. Kruthof, G., Müller, S. (2025). *Can deep reinforcement learning beat 1/N?* Finance Research Letters 75, 106866. DOI: `10.1016/j.frl.2025.106866`.
5. Jiang, Y., Olmo, J., Atwi, M. (2025). *High-dimensional multi-period portfolio allocation using deep reinforcement learning*. International Review of Economics & Finance 98, 103996. DOI: `10.1016/j.iref.2025.103996`.
6. Kato, M. (2024). *Conformal Predictive Portfolio Selection*. arXiv:2410.16333. Used as a hypothesis source for later uncertainty/lower-bound portfolio admission; not treated as proof of trading alpha.

## Comparable-system bug audit

### FinRL CCXT timezone/index drift

Public FinRL issue #1440 documents a CCXT processor path in which naive local-time conversion can shift exchange UTC timestamps and corrupt cross-asset alignment. v0.25 therefore retains timezone-aware UTC normalization and treats timestamp/index consistency as a testable data contract.

Reference: `https://github.com/AI4Finance-Foundation/FinRL/issues/1440`

### Qlib chained dependency/index failures

Public Qlib issue #2233 documents a sequence of compatibility failures involving LightGBM 4+, unhashable segment lists and pandas MultiIndex behavior. The key engineering lesson is that one blocked dependency bug can mask later failures. v0.25 therefore uses fail-fast dependency checks, exact persisted-model environment replay and separate regression gates instead of continuing after partial incompatibility.

Reference: `https://github.com/microsoft/qlib/issues/2233`

## Project-specific lessons carried forward

- exact model binary + feature schema + threshold + environment + dataset hash is the scientific model identity;
- a successful classifier is not automatically a successful portfolio allocator;
- spent external evidence cannot be rebranded as a fresh test after redesign;
- future RL must beat simple allocation baselines after costs and risk, not merely raw-return reward curves;
- all promotion claims require portfolio-level economic evidence, not accuracy/AUC alone.
