# v0.17 Thesis Evidence Synthesis

Date: 2026-09-09

Status: **Scientific synthesis / defense integration — no execution promotion**

## 1. Why v0.17 exists

v0.17 is not another strategy-optimization round. It converts the completed empirical path from v0.10 through v0.16 into a single auditable scientific narrative for the thesis and defense. The governing principle remains **Evidence Before Opinion**.

The central question is no longer merely “which model had the highest return?” The stronger question is:

> Which conclusions survive increasingly strict out-of-sample, robustness, cost, multiple-testing, untouched-holdout and prospective-forward controls?

That distinction is essential because a trading thesis can be technically impressive while still being scientifically weak if it selects favorable backtests after repeated search.

## 2. Evidence ladder

| Version | Scientific role | Main result | Claim status |
|---|---|---|---|
| v0.10 | Purged live-universe OOS tournament | `NO_MODEL_PROMOTED` | Learned models did not establish stable superiority over simple baselines |
| v0.11 | Multi-seed robustness + moving-block bootstrap + FDR + regime analysis | `NO_MODEL_PROMOTED` | Stronger inference rejected promotion and exposed regime concentration |
| v0.12 | External untouched derivatives/order-flow holdout | `NO_INCREMENTAL_DERIVATIVES_EVIDENCE` | Tested derivatives family did not add sufficient net alpha in the frozen specification |
| v0.13 | New prospective microstructure source/window | `FORWARD_COLLECTION_ACTIVE` | New hypothesis collection without tuning the spent holdout |
| v0.14 | Production paper runtime | `PAPER_RUNTIME_OPERATIONAL` | Engineering integration demonstrated; not alpha evidence |
| v0.15 | Scheduled immutable forward evidence | `INSUFFICIENT_FORWARD_SAMPLE` | Prospective evidence collection active under pre-registered minimums |
| v0.16 | Formal forward evaluation + defense package | `INSUFFICIENT_FORWARD_SAMPLE` | Reporting pipeline validated; profitability/alpha/LIVE claims remain prohibited |

## 3. What v0.10 actually taught us

v0.10 used a 4-hour, three-fold purged expanding OOS tournament with explicit one-way cost of 12 bps. The strongest aggregate learned challenger was Logistic Regression, but it failed fold stability and Sharpe superiority. The Ichimoku baseline had the strongest aggregate return in that pilot, yet its fold results varied materially.

The correct conclusion was therefore not “Ichimoku wins” or “ML works.” It was that aggregate performance alone was insufficient and that the experiment required stronger robustness controls.

## 4. What v0.11 changed

v0.11 strengthened the design using three seeds, paired moving-block bootstrap, Benjamini-Hochberg FDR correction, coverage matching and point-in-time regime diagnostics.

The strongest learned candidate, Logistic Regression, underperformed the strongest baseline, Ichimoku, on 729 common OOS periods. The moving-block bootstrap interval for the mean Logistic-minus-Ichimoku edge was entirely below zero, and the one-sided evidence for a positive Logistic edge failed strongly.

The deeper finding was regime dependence. The Ichimoku baseline was positive in HIGH_VOL and TREND_DOWN buckets and negative in RANGE and TREND_UP. Therefore regime dependence became a new hypothesis, not a post-hoc filter to be tuned on the same sample.

## 5. What v0.12 falsified

v0.12 moved to an external, untouched holdout and asked whether point-in-time derivatives information added incremental value. The frozen feature families included price, Ichimoku, funding, premium/basis proxy, futures taker flow and lagged open-interest variables.

None of the learned variants produced positive net holdout performance. The full Logistic model did not outperform price-only on the primary promotion comparison, and the decision remained:

`NO_INCREMENTAL_DERIVATIVES_EVIDENCE`

This is a high-value negative result. It shows that literature support for a feature family does not automatically translate into tradable alpha under a specific implementation, cost model and holdout.

## 6. Why v0.13-v0.16 are scientifically different from backtesting

After v0.12, the project deliberately stopped tuning the consumed holdout. v0.13 opened a new prospective microstructure collection window. v0.14 built a production PAPER engine with persistent state and independent risk. v0.15 created scheduled immutable evidence artifacts. v0.16 added independence filtering and formal reporting.

This creates a clean separation between:

1. historical research evidence;
2. prospective engineering operation;
3. prospective forward performance evidence.

The current prospective sample is still insufficient for a profitability or alpha claim.

## 7. Current claim policy

The following statements are supported:

- the research code is operational and reproducible;
- the project implements purged OOS, explicit costs, robustness inference, an untouched holdout and prospective evidence collection;
- the paper runtime is operational with persistent state and independent risk controls;
- negative findings are retained rather than optimized away;
- no learned model has yet earned promotion to real-money execution.

The following statements are **not** currently supported:

- the strategy is profitable in expectation;
- the project has statistically significant alpha;
- the observed historical Ichimoku performance is universal or regime-stable;
- the derivatives feature family is generally useless;
- Sharpe/Sortino are stable in prospective operation;
- the bot is ready for real-money LIVE execution.

## 8. Deeper methodological upgrade introduced by v0.17

The next formal candidate that ever seeks promotion must be evaluated under a **search-aware validation stack**, not only a conventional backtest.

The recommended stack is:

1. strict point-in-time feature availability;
2. purged/embargoed OOS or untouched holdout;
3. explicit fee, spread, slippage and turnover cost;
4. paired dependence-aware bootstrap;
5. multiple-testing control across the candidate search;
6. White Reality Check or Hansen Superior Predictive Ability when comparing many searched strategies;
7. Probabilistic/Deflated Sharpe analysis to account for finite samples, non-normality and selection bias;
8. Probability of Backtest Overfitting / CPCV where the experimental design permits it;
9. minimum track-record analysis before declaring risk-adjusted skill;
10. prospective forward replication before any live promotion.

## 9. Literature anchors for the stricter audit

- White, H. (2000), *A Reality Check for Data Snooping*, Econometrica 68(5), 1097–1126. DOI: `10.1111/1468-0262.00152`.
- Hansen, P. R. (2005), *A Test for Superior Predictive Ability*, Journal of Business & Economic Statistics 23(4), 365–380. DOI: `10.1198/073500105000000063`.
- Bailey, D. H. & López de Prado, M. (2014), *The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality*, Journal of Portfolio Management 40(5), 94–107. DOI: `10.3905/jpm.2014.40.5.094`.
- Bailey, D. H. & López de Prado, M., *The Sharpe Ratio Efficient Frontier*, Journal of Risk 15(2), supporting Probabilistic Sharpe Ratio and minimum track-record reasoning.
- Sullivan, R., Timmermann, A. & White, H. (1999), *Data-Snooping, Technical Trading Rule Performance, and the Bootstrap*, Journal of Finance 54(5), 1647–1691. DOI: `10.1111/0022-1082.00163`.

Recent crypto-specific research also reinforces two design choices of this thesis: transaction costs can erase otherwise positive forecast performance, and richer order-flow data can be more informative than simply increasing model complexity. These findings motivate better inputs and stricter execution filters, but they do not override the project’s own negative holdout evidence.

## 10. Thesis-level synthesis

The strongest contribution of the project is not a claim that one AI model always wins. It is the construction of a modular research-to-execution system in which increasingly strict evidence gates determine whether complexity is allowed to progress.

The empirical sequence itself is a result:

- a superficially stronger learned candidate failed stronger robustness tests;
- a strong historical technical baseline proved regime-dependent;
- adding derivatives did not produce sufficient incremental holdout alpha;
- prospective PAPER infrastructure works, but the forward sample is not yet mature.

This is scientifically stronger than selecting the best backtest after the fact.

## 11. Official v0.17 conclusion

> Through v0.17, the project demonstrates a functioning and auditable cryptocurrency trading-research system, but it does not yet demonstrate stable tradable alpha. The evidence path from v0.10 to v0.16 shows that increasingly strict validation removes several apparently attractive conclusions. The project therefore retains negative results, freezes consumed holdouts, withholds unstable risk metrics, and keeps real-money execution fail-closed until a future candidate survives search-aware statistical validation and prospective replication.
