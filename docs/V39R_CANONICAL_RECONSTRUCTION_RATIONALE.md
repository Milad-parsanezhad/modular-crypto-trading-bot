# v0.39R — Canonical Strategy Reconstruction Rationale

Status: **pre-v0.39 reconstruction / research-only**  
Live execution: **disabled**  
Reserved holdout: **Kraken remains sealed**

## Why this document exists

Between v0.17 and v0.38 the project accumulated a large amount of useful strategy logic, but the scientific development process progressively decomposed the original "mother strategy" into isolated experiments. This was intentional for causal attribution and falsifiability, but it created a documentation problem: a reader following the repository could see many increasingly specialized experiments without one canonical map showing how the original trading thesis related to them.

v0.39R fixes that documentation and architecture gap **before** any new v0.39 hypothesis is evaluated.

## Root cause analysis: why the mother strategy fragmented from v0.17 to v0.38

### 1. Experimental decomposition replaced architectural continuity

v0.17 still contained a recognizable rule-based strategy laboratory: Ichimoku states, S1–S6 families, IRCP/C2 event generation, CUSUM sampling, triple-barrier outcomes, meta-labeling, cost sensitivity, and risk controls. Later stages increasingly tested one scientific failure mode at a time (cross-venue robustness, portfolio admission, drawdown control, breadth, uncertainty, temporal stability). This was methodologically sound because changing many components simultaneously would destroy attribution.

**Consequence:** the scientific lineage remained valid, but the strategy lineage became difficult to read.

### 2. Promotion gates correctly rejected components, but rejected != forgotten

Many ICT/SMC/Ichimoku candidates were not promoted after out-of-sample, breadth, cost, confidence-interval, and replication gates. The repository therefore moved forward using only the surviving evidence from each stage. However, there was no canonical registry distinguishing:

- implemented but not promoted,
- falsified under a specific protocol,
- still useful as a feature,
- still useful as a candidate generator,
- superseded,
- or not yet formally specified.

**Consequence:** readers can mistake "not in the latest experiment" for "removed from the project."

### 3. Research proxies and trading semantics were mixed across modules

Concepts such as BOS, liquidity sweeps, FVG, order blocks, premium/discount, and Ichimoku states were encoded as causal research proxies in different modules. This is scientifically preferable to subjective chart hindsight, but the project did not maintain one explicit translation layer from discretionary trading language to objective features.

**Consequence:** semantic continuity was weaker than implementation continuity.

### 4. Failure-driven development shifted the optimization target

The research target changed over time:

1. find profitable candidate rules,
2. control costs and drawdown,
3. replicate across venues,
4. correct portfolio admission,
5. improve breadth,
6. model expected net utility and uncertainty,
7. test temporal/model consensus stability.

By v0.38, the main bottleneck was no longer raw average profitability. The strongest observed candidate still failed temporal robustness because the moving-block confidence interval lower bound and positive-quarter persistence were insufficient. This made it scientifically invalid to simply re-enable old strategy components without re-testing them inside the frozen evaluation protocol.

### 5. The project optimized falsifiability, not presentation

From a data-science perspective, the sequence v0.17→v0.38 was a chain of increasingly strict falsification tests. That is desirable for a thesis. The missing piece was a **canonical architecture document and executable reconstruction** that preserves the original strategy thesis while keeping all later scientific safeguards.

## Scientific interpretation of the v0.17→v0.38 path

The correct interpretation is **not** "the strategy kept changing until it worked." The correct interpretation is:

> A broad hypothesis family was progressively decomposed into testable components, subjected to increasingly realistic transaction-cost, causal, portfolio, cross-venue, uncertainty, and temporal-stability constraints, and repeatedly rejected when it failed a frozen gate.

This is a stronger research narrative than a single optimized backtest because it records negative results and prevents silent post-hoc tuning.

## Canonical strategy layers restored by v0.39R

v0.39R reconstructs the project's mother strategy as an explicit pipeline while keeping each layer auditable:

1. **Regime / context**
   - Ichimoku 9/26/52 causal state
   - long-term trend context
   - premium/discount location

2. **Event trigger**
   - CUSUM-style event logic (legacy IRGC-S lineage)
   - liquidity sweep and structural events

3. **ICT/SMC structure**
   - BOS / CHoCH proxy
   - liquidity sweeps
   - FVG / imbalance
   - order-block / mitigation proxy
   - premium/discount / dealing range

4. **Price-action confirmation**
   - objective bar-by-bar confirmation proxies inspired by Brooks-style price action
   - no discretionary chart labels are treated as ground truth

5. **Economic / ML gate**
   - meta-labeling / expected net-R / uncertainty remain research gates
   - no component is promoted merely because it increases in-sample return

6. **Risk / portfolio layer**
   - stop-distance sizing
   - aggregate exposure controls
   - drawdown firewall
   - transaction costs and stress costs

7. **Exit / outcome layer**
   - triple-barrier / structural invalidation / duration-aware evaluation

## What v0.39R does NOT do

- It does not authorize live trading.
- It does not unseal Kraken.
- It does not relax any frozen robustness gate.
- It does not claim ICT/SMC or Brooks terminology is academic ground truth; these are formalized research proxies.
- It does not overwrite v0.38 results.
- It does not promote a candidate before fresh development-only evaluation.

## Governance rule going forward

Every strategy component must have one of the following states in the canonical manifest:

- `FEATURE_ONLY`
- `CANDIDATE_GENERATOR`
- `ACTIVE_IN_RECONSTRUCTION`
- `REJECTED_UNDER_PROTOCOL`
- `SUPERSEDED`
- `UNFORMALIZED`

No future version may silently drop or revive a component without updating the manifest and documenting the evidence that changed its state.

## Why v0.39R precedes v0.39

The purpose of v0.39R is reconstruction and auditability, not hypothesis fishing. Only after the canonical strategy is reconstructed and tested as a deterministic, leakage-safe research object should v0.39 evaluate a **new preregistered learning-process hypothesis** (e.g., regime-shift-aware purged walk-forward temporal re-estimation) on consumed development venues only.

This separation prevents architecture repair from being confused with performance tuning.
