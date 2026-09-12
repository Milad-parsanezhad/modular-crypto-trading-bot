# v0.45 — Post-v0.44 Decision Policy (Prospectively Frozen)

Status: **PREREGISTERED BEFORE v0.44 FINALIZATION**  
Purpose: prevent outcome-driven branching after the v0.44 information-driven / intrinsic-time sampling ablation.

## Scientific boundary

This document was created while the canonical v0.44 workflow run `34700944062` was still incomplete and before `finalize-v44` produced a scientific decision. No partial fold economics are admissible for choosing the path below.

Frozen v0.44 scientific head:

`5efb843d385d194c549a69dc4b2fd38d56ab906c`

Frozen variants:

1. `S0_CLOCK_MOTHER_BASELINE`
2. `S1_CUSUM_LAGGED_VOL`
3. `S2_DIRECTIONAL_CHANGE_LAGGED_ATR`

The v0.45 route may be selected only from the canonical finalized v0.44 artifact. Individual fold logs, partial artifacts, ad-hoc notebook inspection, or manually recomputed metrics must not be used as a routing signal.

## Invariants carried forward

The following remain locked unless a future experiment explicitly preregisters a new scientific question before observing its outcome:

- Mother Strategy and its ICT / SMC / Ichimoku / Al Brooks components;
- competing-risk labeling semantics;
- asset-specific HistGradientBoosting learner family;
- frozen transaction-cost treatment;
- Financial Governor and OOS economic/forecast-skill gates;
- purged walk-forward discipline and timestamp-cluster perturbation logic;
- no outcome-based asset pruning;
- no post-result threshold relaxation;
- no Transformer, PPO/RL, on-chain, order-book, sentiment or alternative-data expansion as a rescue response;
- Kraken remains sealed until the external-holdout route is formally activated;
- `PAPER=false` and `LIVE=false`.

## Route 0 — protocol validity gate

Before scientific interpretation, the canonical finalizer must establish that the v0.44 run is admissible.

If provenance, frozen-lineage hashes, common-fold identity, fail-closed manifests, data-source guards, or any other preregistered integrity condition fails, the result is:

`V44_INVALID_SCIENTIFIC_RUN`

Action:

- do not interpret economics;
- do not alter thresholds, features, labels, costs or model capacity;
- repair only the implementation/provenance defect;
- rerun the same frozen v0.44 scientific contract on a prospectively audited head;
- v0.45 remains inactive.

## Route A — event-driven sampling earns external validation

Activation condition:

- at least one of `S1_CUSUM_LAGGED_VOL` or `S2_DIRECTIONAL_CHANGE_LAGGED_ATR` satisfies **all** frozen v0.44 promotion gates in the canonical final artifact; and
- the promoted event-driven variant satisfies the frozen comparison requirement against `S0_CLOCK_MOTHER_BASELINE` defined by v0.44.

Selection rule:

- if exactly one event-driven variant qualifies, freeze that variant;
- if both qualify, use the deterministic tie-break already encoded by the v0.44 finalizer/preregistration; no new metric may be invented after results are known;
- if the frozen v0.44 contract has no deterministic tie-break, do **not** choose manually: classify as `V44_AMBIGUOUS_PROMOTION` and preregister a separate head-to-head experiment on development venues only.

Next experiment:

`v0.45 — One-Shot Kraken External Holdout`

Required design:

- freeze the winning event clock and every upstream/downstream component;
- create a new immutable preregistration commit and hash before Kraken is read;
- one canonical Kraken holdout read only;
- no same-holdout rescue tuning;
- exact provenance, manifest and artifact hashing;
- economic and forecast-skill gates copied prospectively from the frozen development contract unless an independently justified external-holdout gate was already preregistered;
- a Kraken failure terminates that candidate on the spent holdout.

Successful Kraken performance is **external validation evidence**, not LIVE authorization.

## Route B — sampling hypothesis improves signal but fails promotion

Activation condition:

- an event-driven variant shows the preregistered directional improvement over S0, but fails one or more frozen promotion gates; or
- the event-driven variant is scientifically interesting but too sparse to satisfy the frozen evidence requirement.

Interpretation:

This is not permission to lower the v0.44 threshold after the fact.

Next experiment:

`v0.45 — Sampling Density / Stability Characterization`

Constraints:

- Kraken remains sealed;
- use development venues only;
- define all candidate sampling thresholds **before** executing the new experiment;
- use a small, explicitly bounded sensitivity set rather than an open search;
- preserve the same mother strategy, labels, learner, costs and governor;
- evaluate event density, class/event coverage, forecast skill, perturbation stability and portfolio economics jointly;
- use common chronological folds again where mathematically feasible;
- sparse configurations fail closed;
- no threshold from this experiment may be selected using Kraken.

This route tests whether the failure was evidence scarcity / event-density mismatch rather than a lack of information advantage.

## Route C — all sampling clocks fail forecast-skill / economic promotion

Activation condition:

- S0, S1 and S2 all fail the frozen v0.44 promotion standard; especially if forecast-skill remains non-positive across the experiment.

Interpretation:

The research program should stop treating the sampling clock as the primary bottleneck.

Next experiment:

`v0.45 — Label / Representation Diagnostic`

Scope, in order:

1. competing-risk target separability and censoring diagnostics;
2. calibration decomposition and Brier reliability/resolution analysis;
3. feature information content conditional on market regime and event age;
4. label horizon / barrier interaction diagnostics without changing the trading policy;
5. only after these diagnostics, a separately preregistered representation experiment may be considered.

Explicitly forbidden as an immediate rescue:

- Transformer / Decision Transformer;
- PPO or other RL;
- feature explosion;
- on-chain, sentiment or order-book additions;
- post-hoc asset deletion;
- cost/gate relaxation.

Reason: more model capacity is not a defensible response when the existing target/representation may lack stable predictive information.

## Route D — clock baseline remains scientifically adequate while event clocks do not improve it

Activation condition:

- `S0_CLOCK_MOTHER_BASELINE` satisfies the frozen adequacy standard, but neither S1 nor S2 earns promotion over S0.

Decision:

`INTRINSIC_TIME_HYPOTHESIS_NOT_PROMOTED`

Action:

- retain S0 as the research baseline;
- do not send S1/S2 to Kraken;
- move to the same label/representation diagnostic discipline described in Route C unless another preregistered non-sampling hypothesis already has priority.

## External data firewall

Kraken must remain unread by the v0.44/v0.45 development code until Route A has been activated by the canonical finalized v0.44 decision.

Any accidental pre-activation Kraken read invalidates Kraken as a clean holdout for that candidate and must be recorded as a scientific contamination event.

## Decision record required after v0.44 finalization

Exactly one post-finalization record must be created containing:

- canonical v0.44 workflow run ID;
- exact scientific head SHA;
- final artifact ID and SHA-256 manifest digest;
- finalizer decision code;
- selected route: `INVALID`, `A`, `B`, `C`, or `D`;
- selected variant, if and only if prospectively allowed;
- statement that no partial fold economics were used for routing;
- statement that Kraken remained sealed through route selection;
- `PAPER=false`;
- `LIVE=false`.

## Promotion ladder after this policy

The research ladder remains conservative:

`development ablation -> frozen candidate -> one-shot external holdout -> robustness/search-aware review -> prospective PAPER observation -> only then reconsider execution authorization`

No stage is automatically promoted because the previous stage produced a positive return.

## Current state at policy freeze

`V44_RUNNING / V45_POLICY_FROZEN / KRAKEN_SEALED / PAPER_OFF / LIVE_OFF`
