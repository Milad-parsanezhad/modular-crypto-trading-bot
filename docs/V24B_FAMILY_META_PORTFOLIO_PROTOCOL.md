# v0.24b — Strategy-Family Meta-Labeling + Overlap-Aware Portfolio Risk

## Purpose

This experiment is the direct follow-up to v0.24. It asks two narrower questions:

1. Does a **separate meta-labeling model per strategy family** rank that family's events more effectively than one pooled meta-model?
2. What happens when accepted events are passed through an **overlap-aware portfolio risk engine** instead of naive sequential event compounding?

## Critical validity constraint

The v0.24 terminal CoinEx period was already inspected before v0.24b was designed. Therefore the repeated terminal segment is explicitly labeled **SHADOW / REUSED TEST**. It is useful for engineering diagnostics only and **cannot authorize scientific promotion, Forward PAPER replacement, or LIVE execution**.

Any future promotion requires fresh forward-time observations or an untouched external-venue replication of models and thresholds frozen before that evidence is read.

## Frozen universe

The same seven 4h strategy families from v0.24 are generated:

- `H4_S6_BREAKOUT`
- `H4_KUMO_TRIANGLE`
- `H4_OB_BOS_RETEST`
- `H4_SUPPLY_DEMAND`
- `H4_CORRELATION_DIVERGENCE`
- `H4_D1_S6_VOL_RISK`
- `H4_D1_OB_BOS_RISK`

Family-model eligibility depends only on sample counts:

- development >= 300 events;
- validation >= 80 events;
- shadow >= 100 events.

No outcome metric is allowed to determine eligibility. Sparse families remain `DATA_INSUFFICIENT`.

## Family-specific ML contract

For each eligible family:

- features remain causal `f_*` market/structure/context features from v0.24;
- strategy and symbol identity remain excluded from X;
- constant `f_strategy_*` descriptors are removed inside a family;
- development only is used for fitting;
- model family, seed and threshold are selected on validation only;
- the reused terminal segment is read only after the family champion is frozen;
- the terminal result is labeled SHADOW and has `scientific_promotion_authorized=false`.

The frozen tabular registry and seeds are inherited from v0.23r/v0.24. Threshold search is validation-only.

## Portfolio overlap/risk engine

The portfolio engine processes actual overlapping entry/exit intervals and enforces:

- risk per accepted trade: 0.25% of current realized equity;
- one active position per symbol;
- maximum portfolio open risk: 1.00%;
- maximum strategy open risk: 0.50%;
- maximum same-direction open risk: 0.75%;
- maximum concurrent positions: 5;
- hard realized-equity drawdown kill: 5%.

Entry priority for simultaneous events is deterministic (`entry_time`, `strategy`, `symbol`) and does not depend on future outcomes or ML score.

The engine is overlap-aware but not full intrabar mark-to-market. Drawdown is measured on realized equity after recorded bracket exits. A later stage must replay open positions against point-in-time OHLC to obtain true MTM portfolio drawdown.

## Costs and execution semantics

The event labels inherit the v0.24/v0.19 conservative bracket simulator:

- signal at closed bar t;
- entry at next open;
- stop and target are horizontal barriers;
- `max_hold_bars` is the vertical barrier;
- same-bar stop/target collision is stop-first;
- baseline friction is 24 bps round trip (10 bps fee + 2 bps slippage each way).

## Outputs

The workflow persists:

- strategy event dataset and provenance;
- family sample counts and eligibility;
- family validation leaderboard;
- family frozen champions;
- family shadow predictions;
- family shadow ablation;
- base and family-filter portfolio ledgers;
- overlap-aware portfolio summary;
- fail-closed decision.json.

## Decision semantics

A green CI run proves implementation/provenance integrity, not alpha.

The only permitted v0.24b scientific status is:

`EXPLORATORY_REUSED_SHADOW_NO_PROMOTION`

and the following remain false:

- `forward_paper_authorized`
- `paper_replacement_authorized`
- `live_execution_authorized`
