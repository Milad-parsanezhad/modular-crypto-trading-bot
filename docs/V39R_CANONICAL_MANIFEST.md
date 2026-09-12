# v0.39R Canonical Strategy Manifest

This manifest is the human-readable companion to `research_bot/canonical_strategy_v39r.py`.

| Component | State | Role in v0.39R | Evidence lineage |
|---|---|---|---|
| Ichimoku regime | ACTIVE_IN_RECONSTRUCTION | Context/regime | v0.17+ |
| CUSUM event sampling | ACTIVE_IN_RECONSTRUCTION | Event trigger | IRGC-S / v0.17 |
| Liquidity sweep | ACTIVE_IN_RECONSTRUCTION | Setup | ICT/SMC v0.19+ |
| BOS/CHoCH proxy | ACTIVE_IN_RECONSTRUCTION | Structure | ICT/SMC v0.19+ |
| FVG / imbalance | ACTIVE_IN_RECONSTRUCTION | Displacement context | ICT/SMC v0.19+ |
| Order-block mitigation proxy | ACTIVE_IN_RECONSTRUCTION | Retest context | ICT/SMC v0.19+ |
| Premium / discount | ACTIVE_IN_RECONSTRUCTION | Dealing-range location | ICT/SMC v0.19+ |
| Brooks-inspired bar confirmation proxy | ACTIVE_IN_RECONSTRUCTION | Closed-bar confirmation | Formalized in v0.39R |
| Triple barrier | CANDIDATE_GENERATOR | Outcome labeling / exit research | IRGC-S / v0.17 |
| Meta-labeling | FEATURE_ONLY | Secondary decision gate | v0.17 / v0.21+ |
| Expected net-R | FEATURE_ONLY | Economic gate | v0.36 |
| Conformal uncertainty | FEATURE_ONLY | Uncertainty gate | v0.36 |
| Drawdown firewall | FEATURE_ONLY | Portfolio safety | v0.31+ |
| Temporal consensus | REJECTED_UNDER_PROTOCOL | Negative result retained | v0.38 |
| Kraken holdout | UNFORMALIZED | Reserved external holdout | sealed |

## State semantics

- `FEATURE_ONLY`: retained as a feature/gate but not an independently promoted strategy.
- `CANDIDATE_GENERATOR`: may generate research events but cannot authorize execution.
- `ACTIVE_IN_RECONSTRUCTION`: explicitly part of the v0.39R canonical architecture.
- `REJECTED_UNDER_PROTOCOL`: failed its frozen protocol; retained as negative evidence.
- `SUPERSEDED`: replaced by a later audited implementation.
- `UNFORMALIZED`: not eligible for use until a preregistered definition exists.

## Non-negotiable governance

1. Updating a component state requires an evidence note or test result.
2. No state change can retroactively modify a frozen result.
3. Kraken remains sealed until a development candidate passes all frozen qualification gates.
4. `v0.39R` is a reconstruction release; it is not a live/paper promotion release.
5. Future v0.39 experiments must cite this manifest and explicitly state which canonical components they consume.
