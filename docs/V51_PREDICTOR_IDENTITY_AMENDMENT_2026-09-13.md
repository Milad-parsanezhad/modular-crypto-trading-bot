# v0.51 Predictor-Identity Amendment — 2026-09-13

Status: **PROSPECTIVE ENGINEERING/IDENTITY AMENDMENT BEFORE FIRST ELIGIBLE 4H DECISION**

Parent preregistration commit: `d8ee4576aaf55750dd5910cc0d3b2efcbba3f5b2`  
Parent verified head: `6a8fa49de2d1befa9c17049aa61e13da20a028eb`  
Parent verification run: `34738934441` — SUCCESS  
Parent scientific result: `V50_NONOVERLAP_FAILURE_SUPPORTED`

## Why an amendment is required

The v0.51 preregistration correctly freezes the arbitration arms, venues, assets,
costs, exits, Financial Governor and Expected-R admission rule. A repository audit
performed immediately after preregistration found one specification ambiguity that
must be closed before admissible prospective scoring begins:

> `exact canonical v0.47 C1 predictor` is not a single persisted production model.

The canonical v0.47 implementation fits and calibrates one model per
`fold × symbol`. Its evidence artifact contains 120 supported fold-symbol units,
OOS predictions and calibration diagnostics, but no serialized production model
bundle. Therefore a future observation cannot be scored uniquely until the
production instance is specified.

This is an identity/provenance defect, not an empirical finding. No admissible v0.51
economic outcome was inspected to discover or resolve it.

## Frozen repair

The production scoring identity is now fixed prospectively as:

`V51_PRODUCTION_PREDICTOR = V47_C1_LATEST_CANONICAL_FOLD_PER_ASSET`

For each of the five frozen assets (BTC, ETH, SOL, XRP, DOGE):

1. use **fold 5**, the latest canonical v0.47 fold by chronology;
2. reconstruct the exact v0.46 R1 multinomial base learner using the exact canonical
   v0.44 prepared dataset and frozen v0.47 feature list;
3. use only the fold-5 FIT partition for base-model/state-utility estimation;
4. fit the exact v0.47 C1 scalar temperature only on the fold-5 CALIBRATION partition;
5. verify the reconstructed fold-5 C1 probabilities / Expected-R against the
   canonical v0.47 fold-5 OOS artifact on the same event identities;
6. serialize the verified model, state means, temperature, feature order, class order,
   fold boundaries, dependency versions and source hashes as one immutable bundle;
7. compute SHA-256 digests for the bundle and all verification outputs;
8. all future v0.51 scoring must load this frozen bundle. Refit-on-the-fly is forbidden.

Fold 5 is chosen by a **pre-outcome chronological recency rule**, not by performance.
No fold performance statistic is used to select it.

## Prospective clock correction

The earlier `2026-09-13T08:00:00Z` boundary was frozen before the predictor-identity
ambiguity was found. Because the scoring identity was not yet unique at that instant,
that boundary is **superseded before any admissible 4h v0.51 decision**.

The repaired prospective clock begins at the first clean four-hour UTC boundary after
this identity freeze:

`2026-09-13T12:00:00Z`

The five 30-day blocks are shifted from that boundary. Pre-12:00 UTC observations may
be retained only as engineering smoke data and cannot enter v0.51 support or economics.

## Arbitration arms remain unchanged

This amendment does **not** change either research arm:

- A0: frozen earliest-first non-overlap control;
- A1: `SIMULTANEOUS_MAX_EXPECTED_R_NO_PREEMPTION`.

No threshold, asset, venue, side, event family, cost, exit, Financial Governor rule,
or risk ceiling changes.

## Additional integrity rules implementing the original preregistration text

The following are implementation hardening, not new economic gates:

- all A0/A1 comparisons must use the exact same candidate-identity set;
- candidate-set equality is verified by a deterministic SHA-256 identity digest;
- all 15 `venue × prospective_block` cells must be represented explicitly in support
  accounting; missing cells fail closed rather than disappearing;
- all numerical decision metrics must be finite; `NaN` and `±inf` fail closed;
- outcome columns cannot influence arbitration order;
- future candidate arrival cannot pre-empt an active position;
- Kraken remains sealed; PAPER=false; LIVE=false.

## Evidence boundary

This repair may verify software identity by replaying already-consumed v0.47 fold-5
predictions. That replay is **reproducibility evidence only**. It cannot support A1,
profitability, alpha or promotion.

No v0.51 economic read is permitted until the original support requirements, shifted
to the repaired 12:00 UTC boundary, are satisfied.

## Research rationale

The arbitration problem is a sequential scarce-capital allocation problem. The
literature on overlapping financial labels emphasizes that overlapping outcomes and
concurrent labels require explicit treatment; online portfolio-selection literature
likewise treats transaction costs and risk constraints as part of the allocation
problem rather than an afterthought. Recent crypto order-flow evidence also supports
keeping prediction and economically constrained portfolio construction distinct.

These sources motivate the research question only. They do not modify the frozen A0/A1
comparison after prospective evidence begins.

## Safety state

`RESEARCH_ONLY / KRAKEN_SEALED / PAPER_OFF / LIVE_OFF`

A successful identity-freeze workflow proves reproducibility of the scoring object. It
does not prove that the scorer or arbitration rule is profitable.
