# v0.51 Overlap-Conflict Arbitration — Research Note (2026-09-13)

Status: **POST-PREREGISTRATION LITERATURE CONTEXT ONLY — NO DESIGN CHANGE**

This note documents fresh literature reviewed after the immutable v0.51 design was already frozen. It does **not** amend the preregistered A0/A1 rules, calendar, support thresholds, decision gates, venues, assets, costs, Financial Governor, predictor, or safety state.

## Frozen scientific lineage

- Parent result: `V50_NONOVERLAP_FAILURE_SUPPORTED`
- Immutable v0.51 preregistration commit: `d8ee4576aaf55750dd5910cc0d3b2efcbba3f5b2`
- Immutable preregistration time: `2026-09-13T04:49:28Z`
- Frozen calendar commit: `6a8fa49de2d1befa9c17049aa61e13da20a028eb`
- Prospective start: `2026-09-13T08:00:00Z`
- Policy under test: `SIMULTANEOUS_MAX_EXPECTED_R_NO_PREEMPTION`

## Fresh literature

### Antoniadis, Shahheidar, Shahkarami & Soltani (AAAI 2026)
**A Switching Framework for Online Interval Scheduling with Predictions**. DOI: `10.1609/aaai.v40i43.40930`.

The paper studies irrevocable online interval scheduling where each arriving interval must be accepted or rejected while preserving non-overlap. Its learning-augmented framing separates the value of predictions from robustness to prediction error. This directly supports treating overlap resolution as a distinct decision layer rather than as part of probability-model fitting.

### Boyar et al. (Journal of Computer and System Sciences, 2026)
**Online interval scheduling with predictions**.

This work studies interval scheduling with prediction advice in an online setting and reinforces the same methodological point: predictive information can inform conflict resolution, but the scheduling policy must remain causal and its robustness must be evaluated separately from forecast quality.

### Kong et al. (UAI / PMLR 2025)
**DF²: Distribution-Free Decision-Focused Learning**, PMLR 286:2269–2290.

The paper emphasizes that predictive-model quality and downstream optimization quality are distinct objects. Model mismatch, finite-sample approximation, and optimization error can each degrade the final decision even when a forecaster appears useful. For this thesis, that supports the v0.50→v0.51 separation between frozen Expected-R estimation and overlap arbitration.

## Consequence for v0.51

The literature is consistent with the already-frozen design, but it does not justify changing it after preregistration. Therefore:

1. A0 remains exact earliest-first non-overlap.
2. A1 may use frozen `expected_r_v47` only among candidates with the **same entry time while the slot is flat**.
3. No active trade may be pre-empted by a later candidate.
4. Realized `net_r`, outcome, future paths, or future candidates may not enter the priority key.
5. No policy threshold or hyperparameter will be searched after prospective outcomes become available.
6. Kraken remains sealed; PAPER and LIVE remain disabled.

The purpose of this note is traceability, not design modification.
