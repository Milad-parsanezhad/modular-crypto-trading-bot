# v0.51 Overlap-Conflict Arbitration — Research Note (2026-09-13)

Status: **POST-PREREGISTRATION LITERATURE CONTEXT ONLY — NO A0/A1 DESIGN CHANGE**

This note documents literature reviewed after the immutable v0.51 A0/A1 design was frozen. It does **not** change the arbitration arms, support thresholds, decision gates, venues, assets, costs, exits, Financial Governor or safety state.

A separate pre-outcome engineering amendment was required because the phrase `exact canonical v0.47 C1 predictor` did not identify a single persisted production model. That amendment freezes a unique fold-5 C1 model bundle per allowed asset and moves the admissible prospective boundary to the first clean four-hour UTC boundary after that identity repair.

## Frozen scientific lineage

- Parent result: `V50_NONOVERLAP_FAILURE_SUPPORTED`
- Immutable v0.51 preregistration commit: `d8ee4576aaf55750dd5910cc0d3b2efcbba3f5b2`
- Immutable preregistration time: `2026-09-13T04:49:28Z`
- Frozen calendar commit: `6a8fa49de2d1befa9c17049aa61e13da20a028eb`
- Original boundary: `2026-09-13T08:00:00Z` — superseded before an admissible 4h scoring decision because predictor identity was not yet unique
- Amended admissible boundary: `2026-09-13T12:00:00Z`
- Policy under test: `SIMULTANEOUS_MAX_EXPECTED_R_NO_PREEMPTION`
- Kraken: sealed
- PAPER/LIVE: disabled

## 1. Overlap is a statistical and allocation problem, not merely a code detail

López de Prado, *Advances in Financial Machine Learning*, Chapter 4, treats overlapping outcomes, concurrent labels and average uniqueness explicitly. Financial events that overlap in time cannot be interpreted as IID opportunities, and concurrency affects both effective sample information and return attribution.

For v0.51 this supports preserving a single `venue × symbol` slot and comparing causal conflict-resolution rules instead of pretending every overlapping candidate can be executed independently.

## 2. Online interval scheduling with predictions

### Antoniadis, Shahheidar, Shahkarami & Soltani — AAAI 2026
**A Switching Framework for Online Interval Scheduling with Predictions**. DOI `10.1609/aaai.v40i43.40930`.

The paper studies irrevocable online interval scheduling with prediction advice. It separates forecast information from the causal scheduling rule and emphasizes robustness when predictions are imperfect. That is closely aligned with the thesis architecture: v0.47 supplies frozen predictive information; v0.51 evaluates only how simultaneous conflicts consume that information.

### Boyar et al. — Journal of Computer and System Sciences, 2026
**Online interval scheduling with predictions**.

This line of work reinforces the same mechanism: prediction quality does not by itself solve the scheduling problem. A causal allocation policy still determines which mutually incompatible opportunity is retained.

## 3. Predictive quality and downstream decision quality are distinct

### Kong et al. — UAI / PMLR 2025
**DF²: Distribution-Free Decision-Focused Learning**, PMLR 286:2269–2290.

Decision-focused learning distinguishes forecast/model error from downstream optimization error. This is directly relevant to v0.50: Expected-R admission could improve an intermediate population while a later non-overlap transformation still damage realized economics.

v0.51 therefore holds the predictor fixed and isolates the arbitration layer rather than refitting the forecast model together with the allocator.

## 4. Transaction costs and risk constraints belong inside portfolio selection

Guo, Gu, Fok & Ching, *European Journal of Operational Research* (2023), **Online portfolio selection with state-dependent price estimators and transaction costs**, DOI `10.1016/j.ejor.2023.05.001`, explicitly combines sequential portfolio selection with transaction costs and a risk-parity constraint.

Guo et al., *Quantitative Finance* (2024), **Adaptive online mean-variance portfolio selection with transaction costs**, DOI `10.1080/14697688.2023.2287134`, likewise treats prediction and allocation under trading friction rather than evaluating forecasts in isolation.

The consequence for v0.51 is conservative: Expected-R arbitration is not allowed to claim improvement by relaxing the already-frozen transaction-cost assumptions or the Financial Governor. A1 must be evaluated under the same economic/risk constraints as A0.

## 5. Crypto-specific evidence supports separating information from portfolio construction

Anastasopoulos, Gradojevic, Liu, Maynard & Tsiakas (2026), *Journal of Financial Markets*, **Order flow and cryptocurrency returns**, DOI `10.1016/j.finmar.2026.101047`, report out-of-sample predictive information in international/world cryptocurrency order flow and evaluate portfolio economics, turnover, long-only constraints and transaction-cost robustness.

This does not validate v0.51 or its Expected-R model. It supports the broader methodological distinction already observed inside this repository: useful predictive/event information can still be lost or transformed by portfolio construction, overlap resolution and execution constraints.

## 6. Why v0.51 deliberately remains a two-arm simple experiment

The frozen comparison stays:

1. **A0 — earliest-first control**;
2. **A1 — maximum frozen Expected-R among simultaneous same-entry candidates while flat, no pre-emption**.

No RL allocator, Transformer, optimizer sweep, correlation penalty or CVaR-weighted utility is added now. Adding those after seeing v0.50 would create an attribution problem and enlarge the researcher degrees of freedom.

## 7. Future hypotheses explicitly deferred to a new experiment

Only after the v0.51 prospective sample is spent may a separately preregistered stage consider:

- Expected-R minus incremental CVaR penalty;
- correlation-aware capital-locking / diversification utility;
- uncertainty-adjusted Expected-R lower confidence bound;
- constrained mean-variance or risk-parity arbitration;
- contextual bandit or reinforcement-learning allocation.

These are **not** v0.51 rescue options.

## 8. Predictor identity is part of scientific reproducibility

The v0.47 C1 experiment produced 120 supported `fold × symbol` evaluation units, not one serialized production predictor. Therefore a statement such as “use the exact v0.47 C1 predictor” was insufficient for unique forward inference.

The identity amendment fixes this without using v0.51 outcomes:

- fold 5 chosen solely by chronological recency;
- one base model per frozen v0.51 asset;
- canonical fold-5 temperature parameter;
- FIT-only state means;
- replay verification against canonical fold-5 C1 OOS predictions;
- serialized immutable model bundle + SHA-256 manifest;
- no future refit allowed.

That repair is a provenance prerequisite, not an optimization.

## Consequence for v0.51

1. A0 remains exact earliest-first non-overlap.
2. A1 may use frozen Expected-R only among candidates with the **same entry time while the slot is flat**.
3. No active trade may be pre-empted by a later candidate.
4. Realized `net_r`, outcome, future paths or future candidates may not enter the priority key.
5. A0 and A1 must consume exactly the same candidate-identity set.
6. Missing `venue × block` evidence fails closed rather than being silently removed.
7. Non-finite decision metrics fail closed.
8. No policy threshold or hyperparameter will be searched after prospective outcomes become available.
9. Kraken remains sealed; PAPER and LIVE remain disabled.

The purpose of this note is traceability and hypothesis rationale, not post-result design modification.
