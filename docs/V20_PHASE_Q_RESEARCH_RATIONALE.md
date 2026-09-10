# v0.20 Phase-Q Research Rationale

Date: 2026-09-10

## Why this stage exists

The previous stages showed that apparent trading improvements can disappear under stronger out-of-sample, cost and inference controls. The next bottleneck is therefore not model complexity but the fidelity and maturity of the new microstructure evidence stream.

The v0.20 Phase-Q design deliberately treats data quality as a separate scientific gate. It answers whether the cross-venue REST measurements are sufficiently synchronized and complete before they are allowed to influence a 4-hour feature specification.

## Why immutable GitHub artifacts are used

A rolling file would be convenient but scientifically weaker because later runs could overwrite prior evidence. Per-run GitHub Actions artifacts preserve the run-level output and provide artifact metadata and digests. A separate harvester reconstructs the eligible evidence set for each Phase-Q report.

## Why scheduled-main-only evidence counts

Manual and pull-request runs are useful for engineering smoke tests but are not independent prospective observations. Allowing them to count would let an operator increase the sample size by repeatedly pressing a button. Phase Q therefore counts only scheduled runs from the protected research line (`refs/heads/main`).

## Why actual timestamps remain authoritative

A GitHub cron expression specifies the intended trigger cadence, not a guaranteed wall-clock execution instant. Queueing and runner availability can delay execution. The project stores and evaluates actual timestamps, reports missing measurement slots and refuses to synthesize observations for missed runs.

## Why Gate Q cannot directly start model training

Passing a data-quality gate after seven days is not evidence that the data contain alpha. The next feature family, target definition, model set and statistical comparison must be frozen separately before target-aware evaluation. This prevents the quality pilot from turning into an unregistered feature-search exercise.

## Relation to the thesis

This stage strengthens Chapter 3 methodology and Chapter 4 evidence provenance. A failed quality pilot remains reportable: it would show that a public REST microstructure design was insufficient under the registered operational requirements and would justify a separately registered higher-fidelity collector rather than post-hoc threshold relaxation.
