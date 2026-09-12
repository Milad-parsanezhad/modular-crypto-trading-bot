# v0.47 Engineering Repair Record — Cross-Machine Numerical Reproducibility Guard

Date: 2026-09-12
Status: **PRE-DECISION ENGINEERING DEFECT / SCIENTIFIC RESULT NOT PRODUCED**

## Superseded run

- workflow run: `34704888326`
- head: `174112ea6c8088e91628e58c283c47cf5277e02a`
- failed step: `Execute frozen v0.47 calibration ablation`
- no canonical decision, provenance artifact, or evidence artifact was produced
- C1/C2 outcomes from this run are inadmissible and were not interpreted

## Failure

The runner required bitwise-near equality between a newly refit C0 control and the scalar Brier value stored by the canonical v0.46 run.

Canonical v0.46 R1 common-space median Brier:

`0.6165395472330891`

New v0.47 C0 re-execution:

`0.6165100421842382`

The difference was approximately `2.95e-05` and triggered the fail-closed guard before decision/artifact creation.

## Investigation

The canonical v0.46 and failed v0.47 runs used the same relevant software versions:

- Python 3.11.16
- NumPy 2.4.6
- pandas 3.0.5
- scikit-learn 1.9.1
- SciPy 1.17.1
- Ubuntu runner image 20260907.300.1

However, GitHub Actions placed the canonical v0.46 job in Azure region `westus3` and the failed v0.47 job in `westus`.

scikit-learn's LogisticRegression documentation states that fitted coefficients can differ slightly for the same input across machines because of floating-point arithmetic and related numerical effects. The project uses `lbfgs`; therefore an exact cross-machine point-metric equality guard is not a valid provenance test.

Authoritative reference:
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

scikit-learn also documents that numerical linear algebra can be provided by BLAS/LAPACK implementations underneath NumPy/SciPy:
https://scikit-learn.org/stable/computing/computational_performance.html

## Repair principle

The repair does **not** loosen a scientific gate and does not change the model.

Instead:

1. C0/C1/C2 remain paired within the same workflow run and are compared against the same-run C0, eliminating cross-machine numerical drift from the treatment comparison.
2. The exact canonical v0.46 artifact is downloaded and verified for lineage/provenance.
3. Frozen source files inherited from v0.46 are required to have zero git diff relative to the canonical v0.46 scientific head.
4. The exact fold/symbol support coverage must match the canonical v0.46 R1 coverage.
5. A regression test verifies that v0.47 and v0.46 forecast-metric functions give the same Brier/reliability semantics when supplied identical probabilities and labels.
6. Numerical drift of the same-run C0 versus the historical canonical scalar is recorded transparently as diagnostic metadata, not used to alter thresholds or select an arm.

## Scientific invariants unchanged

- arms C0/C1/C2 unchanged
- temperature and Dirichlet specifications unchanged
- all forecast/economic promotion gates unchanged
- expected-R threshold unchanged (`> 0`)
- features, labels, folds, assets, venues, costs and Financial Governor unchanged
- Kraken remains sealed
- PAPER=false
- LIVE=false

This repair is permitted by the preregistered bug policy because the defect was discovered before any admissible v0.47 scientific decision existed.
