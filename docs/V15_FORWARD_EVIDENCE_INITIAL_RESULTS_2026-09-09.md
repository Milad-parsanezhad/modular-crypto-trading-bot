# v0.15 Initial Forward Evidence Results — 2026-09-09

Workflow run: `34361196453`

Artifact: `v15-forward-evidence-2`

Artifact ID: `10107909348`

Artifact digest: `sha256:8008fc3eea1bf621347754755c91cfe8c56832ba7003e79f7107ca35fba7ea25`

## CI result

The full v0.15 evidence workflow completed successfully.

Deterministic tests:

`3 passed in 0.03s`

Production evidence export and evidence-contract validation also completed successfully.

## First production evidence snapshot

At the time of the first v0.15 snapshot, the Railway forward-paper service reported:

- observations: `2`;
- simulated paper fills: `1`;
- paper equity points: `2`;
- cumulative paper return: `-0.00026116726816582947` (approximately `-0.0261%`);
- current paper drawdown: `-0.00026116726816582947` (approximately `-0.0261%`);
- production safety contract: `PASS`;
- evidence state: `INSUFFICIENT_FORWARD_SAMPLE`;
- live promotion: `PROHIBITED`.

## Interpretation

This is an engineering and forward-data-collection result, not a profitability result.

The first snapshot proves that the scheduled evidence pipeline can independently read the deployed Railway service, validate PAPER-only safety semantics, summarize PostgreSQL-backed forward observations/fills/equity counts, and create a reproducible GitHub Actions artifact.

The sample is deliberately classified as insufficient because it is far below the pre-registered v0.15 minimums of:

- 168 forward hours;
- 100 observations;
- 10 simulated fills.

No Sharpe, alpha, profitability, or live-readiness conclusion is authorized from this sample.

## Reproducibility

The artifact contains three files:

1. `v15_forward_evidence.json` — normalized machine-readable evidence snapshot;
2. `v15_forward_evidence.md` — human-readable thesis/defense snapshot;
3. `raw_production_endpoints.json` — raw public production endpoint responses used to construct the snapshot.

The scheduled workflow now creates the same evidence package every four hours and retains each artifact for 90 days.
