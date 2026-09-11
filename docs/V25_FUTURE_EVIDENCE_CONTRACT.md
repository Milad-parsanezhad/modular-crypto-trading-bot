# v0.25 Future Evidence Contract

Frozen start boundary: **2026-09-11T12:00:00Z**.

No bar, signal, trade outcome or portfolio result before this timestamp can count as v0.25 prospective promotion evidence.

The future evaluator must use:

- the exact v0.24b persisted event filters;
- the v0.25 ranker frozen before future outcome reading;
- unchanged ranking feature schema;
- unchanged event-filter thresholds;
- unchanged portfolio risk contract unless a new version starts a new future clock;
- UTC-aware completed 4h bars;
- explicit data provenance and gap diagnostics;
- realistic cost stress;
- MTM/correlation/CVaR portfolio replay.

If the ranker, feature schema, risk contract or scientific gate changes after future outcomes are observed, that future sample is spent and a new prospective boundary is required.

A future PASS may create a `FORWARD_PAPER_CANDIDATE`; it cannot by itself authorize real-money LIVE execution.
