# v0.19 Audit-Corrected Results & Microstructure Infrastructure Verification

Date: 2026-09-10

Status: **AUDIT VERIFIED / PROSPECTIVE COLLECTION ONLY / NO PROMOTION**

## Reproducibility record

- PR: `#14`
- Verified branch head: `85d3b399934b722e6912b857e660ae1c1e7f2989`
- GitHub Actions workflow: `v19-audit-corrected-research`
- Successful run: `34474370382`
- Artifact: `v19-audit-corrected-results`
- Artifact ID: `10150943098`
- Artifact digest: `sha256:d8b70ebc58c4d2ed59e0590c8c071ef6eb9dab0895d3308392b69ffc250f794a`

The successful workflow completed deterministic audit/maturity-gate tests, the refreshed public-data audit, a live public multi-venue fixed-window collector smoke test, fail-closed contract validation, and immutable artifact upload.

## Additional infrastructure bugs closed in the final test pass

The final microstructure test pass added invariants beyond the earlier M1-M7 audit:

### M8 — unique-venue coverage invariant

Earlier aggregation counted accepted observations. In a pathological duplicate-provider input, two valid observations from the same venue could therefore satisfy `min_venues_per_symbol=2`. The repaired implementation counts **unique venue names**, keeps only the latest valid observation per venue for diagnostics, and gates any duplicate-venue snapshot with `DUPLICATE_VENUE_OBSERVATIONS`.

Regression tests now prove that:

- two CoinEx observations do not count as two venues;
- a duplicated venue cannot silently pass the cross-venue quality gate even when enough unique venues remain;
- `accepted_venue_names` is explicitly materialized.

### M9 — stable fail-closed schema and configured-universe integrity

Insufficient-coverage output previously had a thinner schema than normal output. The repaired implementation always returns stable quality fields including `quality_flags`, accepted venue names and diagnostic placeholders. It also rejects unconfigured venues/symbols and materializes a missing configured symbol as an explicit fail-closed row rather than silently omitting it.

### M10 — trade-window metadata integrity

The fixed 60-second trade window is now validated as part of the evidence contract. `trade_window_start` must equal `observed_at - 60s` and `trade_window_end` must equal `observed_at`; recorded first/last trade timestamps must remain inside the PIT window.

## Verified retrospective audit results

These results are a **refreshed reconstruction audit**, not a replacement for the immutable v0.18 results because the exact original v0.18 raw OHLCV bytes were not archived.

### A — cost-aware reconstruction audit

| Metric | Naive | Cost-aware |
|---|---:|---:|
| Net return | -9.1269% | +0.3350% |
| Sharpe | -1.443 | +0.629 |
| Max drawdown | -13.4355% | -0.7536% |
| Turnover | 115.00 | 6.00 |

Paired moving-block bootstrap 95% CI for the cost-aware minus naive mean return:

`[-0.0002726978, +0.0005570517]`

The interval includes zero. Evidence label:

`REFRESHED_RECONSTRUCTION_AUDIT_NOT_ORIGINAL_V18_SAMPLE`

No alpha/promotion claim is permitted.

### B — corrected v0.11 regime-construction audit

Ten usable OKX symbols were evaluated with the restored cross-sectional top-quartile Ichimoku construction and market-level regime definition.

| Metric | Plain cross-sectional Ichimoku | Market-regime conditioned |
|---|---:|---:|
| Net return | -43.1233% | -34.9873% |
| Sharpe | -0.671 | -0.956 |
| Max drawdown | -65.1716% | -59.3142% |
| Turnover | 506.67 | 362.67 |

Bootstrap 95% CI for conditioned minus plain mean return:

`[-0.0005343910, +0.0005802782]`

The interval includes zero. Evidence label:

`RETROSPECTIVE_CONSTRUCTION_AUDIT_NOT_FRESH_REPLICATION`

Although conditioning reduced the aggregate loss, drawdown and turnover, it worsened the observed Sharpe and did not establish a statistically supported generalizable edge.

## Verified prospective fixed-window collector smoke snapshot

The same successful run collected six public observations with **zero provider failures**: BTC/USDT and ETH/USDT from CoinEx, OKX and KuCoin.

Measurement contract:

- trading/forecast horizon: `4h`;
- target measurement cadence: `30min`;
- reported-trade PIT window: `60s`;
- trade-side semantics: `exchange_reported_side_unverified_aggressor`;
- venue coverage unit: `unique_venue`;
- REST snapshots are explicitly **not** described as an L2/L3 event stream.

Both BTC and ETH passed the smoke-snapshot quality gates with three unique accepted venues each.

| Symbol | Unique venues | Clock skew | Mid dispersion | Mean spread | Mean reported trade imbalance | Status |
|---|---:|---:|---:|---:|---:|---|
| BTC/USDT | 3 | 4.075 s | 3.0141 bps | 0.3942 bps | -0.7976 | OK |
| ETH/USDT | 3 | 3.901 s | 1.3675 bps | 0.0407 bps | +0.1097 | OK |

These are **measurement-health values**, not trading signals.

## Integrity verification

The exported files were independently re-hashed after download.

- `v19_audit.json` self-excluding manifest SHA-256: `a5325eb0992f2170e1043b76da703d869de1a4d98254bfa6cbe78fb6fc2c5602` — verified.
- `forward_microstructure_snapshot.json` self-excluding SHA-256: `2a80acfc111860cb89d64e37fd90bca6a9c3e1f579d78bc9a62e4918f917936a` — verified.
- `unexpected_observation_count = 0`.
- `provider_failures = 0`.
- `authorized_symbol_count = 2` for this smoke snapshot.

## Scientific interpretation

The v0.19 infrastructure has now demonstrated that the fixed-window, point-in-time, multi-venue measurement contract can operate successfully on a real public snapshot and that the regression suite rejects several classes of silent measurement corruption: future trades, stale trades, inconsistent windows, duplicate-provider pseudo-coverage, unconfigured entities, clock skew and payload mutation.

This is **engineering/data-quality evidence only**. It does not establish predictive value. Phase Q remains immature until the pre-registered prospective data-quality requirement is satisfied: at least 168 elapsed hours, at least 336 nominal 30-minute measurement opportunities, and at least 80% authorized coverage for both BTC and ETH. Manual reruns and duplicate artifacts do not count as extra prospective time.

Only after Phase Q passes may the project freeze 4h microstructure aggregations and later perform the registered `PRICE_ONLY` versus `PRICE_PLUS_MICROSTRUCTURE` predictive ablation on at least 250 independent 4h decision timestamps.

## Promotion decision

`SIGNAL_AUTHORIZED = false`

`PAPER_STRATEGY_REPLACEMENT = false`

`LIVE_EXECUTION = false`

The correct current label is:

`V19_MEASUREMENT_INFRASTRUCTURE_VERIFIED_FORWARD_SAMPLE_IMMATURE`
