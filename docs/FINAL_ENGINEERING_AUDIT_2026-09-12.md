# Final Engineering Audit — v0.50 Research Deployment

Date: 2026-09-12  
Audited source: `deploy/research-v50` at `8e12587f80bb1307f5508fdc33f0601f9fc7e1a5`  
Scope: repository-wide regression, point-in-time joins, risk inputs, simulated
execution, forward-paper safety, service contract, CI and deployment workflow.

## Result

The repository is engineering-ready as a fail-closed research service after
the fixes in this audit. It is not approved for PAPER or LIVE execution and
this audit is not evidence of profitability.

Scientific state remains unchanged:

`V50_NONOVERLAP_FAILURE_SUPPORTED / KRAKEN_SEALED / PAPER_OFF / LIVE_OFF`

## Defects found and repaired

| Area | Defect | Consequence before repair | Repair |
|---|---|---|---|
| Point-in-time data | `merge_asof` sorted by asset before time | Normal interleaved multi-asset panels raised `left keys must be sorted` | Globally sort by availability/decision time, then restore input decision order |
| Risk gate | NaN/Inf and invalid negative snapshot values passed comparisons | Invalid risk state could be approved | Fail closed with `INVALID_RISK_SNAPSHOT`; validate sizing inputs |
| Paper execution | NaN order quantities/prices were accepted | Non-finite phantom fills/account corruption | Validate finite policy, request and slippage inputs |
| Paper liquidity | Fill fraction had a forced 5% floor | Zero visible depth still produced a synthetic fill | Allow a true zero fill and return `NO_LIQUIDITY` |
| Exit governance | Drawdown/exposure gates were applied identically to entries and exits | A risk-reducing exit could be blocked by the condition it should reduce | Bypass capital-admission gates for EXIT while retaining liquidity simulation |
| Exit sizing | A position accumulated over several fills could exceed the per-order cap | The engine rejected the entire exit, leaving risk trapped | Liquidate in cap-compliant chunks across subsequent closed bars |
| Fail-closed defaults | `ForwardPaperRunner` defaulted to execution enabled | Direct construction could silently enable simulated execution | Default changed to observation-only; enabling remains explicit |
| Service firewall | Only two of four documented execution environment flags were checked | Legacy deployment flags could contradict the advertised firewall | Guard all documented LIVE/PAPER flags at import |
| Test/CI contract | Three historical service test files expected the retired PAPER API; deployment CI ran only three smoke tests | Full suite had seven failures hidden by narrow CI | Update legacy assertions to current v0.50 contract and run the entire suite in deployment readiness CI |
| Pandas 3 compatibility | A leakage test compared microsecond `DatetimeIndex` integers with nanosecond `Timestamp.value` | The causal sequence test failed only on the current GitHub runner despite the changed data beginning strictly after the comparison boundary | Compare timezone-aware timestamps directly, independent of internal resolution |
| Scheduled workflow | v0.15 still polled the deployed service every four hours using the retired PAPER/Postgres contract | Predictable false alarms and misleading evidence labels | Archive and explicitly disable the obsolete collector |
| Documentation | README still identified v0.41 as current | Repository entry point contradicted deployed v0.50 status | Update current lineage and v0.48–v0.50 references |

The limit-order enum remains fail-closed in the simulator: limit orders are
rejected as unsupported rather than silently receiving market-style fills.

## Fresh verification

- `python -m compileall -q research_bot scripts`: passed.
- `python -m pytest -q`: **321 passed, 3 skipped, 0 failed**.
- `python -m pip check`: no broken requirements.
- all 60 workflow YAML files parsed successfully.
- service smoke: `/health`, `/research/status`, `/research/latest` and
  `/paper/status` returned 200.
- execution smoke: `/decision/evaluate` and `/paper/run-once` returned 423.
- Git diff whitespace validation: passed.

The three skips require the optional PyTorch stack and cover deep temporal and
ICT-vision paths. They were not silently counted as passes.

## Remaining limitations and promotion blockers

1. The v0.50 result is a failure-attribution result, not a profitable trading
   candidate. v0.51 must be preregistered before implementing or comparing an
   alternative overlap arbitration policy.
2. Kraken remains sealed. No external holdout was consumed by this audit.
3. No PostgreSQL integration server or exchange endpoint was exercised in the
   local deterministic audit; those require separate environment-backed tests.
4. Optional PyTorch tests were skipped because the `deep` dependency group was
   not installed in this runtime.
5. Historical baseline modules preserve some older research assumptions (for
   example close-to-close baseline execution). They must not be rewritten
   retroactively; any next-open replacement belongs to a new versioned
   experiment.
6. Multi-asset forward-paper marking retains the last in-process mark and falls
   back to average entry price after restart. This is acceptable while PAPER is
   disabled, but persistent synchronized marks are required before any future
   paper reactivation.

## Release decision

`ENGINEERING_AUDIT_PASS / RESEARCH_SERVICE_ONLY / SCIENTIFIC_PROMOTION_BLOCKED`

Merge may proceed after GitHub CI passes. PAPER/LIVE activation, Kraken access
and profitability claims remain prohibited.
