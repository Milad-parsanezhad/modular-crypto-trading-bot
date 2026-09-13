# v0.51 — Prospective Evidence Integrity Addendum — 2026-09-13

Status: **FROZEN BEFORE THE REPAIRED PROSPECTIVE START**

Parent preregistration: `d8ee4576aaf55750dd5910cc0d3b2efcbba3f5b2`  
Frozen calendar: `6a8fa49de2d1befa9c17049aa61e13da20a028eb`  
Valid predictor-identity amendment: `4838c0d98c408d358929b0767676a5ac024bd8dd`  
Repaired prospective start: `2026-09-13T12:00:00Z`

## Purpose

The v0.51 scientific question is prospective. Therefore a market bar must not become
eligible merely because its exchange timestamp lies after the prospective boundary if
that bar was first downloaded much later. This addendum freezes an operational
first-seen rule before prospective outcomes are used.

This rule is evidence-governance only. It does not alter A0, A1, the predictor,
Expected-R threshold, costs, exits, Financial Governor, risk limits, assets, venues,
or any advancement gate.

## Frozen first-seen rule

A closed 4-hour OHLCV bar may enter v0.51 prospective reconstruction only when all of
the following hold:

1. `bar_close_time >= 2026-09-13T12:00:00Z`;
2. `bar_close_time < prospective_start + 150 days`;
3. the collector observed the bar only after it was fully closed;
4. the immutable `first_seen_at` timestamp is no earlier than `bar_close_time`;
5. `first_seen_at <= bar_close_time + 60 minutes`.

Define:

`capture_lag_minutes = (first_seen_at - bar_close_time) / 60 seconds`

and

`prospective_eligible_v51 = 0 <= capture_lag_minutes <= 60` together with the frozen
prospective-window condition above.

The 60-minute tolerance is an operational allowance for exchange/API/runner delay. It
is fixed without inspecting any v0.51 economic outcome and must not be tuned later.

## Backfill rule

Bars fetched outside the first-seen tolerance are retained as auditable context but
must have `prospective_eligible_v51 = false`. A later re-fetch cannot change the
original `first_seen_at` or upgrade an ineligible historical/backfilled bar into
prospective evidence.

A missed timely observation is recorded as missing evidence; it is never silently
reconstructed as prospective from a later API backfill.

## Intended collection cadence

The operational target is one collector invocation shortly after each four-hour UTC
bar close (nominally minute 10 after `00:00`, `04:00`, `08:00`, `12:00`, `16:00`, and
`20:00` UTC). The scientific eligibility criterion is the frozen 60-minute first-seen
rule above, not the scheduler implementation.

## Immutability and revision handling

- candidate/raw bar identity: `venue × symbol × bar_open_time`;
- earliest `first_seen_at` is immutable;
- a revision to OHLCV values for an already recorded closed-bar identity fails closed;
- raw context is never deleted to make support look better;
- all evidence files carry SHA-256 hashes;
- missing series/venue/block coverage remains explicit and cannot be pruned.

## Safety

`RESEARCH_ONLY / KRAKEN_SEALED / PAPER_OFF / LIVE_OFF`

No order submission, paper execution, live execution, candidate promotion, or economic
conclusion is authorized by this addendum.
