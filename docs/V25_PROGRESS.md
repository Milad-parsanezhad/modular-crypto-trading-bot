# v0.25 Progress Tracker

This file is updated stage-by-stage so GitHub, thesis notes and the defense package share the same research state.

## Discovery and frozen ranking stage

- [x] v0.24d failure mechanism formally recorded
- [x] literature review: learning-to-rank, survival analysis, backtest overfitting, cost-aware DRL
- [x] comparable-system bug review: FinRL timestamp drift, Qlib dependency/index failures
- [x] exact v0.24b event-filter identity inherited
- [x] deterministic frozen-score and score-margin ranking baselines implemented
- [x] Ridge expected-R ranker implemented
- [x] HistGradientBoosting expected-R ranker implemented
- [x] lower-quartile uncertainty-aware ranker implemented
- [x] pairwise logistic learning-to-rank challenger implemented
- [x] priority-aware MTM/CVaR/correlation portfolio replay implemented
- [x] validation-only selection runner implemented
- [x] Colab reproduction/defense notebook added
- [x] GitHub Actions exact-environment ranking workflow added
- [x] ranking CI run completed and artifact frozen (`34590214474` / artifact `10195326409`)
- [x] validation champion frozen: `hgb_expected_r`
- [x] spent OKX/KuCoin transfer diagnostic recorded as non-promotion evidence

## Prospective-evidence hardening stage

The original 12:00 UTC future boundary was replaced **before eligible outcome reading** by the stricter operational collector boundary `2026-09-11T16:00:00Z`. The purpose is to freeze ingestion, blinding, first-look and gate semantics before the prospective clock starts; it is not a performance-motivated retune of the ranker.

- [x] operational prospective boundary frozen (`2026-09-11T16:00:00Z`)
- [x] blind-until-mature protocol implemented
- [x] maturity rules made outcome-independent (time + coverage + event count only)
- [x] first mature PASS/FAIL made terminal for the sample
- [x] append-only first-observed OHLCV archive implemented
- [x] historical exchange restatement detection implemented
- [x] per-symbol SHA-256 provenance implemented
- [x] CCXT cursor bug hardened: full-candle pagination rather than `last_ts + 1ms`
- [x] bounded CCXT retry/fail-closed behavior implemented
- [x] portfolio-level 24/36/60 bps stress implemented
- [x] 36 bps stress included in the actual promotion gate
- [x] rolling CVaR included in the actual promotion gate
- [x] paired moving-block bootstrap on portfolio close-MTM return uplift implemented
- [x] regression tests added for preboundary exclusion, blinding, terminal first-look, append-only restatements, cost-stress monotonicity and CCXT candle cursor
- [x] scheduled prospective collector workflow authored at 4h + 23 minutes
- [ ] current prospective collector CI must complete green on the final frozen branch SHA
- [ ] default-branch scheduler must be installed on `main` and pin that exact green collector SHA
- [ ] first immutable prospective snapshot artifact must be created without economic peeking
- [ ] sufficient future-time evidence accumulated (>=168h and >=200 events/venue)
- [ ] first and only mature future MTM / PF / DD / CVaR / 36bps / paired-CI gate evaluated
- [ ] CPCV / PBO / DSR search-aware review if candidate survives
- [ ] Forward PAPER candidate decision if all prior gates survive

## Defense rule

The defense package must distinguish:

- validation selection evidence;
- spent/post-hoc diagnostic evidence;
- blinded accumulating future evidence;
- the single mature prospective read;
- search-aware audit evidence;
- Forward PAPER evidence.

A green GitHub Actions workflow proves protocol execution and reproducibility, **not profitability**.

`LIVE_EXECUTION = false`
