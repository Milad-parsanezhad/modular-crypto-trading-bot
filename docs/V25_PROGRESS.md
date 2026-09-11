# v0.25 Progress Tracker

This file is updated stage-by-stage so GitHub, thesis notes and the defense package share the same research state.

- [x] v0.24d failure mechanism formally recorded
- [x] literature review: learning-to-rank, survival analysis, backtest overfitting, cost-aware DRL
- [x] comparable-system bug review: FinRL timestamp drift, Qlib dependency/index failures
- [x] future-time boundary frozen (`2026-09-11T12:00:00Z`)
- [x] exact v0.24b event-filter identity inherited
- [x] deterministic frozen-score and score-margin ranking baselines implemented
- [x] Ridge expected-R ranker implemented
- [x] HistGradientBoosting expected-R ranker implemented
- [x] lower-quartile uncertainty-aware ranker implemented
- [x] pairwise logistic learning-to-rank challenger implemented
- [x] priority-aware MTM/CVaR/correlation portfolio replay implemented
- [x] validation-only selection runner implemented
- [x] Colab reproduction/defense notebook added
- [x] GitHub Actions exact-environment workflow added
- [ ] CI scientific/engineering run completed and artifact frozen
- [ ] validation champion and spent-sample diagnostic recorded
- [ ] prospective future-time collector started from frozen ranker artifact
- [ ] sufficient future-time events accumulated
- [ ] future MTM / PF / DD / CVaR / cost-stress gate evaluated
- [ ] CPCV / PBO / DSR search-aware review if candidate survives
- [ ] Forward PAPER candidate decision
- [ ] RL allocator remains locked until the above evidence survives

`LIVE_EXECUTION = false`
