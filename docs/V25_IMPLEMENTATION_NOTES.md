# v0.25 Implementation Notes — Multi-Plan Failure Handling

v0.25 follows a predeclared fallback ladder rather than repeating one failed mechanism.

## Plan A — ranking development

Fit the compact ranking tournament on archived development rows and choose the allocator on archived validation only. The exact v0.24b event filters are immutable.

If a learned ranker does not beat the frozen-score baseline, **the baseline wins**. Complexity is not mandatory.

## Plan B — future-time evidence

Freeze the ranker and evaluate only observations after `2026-09-11T12:00:00Z`. No backfill may be called prospective evidence.

If public-source ingestion is unavailable, record `BLOCKED` and switch data transport/provider without altering the model or symbol-selection rule.

## Plan C — prospective PAPER/shadow accumulation

If the future sample is statistically insufficient, continue frozen evidence collection rather than loosening the gate. Insufficient data is a valid scientific state.

## Plan D — redesign

If the future sample is sufficient and scientifically fails, record the failure. Any redesigned ranker must begin a new future evidence window. The failed future sample becomes spent.

## One-minute engineering anti-loop principle

Transient technical failures may be retried once. Repeated identical failure triggers diagnosis and a changed mechanism rather than an indefinite loop. Scientific failure is never retried by changing a threshold on the same test set.

## RL rule

RL is not a fallback for a failed supervised allocator. RL remains a separate later hypothesis and requires its own preregistered environment, reward, seeds, turnover/cost penalties, risk constraints and fresh evaluation.
