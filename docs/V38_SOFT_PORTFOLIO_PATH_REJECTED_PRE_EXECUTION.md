# v0.38 Soft-Portfolio Path — Rejected Before Scientific Execution

Status: `V38_SOFT_PORTFOLIO_PATH_REJECTED_PRE_EXECUTION`

This branch was implemented as a candidate direction but is not accepted as the next scientific experiment and must not be used as promotion evidence.

Reason: v0.36 failed the event-level temporal stability gate (`block_ci_low <= 0` on CoinEx), while v0.37 already showed that additional portfolio admission materially reduced sample size and damaged cross-venue robustness. Applying another allocation layer before repairing event-alpha temporal stability would target the wrong bottleneck and risks masking rather than solving the failure.

No result from this branch may authorize Kraken access, PAPER replacement, or LIVE execution. The replacement path is `research/v38-temporal-consensus`, which targets temporal/model stability directly using only consumed CoinEx/OKX/KuCoin evidence and keeps Kraken sealed.
