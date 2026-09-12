# v0.46 Execution Note

This stacked empirical branch is descended from preregistration head `630aa870cf9db3f879d98dc9eaf8ed9571b872a1`.

The v0.46 empirical workflow is allowed to read only the canonical consumed v0.44 prepared development artifact from run `34700944062`. It must not instantiate or fetch Kraken and must retain `PAPER=false` and `LIVE=false`.

Any implementation correction discovered before a valid v0.46 empirical result is observed must be documented prospectively in the PR and must not alter the frozen hypothesis, arms, state boundary, features, fold boundaries, costs, or Financial Governor thresholds.
