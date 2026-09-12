# v0.39 Implementation Audit Correction 1

Status: **technical correction; frozen hypothesis unchanged**

The first executable v0.39 characterization run (`34674711993`) completed successfully as software and produced useful diagnostics, but an immediate implementation audit identified two bookkeeping defects relative to the preregistered evaluation semantics. Therefore that run is **not treated as the final scientific v0.39 decision**.

No strategy feature, model hyperparameter, threshold, cost, symbol, venue, time window, seed, fold boundary, or qualification gate is changed by this correction.

## Defect 1 — cross-venue capital coupling

The first runner passed all selected trades from CoinEx, OKX and KuCoin through one shared account-risk state. This allowed a position on one venue to consume risk budget or psychology state for another venue.

The preregistration requires each venue to be evaluated independently. Therefore the corrected implementation runs the identical financial allocator separately for each venue before calculating venue gates.

This correction changes bookkeeping only; it does not change model predictions or event labels.

## Defect 2 — empty folds excluded from positive-fold denominator

The first runner calculated positive-fold fraction using only folds with a finite selected-trade expectancy. A fold selecting zero events was therefore omitted from the denominator.

Under the frozen five-fold protocol, a fold with no selected events provides no positive OOS evidence and must count as non-positive for the positive-fold fraction.

The corrected implementation therefore computes:

`positive_fold_fraction = positive folds / all 5 preregistered folds`

with zero-selection folds counted as false.

## Reporting correction

Worst-group diagnostics (`venue × quarter × regime`) are now written for every evaluated candidate, not only for a winner. This is a reporting correction and cannot change qualification.

## Scientific governance

- First run `34674711993`: retained as a technical/audit run, not final scientific evidence.
- Corrected run: same frozen hypothesis, same data window, same model order, same costs, same gates.
- Kraken remains sealed.
- PAPER/LIVE remain disabled.
- No post-result threshold relaxation or feature selection is permitted.

If the corrected run rejects all candidates, that rejection is retained. If a candidate passes only because of the corrected venue-independent financial accounting, it still must satisfy every original frozen gate and the corrected five-fold denominator.
