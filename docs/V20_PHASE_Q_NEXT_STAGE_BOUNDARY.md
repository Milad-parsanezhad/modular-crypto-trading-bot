# v0.20 Phase-Q Next-Stage Boundary

Phase Q ends only when the real prospective evidence satisfies the frozen data-quality gate. A pass does not select features or train models automatically.

The first post-pass commit must contain a separate pre-registration that freezes:

1. the exact one-row-per-symbol-per-4h feature aggregation rule;
2. the target/label definition;
3. point-in-time availability rules;
4. PRICE_ONLY and PRICE_PLUS_MICROSTRUCTURE candidate definitions;
5. training/validation/holdout geometry;
6. all planned statistical comparisons and multiplicity correction;
7. explicit cost and turnover assumptions;
8. stop rules and promotion gates.

No target-aware feature search is permitted before that commit.
