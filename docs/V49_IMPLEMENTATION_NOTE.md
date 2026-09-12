# v0.49 Implementation Note

This branch implements only the frozen preregistration at `4c6081504b57f8ed513a2d8445e6d17b3230b7a4`.

Invariants:

- exact canonical v0.44 prepared evidence;
- exact canonical v0.47 C1 lineage/evidence;
- base R1 multinomial learner fit on FIT only;
- scalar temperature fit on full CAL only;
- TEST labels are used only for diagnostic measurement, never model/calibrator fitting;
- three preregistered diagnostic families only: calibration-map drift, feature→residual interaction drift, Brier-loss CUSUM drift;
- no pruning, threshold search, model-capacity increase, trading-policy change or external holdout read;
- Kraken sealed; PAPER=false; LIVE=false;
- candidate promotion is impossible in v0.49.
