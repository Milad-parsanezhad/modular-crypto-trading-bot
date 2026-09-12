# Final Research-Bot Architecture

## Scope

This document defines software completeness for the thesis repository. The system is complete as a reproducible, fail-closed **research trading bot** when all components below are present and CI-valid. Scientific alpha qualification is a separate gate and is never inferred from software completeness.

## End-to-end path

1. **Market data** — public OHLCV acquisition on consumed development venues.
2. **Causal feature layer** — point-in-time features and completed higher-timeframe context only.
3. **Independent strategy engines** — ICT, SMC, Ichimoku, Al Brooks-inspired price action, MTF.
4. **Mother strategy** — independent engine outputs become features; no hard all-engine conjunction.
5. **Event layer** — broad causal mother events with semantic event-family identity.
6. **Learning layer** — frozen experiment-specific models under purged walk-forward validation.
7. **Path/risk layer** — v0.41 target-vs-stop competing risks, timeout value, duration and conservative expected R.
8. **Uncertainty layer** — split-conformal lower expected-R admission.
9. **Trade realization** — next-open conservative brackets, stop-first same-bar ordering, non-overlap enforcement.
10. **Financial governor** — uncertainty-aware stop-risk sizing, portfolio caps, drawdown throttling/firewall and deterministic psychology governance.
11. **Evidence layer** — reproducible decision JSON/CSV/GZIP artifacts, workflow run IDs and SHA-256 digests.
12. **CLI** — doctor, manifest, status, feature building and frozen characterization from one executable surface.

## Strategy engines

### ICT

- liquidity sweep;
- MSS/CHoCH/BOS context;
- displacement;
- premium/discount.

### SMC

- BOS/CHoCH;
- FVG;
- order-block mitigation;
- supply/demand retest.

### Ichimoku

- Kumo regime;
- Tenkan/Kijun relation;
- Kijun slope;
- breakout;
- pullback.

### Al Brooks-inspired objective engine

- Always-In direction;
- trend vs range;
- breakout / failed breakout;
- H1/H2 and L1/L2;
- wedge;
- micro double top/bottom;
- signal bar;
- follow-through;
- measured-move context.

The Brooks layer is an objective research formalization and not a claim to reproduce all discretionary Brooks chart reading.

## Financial contract

- base stop-risk per trade: 0.25% equity;
- max stop-risk per trade: 0.50%;
- aggregate open stop-risk: 2.00%;
- same-direction open stop-risk: 1.50%;
- max asset nominal weight: 35%;
- max portfolio gross: 70%;
- drawdown throttles at 2.0% and 3.5%;
- hard new-risk firewall at 5.0%;
- no martingale;
- no averaging down;
- no revenge-trading logic;
- no discretionary override;
- no automatic risk increase after losses;
- two-day cooldown after three consecutive realized losses in the frozen research simulator.

## Anti-overfit contract

- causal rolling robust normalization;
- purged chronological walk-forward folds;
- embargo;
- calibration separated from model fitting;
- fixed seeds;
- median ensemble instead of best-seed selection;
- frozen preregistration before empirical characterization;
- same-test rescue tuning prohibited;
- zero-selection folds count as non-positive;
- venue-level gates must all pass;
- worst venue/quarter/regime diagnostics retained;
- external holdout remains sealed until explicitly authorized by a development winner.

## Executable interface

Install:

```bash
python -m pip install -e .
```

Health:

```bash
modular-crypto-bot doctor
```

Current state:

```bash
modular-crypto-bot status
```

Manifest:

```bash
modular-crypto-bot manifest --version v41
```

Feature build:

```bash
modular-crypto-bot features --input data.csv --output features.csv
```

Frozen characterization:

```bash
modular-crypto-bot characterize --version v41 --output-dir results/v41_characterization
```

## Intentional non-features

There is no `live`, `order`, `buy`, `sell` or PAPER execution command. This is intentional. Execution permission is a scientific promotion state, not a missing software feature.

Kraken remains the sealed external holdout. `PAPER=false`; `LIVE=false`.

## Definition of complete

The bot is **software-complete for thesis research** when:

- the CLI installs and passes `doctor`;
- causal strategy features are reproducible;
- frozen characterization runs end-to-end;
- risk limits are invariant-tested;
- results and negative evidence are persisted;
- no hidden path can place an order;
- CI passes.

The bot is **not scientifically promoted** until a preregistered development candidate passes every frozen robustness gate and a subsequent, separately authorized untouched holdout protocol succeeds.
