# v0.20 Source-to-Rule Traceability

## Uploaded-source rules preserved

- Order Block family: BOS must precede a valid OB hypothesis; confirmed entries wait for HTF reaction then seek LTF structure confirmation and retest.
- Supply/Demand family: DBD/DBR/RBR/RBD bases are contextual zones; HTF establishes direction and LTF executes.
- M1Trades/TTrades family: HTF context is mandatory for LTF setups; sweep -> structure shift -> origin/FVG retest is treated as a sequence, not a standalone candle pattern.
- Unicorn family: liquidity sweep + structure/breaker behavior + FVG overlap is treated as a confluence hypothesis.
- OSOK/fractal family: higher-timeframe swing/narrative is mapped to a lower-timeframe CISD/entry layer.
- Ichimoku family: 9/26/52 components are calculated causally; no future-shifted visual cloud data are permitted in a decision.

## Web/academic rules added

- Time-series momentum is added only at Daily/Weekly-context horizon, where its empirical source family is conceptually aligned; it is not cloned blindly to M1.
- Volatility management is a sizing overlay: rising realized volatility can reduce risk, never increase it above the frozen base risk.
- Transaction costs are part of the decision economics; they are never removed because a candidate looks better gross of costs.
- Multiple-testing controls increase as the strategy registry grows.
- Real order flow/open interest are reserved for a separate data-rich experiment instead of being approximated with fake candle-derived substitutes.

## Psychology-to-code translation

Because an automated system does not experience human emotions, psychology is represented by governance constraints that remove common behavioral failure modes: revenge-like rapid re-entry after a losing streak, overtrading, risk escalation during drawdown, volatility-insensitive sizing and discretionary override of a hard loss limit.
