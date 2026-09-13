# v0.20 Implementation Notes

The v0.20 lab is intentionally fail-closed. It extends v0.19 rather than rewriting its historical evidence.

Core invariants:

- 42 total candidates, seven per timeframe (1m/5m/15m/1h/4h/1d).
- 30 v0.19 candidates are retained unchanged as baselines.
- 12 new candidates use completed higher-timeframe context; no in-progress HTF candle is visible to LTF decisions.
- New strategies use a Search-and-Destroy/no-trade proxy after recent two-sided liquidity sweeps in a weak-trend region.
- Order-flow/OI/funding/basis/CVD/liquidation hypotheses are excluded from OHLCV-only testing unless genuine point-in-time data are available.
- Risk psychology is translated to deterministic governance: volatility throttle, drawdown throttle, 5% hard kill, loss-streak cooldown and daily anti-overtrading cap.
- Raw signal attempts and executed risk-managed trades are distinguished. The requested 1,000+ observation gate is applied to pre-test signal trades, while validation also requires 200+ actually executed trades.
- Search intensity is recorded (42 trials) and a conservative multiplicity-adjusted screen is combined with moving-block bootstrap evidence.
- Historical evidence can at most create a FORWARD_PAPER_CANDIDATE. LIVE stays false.
