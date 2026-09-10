# v0.19 Multi-Timeframe Strategy Tournament — Official Results

Date: 2026-09-10

Status: **verified research evidence; no strategy promoted; LIVE remains disabled**

GitHub Actions run: `34462995517`  
Artifact: `v19-multitimeframe-tournament`  
Artifact ID: `10146553844`  
Artifact digest: `sha256:ddc91a305f358243aec9da0d5762324e37e712d5d9223edf0203602a78904f5f`

## Experiment

The frozen v0.19 registry contained **30 source-derived candidates**, five each for `1m`, `5m`, `15m`, `1h`, `4h` and `1d`. The public-data runner requested CoinEx spot OHLCV for 12 assets: BTC, ETH, SOL, XRP, DOGE, ADA, LTC, BCH, LINK, TRX, AVAX and DOT against USDT.

Execution remained causal: signal at closed bar `t`, fill at open `t+1`, 10 bps fee + 2 bps slippage one way, ATR-normalized stop, 3R default target, 0.25% account risk per trade, no overlapping trade for the same symbol/candidate and stop-first resolution when stop and target occur in the same candle.

The chronological split was 60% development, 20% validation and 20% untouched test. A candidate required at least 1,000 development+validation closed trades before test qualification, positive validation expectancy, validation profit factor >=1.05, >=60% positive assets and <=5% validation maximum drawdown. Only the validation ranking could select the candidate that was allowed to face the final promotion gate.

## Sample-depth result

- 20/30 candidates accumulated at least 1,000 pre-test trades.
- 23/30 accumulated at least 200 test trades.
- 5/30 had positive validation expectancy.
- 3/30 had validation profit factor >=1.05.
- 2/30 had >=60% positive assets in validation.
- 5/30 individually respected the <=5% validation drawdown threshold.
- **No candidate satisfied all frozen pre-test requirements simultaneously.**

Formal decision: `NO_STRATEGY_PROMOTED`.

## Most informative candidates

| Candidate | TF | Pre-test trades | Validation PF | Validation expectancy R | Validation MDD | Positive assets | Test PF | Test expectancy R | Test MDD | Test return |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| H4_OB_BOS_RETEST | 4h | 2,016 | 1.370 | +0.244 | -14.40% | 83.33% | 0.784 | -0.163 | -24.45% | -19.09% |
| H4_S6_BREAKOUT | 4h | 1,734 | 1.247 | +0.168 | -8.74% | 66.67% | 0.820 | -0.142 | -29.76% | -14.38% |
| D1_ICT_SWING_PD | 1d | 829 | 0.894 | -0.063 | -7.18% | 41.67% | 1.082 | +0.043 | -3.90% | +2.21% |
| D1_FIB_INSTITUTIONAL | 1d | 639 | 0.880 | -0.072 | -4.58% | 41.67% | 1.071 | +0.038 | -4.42% | +1.60% |

The H4 OB/BOS and S6 candidates looked attractive in validation but violated the 5% drawdown gate and then failed on the untouched test. This is exactly the pattern the protocol was designed to catch.

The two daily candidates with positive final-test expectancy (`D1_ICT_SWING_PD` and `D1_FIB_INSTITUTIONAL`) cannot be promoted because their validation evidence was negative and their sample depth was below the 1,000-trade pre-test requirement. Their positive test outcomes therefore do not reverse the pre-registered decision.

## Low-timeframe cost result

The 1m and much of the 5m/15m family performed poorly after the frozen 24 bps round-trip friction. This is scientifically useful: a source pattern can be visually compelling while still being economically untradeable at retail crypto friction. The correct next study is not to remove costs after seeing the losses. It is to pre-register a separate maker/limit-fill experiment with spread, queue/fill probability and venue-specific fees.

## Interpretation

v0.19 successfully converted the educational material into testable, causal strategy families, but **did not discover a strategy that clears the project's research gates**. Consequently:

- no v0.19 candidate replaces the current PAPER strategy;
- no real-money or LIVE authorization is created;
- the uploaded-source strategies remain a valuable hypothesis library rather than certified alpha;
- any next iteration must use new/frozen evidence instead of tuning these rules on the consumed final test.

## Next justified experiments

1. Make the MTF implementations structurally closer to the source models by using actual resampled HTF PD arrays/zones instead of same-timeframe context proxies.
2. Test limit-order/maker execution for M1/M5/M15 with realistic fill probability, spread and adverse-selection modelling; keep the 24/36/60 bps market-execution scenario as stress evidence.
3. Add derivatives-only candidates using real point-in-time OI, funding, basis, CVD/taker flow and liquidations; do not backfill unavailable OI snapshots.
4. Expand the daily/swing cross-sectional universe so legitimate sparse strategies can reach the requested 1,000+ observations without loosening signal definitions.
5. After a new validation winner exists, use search-aware SPA/Reality Check, DSR/PSR and PBO/CPCV as statistically appropriate before a fresh forward-PAPER run.

The current decision remains fail-closed: **`NO_STRATEGY_PROMOTED`**.
