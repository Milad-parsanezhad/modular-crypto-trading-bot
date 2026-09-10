# v0.20 Web Source Matrix

Frozen 2026-09-10. These public sources are design evidence, not profitability guarantees.

| Source | Evidence used in v0.20 | Implementation consequence |
|---|---|---|
| Official Inner Circle Trader site and official 2022 Mentorship | Primary ICT study path; FVG, market structure shift, institutional order-flow concepts | ICT rules remain explicit hypotheses; new variants require completed HTF context and causal LTF triggers |
| User-supplied ICT/TTrades/M1Trades/Order Blocks/Supply-Demand/Unicorn material | Liquidity sweep, BOS/MSS/CISD, FVG, OB, breaker, PD arrays, sessions, OSOK/fractal mapping | 42-candidate registry; two new source-derived variants for every timeframe |
| CMT Association 2026 Program Guide | Risk targeting, leverage, VaR/CVaR, diversification, ATR stops, reward/risk, behavioral/system discipline | ATR stops, fixed risk budget, drawdown throttle/kill switch, system-first rules |
| CFA Institute 2026 Backtesting/Active Investing/Behavioral Biases | Walk-forward/rolling testing; look-ahead, survivorship, overfit, costs; confirmation/loss-aversion/overconfidence biases | causal HTF availability, explicit friction, immutable search registry, deterministic anti-overtrading/cooldown controls |
| CME Open Interest education | OI is contextual and should be combined with other analysis | OI not imputed from OHLCV; separate derivatives track only when historical point-in-time OI exists |
| NFA/CFTC investor education | leverage and crypto speculation can amplify losses; use risk capital | LIVE remains disabled; hard loss governance and paper-first promotion |
| White (Reality Check) + Sullivan/Timmermann/White | model search creates data-snooping bias | trial count preserved; conservative multiplicity screen before any candidate can advance |
| Bailey & Lopez de Prado DSR; Bailey et al. PBO | selection bias/backtest overfit can inflate apparent Sharpe | v0.20 does not call a high historical score proof; formal DSR/PBO/SPA is a later inference gate |
| Moskowitz/Ooi/Pedersen Time Series Momentum | slower-horizon return persistence is a legitimate empirical family | D1 weekly-context TSMOM candidate; not blindly ported to all intraday horizons |
| Moreira & Muir Volatility-Managed Portfolios | risk should decline when volatility rises | causal ATR%-based risk scale, capped at base risk |
| 2026 Journal of Financial Markets: Order flow and cryptocurrency returns | point-in-time order flow has predictive information for crypto cross-section | order-flow candidate deferred until real timestamped order-flow dataset exists; never fabricated from candles |
| 2026 BTC walk-forward ML under costs | naive forecasts can fail after costs; cost-aware abstention can matter | 24 bps round-trip friction retained; no retrospective fee removal |

## URLs / identifiers

- https://www.theinnercircletrader.com/
- https://www.youtube.com/@InnerCircleTrader
- https://cmtassociation.org/cmt-program/
- https://www.cfainstitute.org/insights/professional-learning/refresher-readings/2026/backtesting-and-simulation
- https://www.cmegroup.com/education/courses/introduction-to-futures/open-interest
- https://www.nfa.futures.org/investors/investor-resources/files/investor-best-practices.html
- https://www.cftc.gov/LearnAndProtect/AdvisoriesAndArticles/understand_risks_of_virtual_currency.html
- White (2000), DOI 10.1111/1468-0262.00152
- Sullivan, Timmermann & White, DOI 10.1111/0022-1082.00163
- Bailey & Lopez de Prado, DOI 10.3905/jpm.2014.40.5.094
- Bailey et al., DOI 10.21314/JCF.2016.322
- Moreira & Muir, DOI 10.1111/jofi.12513
- Anastasopoulos et al. (2026), DOI 10.1016/j.finmar.2026.101047
