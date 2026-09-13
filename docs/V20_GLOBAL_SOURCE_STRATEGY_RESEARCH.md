# v0.20 Global-Source Multi-Timeframe Strategy + Risk/Psychology Lab

Date frozen: 2026-09-10

Status: RESEARCH / PAPER ONLY. LIVE execution is not authorized.

## Scope and honesty constraint

No finite research process can literally crawl every educational website in the world. v0.20 therefore uses a broad, auditable hierarchy: (1) primary/official ICT material, (2) the user-supplied ICT/TTrades/M1Trades/Order-Block/Supply-Demand/Unicorn books and notes, (3) professional curricula and regulator education, and (4) peer-reviewed or clearly labelled academic/preprint research. Educational claims are treated as hypotheses until causal, cost-aware out-of-sample evidence supports them.

## Primary educational sources incorporated

- The Inner Circle Trader official site / Free Library / Mentorship Core Content.
- Official ICT 2022 Mentorship material, especially Fair Value Gap, market-structure shift and institutional-order-flow lessons.
- User-supplied ICT Mentorship Handbook / TTrades notes / M1Trades System / Order Blocks / Supply & Demand / Unicorn Model / Advanced Technical Analysis.
- CMT Association 2026 curriculum: system design/testing, ATR-based stops, reward/risk, VaR/CVaR, leverage, diversification, behavioral finance and risk governance.
- CFA Institute 2026 material: rolling/walk-forward backtesting, look-ahead and survivorship bias, risk/trading-cost models, portfolio correlation and behavioral biases.
- CME education: open interest is a contextual confirmation variable, not a standalone signal.
- NFA/CFTC risk education: leveraged/crypto trading can generate large losses; capital preservation and risk capital are mandatory constraints.

## Academic evidence used as design input

1. White, *A Reality Check for Data Snooping* — repeated model search creates selection risk.
2. Sullivan, Timmermann & White, *Data-Snooping, Technical Trading Rule Performance, and the Bootstrap* — technical-rule universes must be evaluated with the search process in mind.
3. Bailey & Lopez de Prado, *The Deflated Sharpe Ratio* — Sharpe inflation from multiple testing and non-normal returns must be accounted for.
4. Bailey et al., *The Probability of Backtest Overfitting* — PBO/CSCV motivates search-aware validation rather than selecting the best backtest.
5. Moskowitz, Ooi & Pedersen, *Time Series Momentum* — trend persistence is a legitimate research family, particularly at slower horizons.
6. Moreira & Muir, *Volatility-Managed Portfolios* — exposure should fall when volatility rises; v0.20 translates this as a frozen risk-scaling rule, not as an alpha claim.
7. 2026 *Order flow and cryptocurrency returns* (Journal of Financial Markets) — real point-in-time order flow is a promising crypto predictor; it is deliberately NOT fabricated from OHLCV and remains a separate data track.
8. 2026 BTC walk-forward ML under transaction costs — cost-aware abstention is more important than raw directional accuracy in some configurations; v0.20 retains explicit friction and fail-closed promotion.

## Source-derived strategy architecture

The registry contains 42 candidates: the 30 frozen v0.19 baselines plus 12 new v0.20 variants (two new variants for every timeframe: 1m, 5m, 15m, 1h, 4h, 1d).

### ICT/TTrades mechanics

Mechanical features include BSL/SSL liquidity sweeps, causal swing structure, BOS/MSS proxies, FVG, OB/origin retests, premium/discount, OTE, session windows, supply/demand bases and causal Ichimoku. Search-and-Destroy/double-sided sweep conditions are used as a NO_TRADE regime for new v0.20 candidates.

### Completed-HTF rule

A lower-timeframe decision may only use a higher-timeframe candle after that HTF candle closes. Mapping is frozen as:

- 1m -> completed 15m context
- 5m -> completed 1h context
- 15m -> completed 4h context
- 1h -> completed 1d context
- 4h -> completed 1d context
- 1d -> completed 7d context

This explicitly blocks a common multi-timeframe look-ahead error.

## Risk and psychology policy

A bot has no fear, greed or revenge impulse; therefore "trading psychology" is implemented as process governance rather than sentiment theatre.

Frozen controls:

- Base risk: 0.25% of current equity per executed trade.
- Absolute per-trade ceiling: 0.50%.
- Volatility scaling: causal rolling ATR% target; risk scales down as current volatility rises, never up beyond base risk.
- Drawdown throttle: full risk above -2%; 0.75x from -2% to -3.5%; 0.50x from -3.5% to -5%.
- Hard kill: at -5% strategy drawdown, no new trades in that evaluation path.
- Three consecutive losses trigger a timeframe-specific cooldown.
- Intraday families are capped at four executions per symbol/day.
- Rejected attempts remain in the immutable evidence ledger with a reason code.

Fractional/optimal Kelly is not used for live sizing in v0.20 because edge estimates are too uncertain and selection-sensitive. It can be added later as a diagnostic using development data only.

## Validation contract

- Signals use only closed-bar information; fills occur at the next bar open.
- 10 bps fee + 2 bps slippage are charged one way in the bracket simulator (24 bps round trip).
- Same-bar stop/target collision resolves stop-first.
- Minimum 1,000 development+validation signal trades are required for internal eligibility.
- Minimum 200 executed validation trades.
- Positive validation expectancy after cost.
- Profit factor >= 1.05.
- >=60% positive assets.
- Risk-managed maximum drawdown <=5%.
- Moving-block bootstrap lower mean-return confidence bound >0.
- Conservative multiplicity-adjusted p-value <=0.10 across the full 42-candidate search.
- Only an internally eligible winner is sent to a second public crypto venue for external replication.
- External replication must independently pass minimum sample, PF, expectancy, breadth, drawdown and block-bootstrap gates.
- A pass creates only a FORWARD_PAPER_CANDIDATE. It never authorizes LIVE execution or automatic replacement of an existing PAPER strategy.

## Important statistical limitation

The v0.20 multiplicity screen is deliberately labelled as a conservative normal/Bonferroni screen combined with a moving-block bootstrap. It is not falsely presented as White Reality Check, SPA, DSR or PBO. Those tests remain the next formal inference layer once a candidate survives the current economic/risk gates.

## Data integrity

Historical OI, funding, basis, CVD, liquidation, depth and order-flow fields are used only when a genuine point-in-time source exists. OHLCV never silently substitutes for these variables. The order-flow research therefore motivates a dedicated derivatives/microstructure experiment rather than an invented feature.

## Promotion semantics

`NO_STRATEGY_PROMOTED`: no candidate has enough robust evidence.

`FORWARD_PAPER_CANDIDATE`: candidate passed the frozen historical/internal and external-venue replication gates and may begin a fresh forward PAPER observation period.

`LIVE`: impossible in this protocol.
