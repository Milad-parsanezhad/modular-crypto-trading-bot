from __future__ import annotations

import argparse
import json
from pathlib import Path


def pct(x):
    return "—" if x is None else f"{100*float(x):+.2f}%"


def num(x, digits=3):
    return "—" if x is None else f"{float(x):.{digits}f}"


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input", default="artifacts/v18/v18_cost_regime_results.json")
    p.add_argument("--output", default="artifacts/v18/V18_RESULTS.md")
    a=p.parse_args()
    z=json.loads(Path(a.input).read_text(encoding="utf-8"))
    A=z["experiment_a"]; B=z["experiment_b"]
    an=A["naive_portfolio"]; ac=A["cost_aware_portfolio"]; ai=A["paired_block_bootstrap"]
    bp=B["plain_ichimoku"]; bc=B["regime_conditioned_ichimoku"]; bi=B["paired_block_bootstrap"]
    text=f"""# v0.18 Cost-Aware Alpha Conversion & External Regime Replication — Results

Status: **Research evidence only — no PAPER replacement or LIVE authorization**

## Sources

- Experiment A: {z['experiment_a_source']}
- Experiment B: {z['experiment_b_external_source']}

## Experiment A — Frozen forecast: naive vs cost-aware conversion

Decision: **`{A['decision']}`**

| Metric | Naive sign trading | Cost-aware abstention |
|---|---:|---:|
| Net return | {pct(an.get('net_total_return'))} | {pct(ac.get('net_total_return'))} |
| Sharpe | {num(an.get('sharpe'))} | {num(ac.get('sharpe'))} |
| Max drawdown | {pct(an.get('max_drawdown'))} | {pct(ac.get('max_drawdown'))} |
| Turnover sum | {num(an.get('turnover_sum'))} | {num(ac.get('turnover_sum'))} |
| Explicit cost sum | {num(an.get('cost_sum'),5)} | {num(ac.get('cost_sum'),5)} |
| Exposure | {pct(an.get('exposure'))} | {pct(ac.get('exposure'))} |

Paired moving-block bootstrap on cost-aware minus naive mean net return:

- n: `{ai.get('n')}`
- observed mean difference: `{num(ai.get('mean'),6)}`
- 95% CI: `[{num(ai.get('ci_low'),6)}, {num(ai.get('ci_high'),6)}]`

Interpretation: support is granted only if the cost-aware portfolio is positive net, reduces turnover, and the lower 95% block-bootstrap bound is above zero. Otherwise this hypothesis remains unsupported in the frozen v0.18 configuration.

## Experiment B — Frozen Ichimoku/regime external replication

Decision: **`{B['decision']}`**

Frozen favorable regimes: `HIGH_VOL`, `TREND_DOWN`.

| Metric | Plain Ichimoku | Regime-conditioned Ichimoku |
|---|---:|---:|
| Net return | {pct(bp.get('net_total_return'))} | {pct(bc.get('net_total_return'))} |
| Sharpe | {num(bp.get('sharpe'))} | {num(bc.get('sharpe'))} |
| Max drawdown | {pct(bp.get('max_drawdown'))} | {pct(bc.get('max_drawdown'))} |
| Turnover sum | {num(bp.get('turnover_sum'))} | {num(bc.get('turnover_sum'))} |
| Explicit cost sum | {num(bp.get('cost_sum'),5)} | {num(bc.get('cost_sum'),5)} |
| Exposure | {pct(bp.get('exposure'))} | {pct(bc.get('exposure'))} |

Paired moving-block bootstrap on conditioned minus plain mean net return:

- n: `{bi.get('n')}`
- observed mean difference: `{num(bi.get('mean'),6)}`
- 95% CI: `[{num(bi.get('ci_low'),6)}, {num(bi.get('ci_high'),6)}]`

Interpretation: this is a direct external-venue replication of the frozen v0.11 regime hypothesis. No threshold or regime definition is tuned on this external sample.

## Scientific boundary

- `live_execution_authorized = false`
- `paper_strategy_promotion_authorized = false`
- support for either hypothesis, if observed, requires v0.17 search-aware statistical audit and prospective replication before any promotion.
- a negative result is retained as first-class evidence and is not tuned away on the same sample.
"""
    out=Path(a.output); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(text,encoding="utf-8")
    print(out)

if __name__=="__main__":
    main()
