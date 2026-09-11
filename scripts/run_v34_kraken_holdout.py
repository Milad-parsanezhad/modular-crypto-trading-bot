from __future__ import annotations

import argparse, hashlib, json, time
from pathlib import Path
from typing import Any
import ccxt
import numpy as np
import pandas as pd

from research_bot.breadth_stability_v33 import V33_CANDIDATES, apply_breadth_gate, cross_sectional_breadth
from research_bot.causal_evaluation_v25 import evaluate_external_replication_v25
from research_bot.cross_venue_robustness_v27 import compact_external_metrics
from research_bot.drawdown_firewall_v31 import apply_drawdown_firewall_v31, allocator_diagnostics_v31
from research_bot.kraken_holdout_v34 import v34_decision
from research_bot.multitimeframe_strategies_v19 import StrategySpec, TournamentConfig
from research_bot.multitimeframe_strategies_v20 import RiskPsychologyPolicy, V20ValidationConfig, simulate_v20_trades
from research_bot.portfolio_allocator_v25 import PortfolioRiskBudgetV25
from research_bot.regime_event_alpha_v30 import V30_CANDIDATES, cross_sectional_dispersion, generate_direction_v30

LOCKED_SYMBOLS = ["BTC/USDT","ETH/USDT","SOL/USDT","XRP/USDT","DOGE/USDT","ADA/USDT","LTC/USDT","BCH/USDT","LINK/USDT","TRX/USDT","AVAX/USDT","DOT/USDT","ETC/USDT","ATOM/USDT","XLM/USDT","UNI/USDT","FIL/USDT","AAVE/USDT","NEAR/USDT","ALGO/USDT"]


def safe(v: Any) -> Any:
    if isinstance(v, dict): return {str(k):safe(x) for k,x in v.items()}
    if isinstance(v, (list,tuple)): return [safe(x) for x in v]
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating,)): return float(v) if np.isfinite(v) else None
    if isinstance(v, float) and not np.isfinite(v): return None
    if isinstance(v, pd.Timestamp): return v.isoformat()
    return v


def fetch_kraken(as_of: pd.Timestamp, bars: int = 1800):
    ex = ccxt.kraken({"enableRateLimit": True})
    ex.load_markets()
    step_ms = int(ex.parse_timeframe("1d") * 1000)
    end_ms = int(as_of.timestamp()*1000)
    frames, errors = {}, {}
    try:
        for symbol in LOCKED_SYMBOLS:
            if symbol not in ex.markets:
                errors[symbol] = "market unavailable"; continue
            try:
                cursor = end_ms - (bars+120)*step_ms
                rows=[]; loops=0
                while cursor < end_ms and len(rows) < bars+120 and loops < 40:
                    loops += 1
                    batch = ex.fetch_ohlcv(symbol, timeframe="1d", since=cursor, limit=min(720,bars+120-len(rows)))
                    if not batch: break
                    rows.extend(r for r in batch if int(r[0]) < end_ms)
                    nxt = int(batch[-1][0]) + step_ms
                    if nxt <= cursor: break
                    cursor = nxt
                    time.sleep(ex.rateLimit/1000.0 if ex.rateLimit else 0.05)
                if not rows: errors[symbol]="no rows"; continue
                x = pd.DataFrame(rows, columns=["timestamp_ms","open","high","low","close","volume"])
                x["timestamp"] = pd.to_datetime(x["timestamp_ms"], unit="ms", utc=True)
                x = x[["timestamp","open","high","low","close","volume"]].drop_duplicates("timestamp").sort_values("timestamp")
                x = x.loc[x["timestamp"] + pd.Timedelta(days=1) <= as_of].tail(bars).reset_index(drop=True)
                if len(x) >= 600: frames[symbol]=x
                else: errors[symbol]=f"insufficient rows={len(x)}"
            except Exception as exc:
                errors[symbol]=f"{type(exc).__name__}: {exc}"
    finally:
        try: ex.close()
        except Exception: pass
    return frames, errors


def frame_digest(frame):
    x=frame[["timestamp","open","high","low","close","volume"]].copy(); x["timestamp"]=pd.to_datetime(x["timestamp"],utc=True).astype(str)
    return hashlib.sha256(x.to_csv(index=False,float_format="%.12g").encode()).hexdigest()


def snapshot(frames, path):
    parts=[]
    for s,f in sorted(frames.items()):
        x=f.copy(); x.insert(0,"symbol",s); parts.append(x)
    if parts: pd.concat(parts,ignore_index=True).to_csv(path,index=False,compression="gzip")


def evaluate_winner(v33, frames):
    base = next(x for x in V30_CANDIDATES if x.name == "V30_D1_CUSUM_BREAKOUT_3")
    market = frames["BTC/USDT"]
    dispersion = cross_sectional_dispersion(frames, "1d")
    breadth = cross_sectional_breadth(frames)
    spec = StrategySpec(v33.name,"1d","v33_breadth_stability","Frozen v0.30 CUSUM breakout + causal cross-sectional breadth confirmation",rr=base.rr,stop_atr=base.stop_atr,max_hold_bars=base.max_hold_bars,long_short=True)
    risk=RiskPsychologyPolicy(); budget=PortfolioRiskBudgetV25()
    tournament=TournamentConfig(risk_per_trade=risk.base_risk_per_trade,min_pretest_trades=1000,min_test_trades=200)
    parts=[]
    for symbol, frame in sorted(frames.items()):
        direction, features = generate_direction_v30(base, frame, market_frame=market, dispersion=dispersion)
        direction, features = apply_breadth_gate(direction, features, breadth, v33)
        led=simulate_v20_trades(spec,frame,direction,features,symbol,tournament=tournament,policy=risk)
        if not led.empty: parts.append(led)
    if not parts: return pd.DataFrame(), {}
    raw=pd.concat(parts,ignore_index=True).sort_values(["entry_time","symbol"],kind="mergesort").reset_index(drop=True)
    allocated=apply_drawdown_firewall_v31(raw,spec,risk_policy=risk,budget=budget)
    m=dict(evaluate_external_replication_v25(allocated,validation=V20ValidationConfig(min_external_trades=200)))
    m.update(allocator_diagnostics_v31(allocated))
    return allocated, compact_external_metrics(m, venue="kraken")


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--v33-artifact-dir",required=True); ap.add_argument("--output-dir",default="artifacts/v34-kraken-holdout"); a=ap.parse_args()
    src,out=Path(a.v33_artifact_dir),Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    decision=json.loads((src/"decision_v33.json").read_text()); lock=json.loads((src/"development_lock_v33.json").read_text())
    prereq = decision.get("decision") == "V33_BREADTH_CANDIDATE_LOCKED_FOR_FRESH_HOLDOUT" and bool(decision.get("winner"))
    if not prereq:
        d=v34_decision(decision,lock,None); (out/"decision_v34.json").write_text(json.dumps(d,indent=2,sort_keys=True)); print(json.dumps(d,indent=2)); return

    winner=next(x for x in V33_CANDIDATES if x.name == decision["winner"])
    as_of=pd.Timestamp(lock["locked_at_utc"]); as_of=as_of.tz_convert("UTC") if as_of.tzinfo else as_of.tz_localize("UTC")
    frames,errors=fetch_kraken(as_of)
    provenance={"venue":"kraken","timeframe":"1d","as_of_utc":as_of.isoformat(),"symbol_count":len(frames),"rows":{s:len(f) for s,f in sorted(frames.items())},"sha256_by_symbol":{s:frame_digest(f) for s,f in sorted(frames.items())},"errors":errors}
    (out/"holdout_provenance_v34.json").write_text(json.dumps(safe(provenance),indent=2,sort_keys=True))
    if len(frames) < 15 or "BTC/USDT" not in frames:
        d=v34_decision(decision,lock,{"trades":0}); d["reason"]="Untouched Kraken coverage was insufficient; no fallback venue was substituted."
        (out/"decision_v34.json").write_text(json.dumps(d,indent=2,sort_keys=True)); print(json.dumps(d,indent=2)); return
    snapshot(frames,out/"kraken_1d_ohlcv_v34.csv.gz")
    ledger,metrics=evaluate_winner(winner,frames)
    if not ledger.empty: ledger.to_csv(out/"kraken_holdout_ledger_v34.csv",index=False)
    (out/"holdout_metrics_v34.json").write_text(json.dumps(safe(metrics),indent=2,sort_keys=True))
    d=v34_decision(decision,lock,metrics)
    (out/"decision_v34.json").write_text(json.dumps(d,indent=2,sort_keys=True))
    print(json.dumps({"decision":d,"metrics":metrics,"provenance":provenance},indent=2,default=str))

if __name__=="__main__": main()
