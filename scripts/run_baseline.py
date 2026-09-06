from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from research_bot.data import fetch_with_fallback
from research_bot.features import add_features
from research_bot.regime import add_regime_features
from research_bot.model import fit_predict_holdout,positions_from_probabilities
from research_bot.backtest import backtest_positions,performance_metrics

def _json_clean(obj):
    if isinstance(obj,dict): return {k:_json_clean(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)): return [_json_clean(v) for v in obj]
    if isinstance(obj,(np.floating,float)): return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj,(np.integer,int)): return int(obj)
    return obj

def main():
    p=argparse.ArgumentParser(); p.add_argument("--exchange",default="coinex"); p.add_argument("--symbol",default="BTC/USDT"); p.add_argument("--timeframe",default="4h"); p.add_argument("--limit",type=int,default=2000); p.add_argument("--fee-bps",type=float,default=10.0); p.add_argument("--slippage-bps",type=float,default=2.0); p.add_argument("--upper",type=float,default=0.56); p.add_argument("--output",default="artifacts/baseline_report.json"); args=p.parse_args()
    fallback=[args.exchange]+[x for x in ["coinex","kraken","okx"] if x!=args.exchange]
    raw,used_exchange=fetch_with_fallback(fallback,symbol=args.symbol,timeframe=args.timeframe,limit=args.limit)
    feat=add_regime_features(add_features(raw)); model,train,test,cols=fit_predict_holdout(feat,train_fraction=0.70,hurdle_bps=args.fee_bps+args.slippage_bps)
    positions=positions_from_probabilities(test["prob_up"],upper=args.upper,allow_short=False)
    _,ml_metrics=backtest_positions(test["future_return"],positions,timeframe=args.timeframe,fee_bps=args.fee_bps,slippage_bps=args.slippage_bps)
    bh_metrics=performance_metrics(test["future_return"],timeframe=args.timeframe)
    _,mom_metrics=backtest_positions(test["future_return"],(test["mom_24"]>0).astype(float),timeframe=args.timeframe,fee_bps=args.fee_bps,slippage_bps=args.slippage_bps)
    report={"research_status":"baseline_v0.1_not_production","exchange":used_exchange,"symbol":args.symbol,"timeframe":args.timeframe,"bars_raw":int(len(raw)),"train_rows":int(len(train)),"test_rows":int(len(test)),"features":cols,"frictions":{"fee_bps":args.fee_bps,"slippage_bps":args.slippage_bps},"strategy":{"type":"cost-aware abstaining long/flat HistGradientBoosting baseline","probability_upper":args.upper,"allow_short":False},"metrics":{"buy_hold":bh_metrics,"momentum_24":mom_metrics,"ml_abstain":ml_metrics},"warnings":["This is a first holdout baseline, not evidence of profitability.","Funding, order-book slippage, market impact, CPCV/PBO/DSR and paper trading are not yet implemented.","No API keys are required or used; only public market data are fetched."]}
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(_json_clean(report),indent=2),encoding="utf-8")
    print("===BASELINE_REPORT_JSON==="); print(json.dumps(_json_clean(report),indent=2)); print("===END_BASELINE_REPORT_JSON==="); print(f"Saved: {out}")
if __name__=="__main__": main()
