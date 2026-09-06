"""Command line for data, research and simulated forward execution."""
import argparse
from dataclasses import replace
from pathlib import Path
import json
import time
from datetime import datetime,timezone
from .config import load_config
from .data import read_csv,synthetic_candles,save_candles,fetch_public
from .experiment import run_experiment


def main():
    parser = argparse.ArgumentParser(description="Ichimoku/ML/RL research and paper trading")
    sub = parser.add_subparsers(dest="command",required=True)
    for name in ("demo","research"):
        p = sub.add_parser(name)
        p.add_argument("--config",default="configs/smoke.toml" if name=="demo" else "configs/research_4h.toml")
        p.add_argument("--output",required=True)
        if name=="research":
            p.add_argument("--data",required=True)
            p.add_argument("--resume",action="store_true",help="Reuse verified completed model checkpoints")
        else:
            p.add_argument("--rows",type=int,default=1600)
    p = sub.add_parser("fetch")
    p.add_argument("--exchange",default="coinex",choices=["coinex","binance","kraken"])
    p.add_argument("--symbol",default="BTC/USDT")
    p.add_argument("--timeframe",default="4h",choices=["1h","4h"])
    p.add_argument("--since",required=True)
    p.add_argument("--until")
    p.add_argument("--output",required=True)
    p = sub.add_parser("fetch-archive")
    p.add_argument("--symbol",default="BTC/USDT",choices=["BTC/USDT","ETH/USDT"])
    p.add_argument("--timeframe",default="4h",choices=["1h","4h"])
    p.add_argument("--since",required=True)
    p.add_argument("--until",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--cache-dir",default="data/raw/archive-cache")
    for name in ("paper-step","paper-watch"):
        p=sub.add_parser(name)
        p.add_argument("--bundle",required=True)
        p.add_argument("--state",required=True)
        if name=="paper-step":
            p.add_argument("--data",required=True)
        else:
            p.add_argument("--exchange",default="coinex",choices=["coinex","binance","kraken"])
            p.add_argument("--poll-seconds",type=int,default=60)
            p.add_argument("--iterations",type=int,default=0,help="0 means run until interrupted")
    args=parser.parse_args()
    if args.command in ("demo","research"):
        cfg=load_config(args.config)
        if args.command=="demo":
            raw=synthetic_candles(args.rows,cfg.timeframe)
            source_kind,meta="synthetic_software_test",{"generator_seed":314}
        else:
            raw=read_csv(args.data,cfg.timeframe)
            sidecar=Path(args.data).with_suffix(".manifest.json")
            meta=json.loads(sidecar.read_text()) if sidecar.exists() else None
            if meta and (meta.get("symbol")!=cfg.symbol or meta.get("timeframe")!=cfg.timeframe):
                raise ValueError("Dataset manifest symbol/timeframe differs from experiment")
            if meta:
                import hashlib
                if meta["sha256"] != hashlib.sha256(Path(args.data).read_bytes()).hexdigest():
                    raise ValueError("Dataset checksum mismatch")
            source_kind=meta.get("kind","user_supplied") if meta else "user_supplied_unverified"
        table=run_experiment(raw,cfg,args.output,source_kind,meta,resume=getattr(args,"resume",False))
        print(table[["fold","model","features","total_return","max_drawdown"]].to_string(index=False))
        print(f"Saved {args.output}; source_kind={source_kind}")
    elif args.command=="fetch-archive":
        from .archive import fetch_archive
        if Path(args.output).exists():
            raise FileExistsError("Choose a new snapshot filename")
        raw, metadata = fetch_archive(args.symbol,args.timeframe,args.since,args.until,args.cache_dir)
        print(json.dumps(save_candles(raw,args.output,metadata),indent=2))
    elif args.command=="fetch":
        if Path(args.output).exists():
            raise FileExistsError("Choose a new snapshot filename")
        raw=fetch_public(args.exchange,args.symbol,args.timeframe,args.since,args.until)
        manifest=save_candles(raw,args.output,dict(source=args.exchange,kind="historical",symbol=args.symbol,
                                                 timeframe=args.timeframe,retrieved_at=datetime.now(timezone.utc).isoformat()))
        print(json.dumps(manifest,indent=2))
    else:
        from .paper import load_bundle,paper_step
        bundle=load_bundle(args.bundle)
        cfg=bundle["configuration"]
        if args.command=="paper-step":
            print(json.dumps(paper_step(args.bundle,read_csv(args.data,cfg.timeframe),args.state),indent=2))
        else:
            if args.poll_seconds<30 or args.iterations<0:
                raise ValueError("poll-seconds must be >=30 and iterations >=0")
            i=0
            while args.iterations==0 or i<args.iterations:
                raw=fetch_public(args.exchange,cfg.symbol,cfg.timeframe,bundle["history_start"])
                print(json.dumps(paper_step(args.bundle,raw,args.state)),flush=True)
                i+=1
                if args.iterations==0 or i<args.iterations:
                    time.sleep(args.poll_seconds)


if __name__=="__main__":
    main()
