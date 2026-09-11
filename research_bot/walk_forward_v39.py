from __future__ import annotations

"""v0.39 anchored walk-forward stability diagnostics for the frozen v0.38 allocation.

No model or allocation parameter is changed. CoinEx/OKX/KuCoin remain consumed;
Kraken remains sealed. v0.39 is validation, not another optimizer.
"""

from typing import Any, Iterable
import math
import numpy as np
import pandas as pd

DEVELOPMENT_VENUES_V39=("coinex_consumed","okx_consumed","kucoin_consumed")
RESERVED_HOLDOUT_VENUE_V39="kraken"
N_FOLDS_V39=5
REQUIRED_POSITIVE_FOLDS_V39=4
TOTAL_EFFECTIVE_TRIALS_V39=133


def preregistration_manifest_v39()->dict[str,Any]:
    return {
        "version":"v0.39","experiment":"ANCHORED_WALK_FORWARD_STABILITY",
        "development_venues":list(DEVELOPMENT_VENUES_V39),"reserved_holdout_venue":"kraken",
        "folds":N_FOLDS_V39,"required_positive_folds":REQUIRED_POSITIVE_FOLDS_V39,
        "parent_parameters_frozen":True,"new_hyperparameter_trials":0,
        "total_effective_trials":TOTAL_EFFECTIVE_TRIALS_V39,
        "fold_pass":"PF>=1.0 and risk-weighted expectancy>0 and fold MDD<=5%",
        "aggregate_parent_gate_required":True,"kraken_touched":False,
        "paper_replacement_authorized":False,"live_execution_authorized":False,
    }


def _pf(values:Iterable[float])->float:
    a=np.asarray(list(values),dtype=float); a=a[np.isfinite(a)]
    w=float(a[a>0].sum()); l=float(-a[a<0].sum())
    return w/l if l>0 else (np.inf if w>0 else np.nan)


def _fold_metrics(frame:pd.DataFrame)->dict[str,Any]:
    if frame.empty:
        return {"trades":0,"profit_factor":np.nan,"expectancy_r":np.nan,"max_drawdown":np.nan,"positive_asset_fraction":0.0,"fold_pass":False}
    x=frame.copy(); x["exit_time"]=pd.to_datetime(x["exit_time"],utc=True)
    ret=pd.to_numeric(x["account_return_v38"],errors="coerce").fillna(0.0)
    risk=pd.to_numeric(x["risk_fraction_v38"],errors="coerce").fillna(0.0)
    batch=x.assign(_ret=ret).groupby("exit_time",sort=True)["_ret"].sum()
    eq=(1.0+batch).cumprod(); peak=eq.cummax(); dd=eq/peak-1.0
    denom=float(risk.sum()); exp=float(ret.sum()/denom) if denom>0 else np.nan
    bysym=x.assign(_ret=ret).groupby("symbol")["_ret"].sum()
    breadth=float((bysym>0).mean()) if len(bysym) else 0.0
    pf=float(_pf(ret.to_numpy(dtype=float)))
    mdd=float(dd.min()) if len(dd) else 0.0
    passed=bool(len(x)>0 and np.isfinite(pf) and pf>=1.0 and np.isfinite(exp) and exp>0 and abs(mdd)<=0.05)
    return {"trades":int(len(x)),"profit_factor":pf,"expectancy_r":exp,"max_drawdown":mdd,"positive_asset_fraction":breadth,"fold_pass":passed,
            "start":x["exit_time"].min().isoformat(),"end":x["exit_time"].max().isoformat()}


def walk_forward_v39(allocated:pd.DataFrame)->dict[str,Any]:
    x=allocated.loc[allocated.get("executed_v38",False).astype(bool)].copy() if not allocated.empty else pd.DataFrame()
    if x.empty:
        return {"folds":[],"positive_folds":0,"stable":False}
    x["exit_time"]=pd.to_datetime(x["exit_time"],utc=True)
    x=x.sort_values(["exit_time","symbol"],kind="mergesort").reset_index(drop=True)
    times=pd.Index(sorted(x["exit_time"].unique()))
    chunks=np.array_split(np.arange(len(times)),N_FOLDS_V39)
    folds=[]
    for i,idx in enumerate(chunks,1):
        if len(idx)==0:
            fm={"trades":0,"fold_pass":False}
        else:
            subset=x.loc[x["exit_time"].isin(times[idx])].copy()
            fm=_fold_metrics(subset)
        fm["fold"]=i; folds.append(fm)
    positive=sum(bool(f.get("fold_pass")) for f in folds)
    no_dd_breach=all((not np.isfinite(float(f.get("max_drawdown",np.nan)))) or abs(float(f.get("max_drawdown",0)))<=0.05 for f in folds if int(f.get("trades",0))>0)
    return {"folds":folds,"positive_folds":int(positive),"required_positive_folds":REQUIRED_POSITIVE_FOLDS_V39,
            "no_fold_drawdown_breach":bool(no_dd_breach),"stable":bool(positive>=REQUIRED_POSITIVE_FOLDS_V39 and no_dd_breach)}


def decision_v39(parent_v38_eligible:bool,venue_results:dict[str,dict[str,Any]])->dict[str,Any]:
    stable=bool(parent_v38_eligible and all(bool(venue_results[v].get("stable")) for v in DEVELOPMENT_VENUES_V39))
    return {"version":"v0.39","decision":"V39_TEMPORALLY_STABLE_CANDIDATE_LOCKED_FOR_EXECUTION_STRESS" if stable else "NO_V39_TEMPORALLY_STABLE_CANDIDATE",
            "winner":"SOFT_CAUSAL_WATERFILL" if stable else None,"parent_v38_eligible":bool(parent_v38_eligible),
            "development_venues":list(DEVELOPMENT_VENUES_V39),"reserved_holdout_venue":RESERVED_HOLDOUT_VENUE_V39,
            "kraken_touched":False,"holdout_authorized_to_run":False,"paper_replacement_authorized":False,"live_execution_authorized":False}
