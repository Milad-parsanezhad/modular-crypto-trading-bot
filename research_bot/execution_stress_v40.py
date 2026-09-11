from __future__ import annotations

"""v0.40 deterministic execution-cost, slippage and latency stress.

No alpha/model retraining. This stage consumes the frozen v0.38 allocated ledger
and the frozen v0.39 decision. Kraken remains untouched.
"""

from dataclasses import asdict, dataclass
from typing import Any, Iterable
import math
import numpy as np
import pandas as pd

DEVELOPMENT_VENUES_V40=("coinex_consumed","okx_consumed","kucoin_consumed")
RESERVED_HOLDOUT_VENUE_V40="kraken"
TOTAL_EFFECTIVE_TRIALS_V40=133

@dataclass(frozen=True)
class ExecutionScenarioV40:
    name:str
    target_round_trip_bps:float
    adverse_latency_bps:float

SCENARIOS_V40=(
    ExecutionScenarioV40("BASELINE_24",24.0,0.0),
    ExecutionScenarioV40("STRESS_36_PLUS5",36.0,5.0),
    ExecutionScenarioV40("STRESS_60_PLUS10",60.0,10.0),
)


def preregistration_manifest_v40()->dict[str,Any]:
    return {"version":"v0.40","experiment":"EXECUTION_REALISM_STRESS","scenarios":[asdict(s) for s in SCENARIOS_V40],
            "input_cost_already_embedded_bps":24.0,"no_alpha_retraining":True,"development_venues":list(DEVELOPMENT_VENUES_V40),
            "reserved_holdout_venue":RESERVED_HOLDOUT_VENUE_V40,"total_effective_trials":TOTAL_EFFECTIVE_TRIALS_V40,
            "stress_gate":"parent v0.39 valid AND 36+5 scenario keeps PF>=1.05/expectancy>0/MDD<=5% and 60+10 scenario keeps PF>=1.0/expectancy>0/MDD<=5% on every venue",
            "kraken_touched":False,"paper_replacement_authorized":False,"live_execution_authorized":False}


def _pf(v:Iterable[float])->float:
    a=np.asarray(list(v),dtype=float); a=a[np.isfinite(a)]
    w=float(a[a>0].sum()); l=float(-a[a<0].sum())
    return w/l if l>0 else (np.inf if w>0 else np.nan)


def _block_ci(v:Iterable[float],samples:int=750,block:int=20,seed:int=314)->tuple[float,float]:
    a=np.asarray(list(v),dtype=float); a=a[np.isfinite(a)]; n=len(a)
    if n<max(30,block): return np.nan,np.nan
    b=min(block,n); starts=np.arange(0,n-b+1); rng=np.random.default_rng(seed); need=int(math.ceil(n/b)); means=[]
    for _ in range(samples):
        idx=[]
        for s in rng.choice(starts,size=need,replace=True): idx.extend(range(int(s),int(s)+b))
        means.append(float(np.mean(a[np.asarray(idx[:n],dtype=int)])))
    return float(np.quantile(means,.025)),float(np.quantile(means,.975))


def apply_execution_scenario_v40(allocated:pd.DataFrame,scenario:ExecutionScenarioV40)->pd.DataFrame:
    x=allocated.copy(); x=x.loc[x.get('executed_v38',False).astype(bool)].copy()
    if x.empty: return x
    entry=pd.to_numeric(x['entry'],errors='coerce'); stop=pd.to_numeric(x['stop'],errors='coerce')
    stop_fraction=(entry-stop).abs()/entry.replace(0,np.nan)
    incremental_cost=max(0.0,(scenario.target_round_trip_bps-24.0+scenario.adverse_latency_bps)/10000.0)
    penalty_r=incremental_cost/stop_fraction.replace(0,np.nan)
    x['scenario_v40']=scenario.name
    x['incremental_cost_fraction_v40']=incremental_cost
    x['stressed_r_v40']=pd.to_numeric(x['r_multiple'],errors='coerce')-penalty_r
    x['stressed_account_return_v40']=pd.to_numeric(x['risk_fraction_v38'],errors='coerce').fillna(0.0)*x['stressed_r_v40']
    return x


def metrics_v40(stressed:pd.DataFrame)->dict[str,Any]:
    if stressed.empty: return {'trades':0,'profit_factor':np.nan,'expectancy_r':np.nan,'positive_asset_fraction':0.0,'block_ci_low':np.nan,'max_drawdown':np.nan}
    x=stressed.copy(); x['exit_time']=pd.to_datetime(x['exit_time'],utc=True)
    ret=pd.to_numeric(x['stressed_account_return_v40'],errors='coerce').fillna(0.0)
    risk=pd.to_numeric(x['risk_fraction_v38'],errors='coerce').fillna(0.0)
    r=pd.to_numeric(x['stressed_r_v40'],errors='coerce')
    batch=x.assign(_r=ret).groupby('exit_time',sort=True)['_r'].sum(); eq=(1+batch).cumprod(); dd=eq/eq.cummax()-1
    bysym=x.assign(_r=ret).groupby('symbol')['_r'].sum(); lo,hi=_block_ci(batch.to_numpy(dtype=float))
    return {'trades':int(len(x)),'profit_factor':float(_pf(ret)),'expectancy_r':float((ret.sum()/risk.sum()) if float(risk.sum())>0 else np.nan),
            'positive_asset_fraction':float((bysym>0).mean()) if len(bysym) else 0.0,'block_ci_low':lo,'block_ci_high':hi,
            'max_drawdown':float(dd.min()) if len(dd) else 0.0,'total_return':float(eq.iloc[-1]-1) if len(eq) else 0.0,
            'mean_stressed_r':float(r.mean())}


def screen_v40(parent_v39_valid:bool,results:dict[str,dict[str,dict[str,Any]]])->dict[str,Any]:
    venue_ok={}
    for v in DEVELOPMENT_VENUES_V40:
        m36=results[v]['STRESS_36_PLUS5']; m60=results[v]['STRESS_60_PLUS10']
        venue_ok[v]=bool(int(m36['trades'])>=200 and float(m36['profit_factor'])>=1.05 and float(m36['expectancy_r'])>0 and abs(float(m36['max_drawdown']))<=.05 and
                         int(m60['trades'])>=200 and float(m60['profit_factor'])>=1.0 and float(m60['expectancy_r'])>0 and abs(float(m60['max_drawdown']))<=.05)
    return {'parent_v39_valid':bool(parent_v39_valid),'venue_execution_ok':venue_ok,'execution_stress_eligible_v40':bool(parent_v39_valid and all(venue_ok.values()))}


def decision_v40(screen:dict[str,Any])->dict[str,Any]:
    ok=bool(screen.get('execution_stress_eligible_v40',False))
    return {'version':'v0.40','decision':'V40_EXECUTION_ROBUST_CANDIDATE_LOCKED_FOR_SHADOW' if ok else 'NO_V40_EXECUTION_ROBUST_CANDIDATE',
            'winner':'SOFT_CAUSAL_WATERFILL' if ok else None,'reserved_holdout_venue':'kraken','kraken_touched':False,
            'holdout_authorized_to_run':False,'paper_replacement_authorized':False,'live_execution_authorized':False}
