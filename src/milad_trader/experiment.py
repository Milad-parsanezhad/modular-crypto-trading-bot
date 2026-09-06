"""Versioned walk-forward experiments. Test data never select thresholds."""
from pathlib import Path
from dataclasses import replace, asdict
import hashlib
import importlib.metadata
import json
import platform
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from .features import make_features,feature_columns,labels,ichimoku_signal
from .models import Predictor
from .engine import backtest
from .metrics import performance,paired_block_interval
from .splits import walk_forward


def digest_candles(df):
    return hashlib.sha256(df.to_csv(index_label="timestamp").encode()).hexdigest()


def code_fingerprint():
    root = Path(__file__).parent
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob("*.py"))}


def environment_versions():
    result = {"python":platform.python_version()}
    for name in ("numpy","pandas","scikit-learn","torch","stable-baselines3","gymnasium","xgboost","ccxt"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False,default=str))


def buy_and_hold(frame,start,end,risk):
    """Passive full-allocation comparator with fees/slippage, no risk overlays."""
    curve = pd.DataFrame(index=frame.index[start:end])
    price = frame.open.iloc[start+1]*(1+risk.slippage_bps/10000)
    quantity = risk.initial_cash/(price*(1+risk.fee))
    curve["equity"] = quantity*frame.close.iloc[start:end]
    curve.iloc[0,0] = risk.initial_cash
    exit_price = frame.close.iloc[end-1]*(1-risk.slippage_bps/10000)
    final = quantity*exit_price*(1-risk.fee)
    curve.iloc[-1,0] = final
    curve["net_return"] = curve.equity.pct_change().fillna(0.)
    fills = pd.DataFrame([
        dict(timestamp=frame.index[start+1],side="buy",price=price,quantity=quantity,fee=price*quantity*risk.fee,pnl=None,reason="passive_entry"),
        dict(timestamp=frame.index[end-1],side="sell",price=exit_price,quantity=quantity,fee=exit_price*quantity*risk.fee,pnl=final-risk.initial_cash,reason="terminal_liquidation")])
    return curve,fills


def run_experiment(raw,config,output,source_kind="user_supplied",source_metadata=None,resume=False):
    output = Path(output)
    output.mkdir(parents=True,exist_ok=resume)
    frame = make_features(raw)
    folds = list(walk_forward(len(frame),config))
    target = labels(frame,config.horizon,2*config.risk.fee+2*config.risk.slippage_bps/10000).to_numpy()
    manifest = dict(config=config.to_dict(),source_kind=source_kind,source_metadata=source_metadata,
                    data_sha256=digest_candles(raw),raw_rows=len(raw),feature_rows=len(frame),
                    first_open=str(raw.index[0]),last_open=str(raw.index[-1]),
                    versions=environment_versions(),source_sha256=code_fingerprint(),
                    folds=[asdict(f) for f in folds],status="running",
                    scope="development walk-forward; not an untouched final test")
    records,intervals,regimes,costs = [],[],[],[]
    completed = set()
    if resume and (output/"manifest.json").exists():
        previous = json.loads((output/"manifest.json").read_text())
        for key in ("config","source_kind","data_sha256","source_sha256","versions"):
            if previous[key] != manifest[key]:
                raise ValueError(f"Resume mismatch: {key}; choose a new experiment directory")
        if previous["status"] == "completed":
            return pd.read_csv(output/"metrics.csv")
        checkpoint = output/"progress.json"
        if checkpoint.exists():
            state = json.loads(checkpoint.read_text())
            records,intervals,regimes,costs = (state[k] for k in ("records","intervals","regimes","costs"))
            completed = set(state["completed"])
    write_json(output/"manifest.json",manifest)

    def checkpoint(name):
        completed.add(name)
        temporary = output/"progress.tmp.json"
        write_json(temporary,dict(records=records,intervals=intervals,regimes=regimes,costs=costs,
                                 completed=sorted(completed)))
        temporary.replace(output/"progress.json")

    for fold in folds:
        start,end = fold.test
        val_start,val_end = fold.validation
        # True hold comparator and risk-matched always-long comparator are distinct.
        reference,reference_fills = buy_and_hold(frame,start,end,config.risk)
        baseline = {
            "buy_and_hold":(reference,reference_fills),
            "cash":backtest(frame,np.zeros(len(frame),dtype=int),config.risk,start,end),
            "long_with_risk":backtest(frame,np.ones(len(frame),dtype=int),config.risk,start,end),
            "ichimoku":backtest(frame,frame.apply(ichimoku_signal,axis=1).to_numpy(),config.risk,start,end)
        }
        for name,(curve,fills) in baseline.items():
            if f"fold{fold.number}_{name}" in completed:
                continue
            records.append(dict(fold=fold.number,seed=None,model=name,features="rule",threshold=None,
                                **performance(curve,fills,config.bars_per_year)))
            curve.to_csv(output/f"fold{fold.number}_{name}_equity.csv")
            checkpoint(f"fold{fold.number}_{name}")
        for seed in config.seeds:
            for include_ichi in ([True,False] if config.ablation else [True]):
                cols = feature_columns(frame,include_ichi)
                values = frame[cols].to_numpy(float)
                train_idx = np.arange(config.lookback-1,fold.train[1])
                val_idx = np.arange(val_start,val_end)
                test_idx = np.arange(start,end)
                predictions = {}
                for kind in config.models:
                    name = f"fold{fold.number}_seed{seed}_{kind}_{'ichi' if include_ichi else 'no_ichi'}"
                    if name in completed:
                        restored = joblib.load(output/(name+".joblib"))
                        if kind in ("lstm","xgboost"):
                            predictor = restored["predictor"]
                            predictions[kind] = (predictor.predict(values,test_idx),
                                                 predictor.predict(values,val_idx),restored["threshold"])
                        print(f"Resumed completed {name}",flush=True)
                        continue
                    available_after = frame.index[val_end+config.horizon]
                    bundle = dict(kind=kind,config=config.to_dict(),columns=cols,
                                  trained_through=str(available_after),
                                  history_start=str(raw.index[0]),symbol=config.symbol,
                                  history_prefix_sha256=digest_candles(raw.loc[:available_after]),
                                  source_kind=source_kind)
                    threshold,validation_score = None,None
                    if kind == "ppo":
                        from .rl import fit_ppo,evaluate_ppo
                        agent,scaler = fit_ppo(frame,values,fold.train[1],config,seed)
                        curve,fills = evaluate_ppo(frame,values,start,end,agent,scaler,config)
                        agent.save(output/(name+"_agent"))
                        bundle.update(agent_file=name+"_agent.zip",scaler=scaler,actual_training_steps=agent.num_timesteps,
                                      training_sampling=agent.research_sampling)
                    else:
                        predictor = Predictor(kind,config.lookback,seed,config.epochs).fit(values,target,train_idx,val_idx)
                        val_probability = predictor.predict(values,val_idx)
                        scores = []
                        for t in config.thresholds:
                            actions = np.zeros(len(frame),dtype=int)
                            actions[val_idx] = val_probability >= t
                            vc,_ = backtest(frame,actions,config.risk,val_start,val_end)
                            scores.append(vc.equity.iloc[-1]/config.risk.initial_cash-1)
                        best = int(np.argmax(scores))
                        threshold,validation_score = config.thresholds[best],scores[best]
                        probability = predictor.predict(values,test_idx)
                        actions = np.zeros(len(frame),dtype=int)
                        actions[test_idx] = probability >= threshold
                        curve,fills = backtest(frame,actions,config.risk,start,end)
                        bundle.update(predictor=predictor,threshold=threshold)
                        predictions[kind] = (probability,val_probability,threshold)
                        known = np.isfinite(target[test_idx])
                        true = target[test_idx][known].astype(int)
                        prob = probability[known]
                        forecast = dict(accuracy=float(accuracy_score(true,prob>=.5)),
                                        roc_auc=float(roc_auc_score(true,prob)) if len(np.unique(true))>1 else None)
                        # Fixed stress multipliers; never choose a model using these test returns.
                        for multiple in (1.,2.,3.):
                            stressed = replace(config.risk,fee=config.risk.fee*multiple,
                                               slippage_bps=config.risk.slippage_bps*multiple)
                            sc,sf = backtest(frame,actions,stressed,start,end)
                            costs.append(dict(fold=fold.number,seed=seed,model=kind,ichimoku=include_ichi,
                                              cost_multiplier=multiple,**performance(sc,sf,config.bars_per_year)))
                    joblib.dump(bundle,output/(name+".joblib"))
                    record = dict(fold=fold.number,seed=seed,model=kind,features="ichi" if include_ichi else "no_ichi",
                                  threshold=threshold,validation_return=validation_score,
                                  **performance(curve,fills,config.bars_per_year))
                    if kind != "ppo":
                        record.update(forecast)
                    records.append(record)
                    curve.to_csv(output/(name+"_equity.csv"))
                    fills.to_csv(output/(name+"_fills.csv"),index=False)
                    paired = paired_block_interval(curve.net_return.iloc[1:],reference.net_return.iloc[1:],
                                                    block=12 if config.timeframe=="4h" else 48,seed=seed)
                    intervals.append(dict(fold=fold.number,seed=seed,model=kind,ichimoku=include_ichi,**paired))
                    # Regime is determined at the preceding decision candle, never using future labels.
                    for regime in ("bull","bear","sideways"):
                        mask = frame.regime.iloc[start:end-1].to_numpy() == regime
                        r = curve.net_return.iloc[1:].to_numpy()[mask]
                        regimes.append(dict(fold=fold.number,seed=seed,model=kind,ichimoku=include_ichi,
                                            regime=regime,bars=len(r),mean_net_return=float(r.mean()) if len(r) else None))
                    print(f"Completed {name}: return={record['total_return']:.4f}",flush=True)
                    checkpoint(name)
                if "lstm" in predictions and "xgboost" in predictions:
                    ensemble_name = f"fold{fold.number}_seed{seed}_ensemble_{include_ichi}"
                    if ensemble_name in completed:
                        continue
                    # Agreement uses each model's independently validation-chosen threshold.
                    a,_,ta = predictions["lstm"]
                    b,_,tb = predictions["xgboost"]
                    actions = np.zeros(len(frame),dtype=int)
                    actions[test_idx] = (a>=ta)&(b>=tb)
                    ec,ef = backtest(frame,actions,config.risk,start,end)
                    records.append(dict(fold=fold.number,seed=seed,model="lstm_xgboost_agreement",
                                        features="ichi" if include_ichi else "no_ichi",threshold=None,
                                        **performance(ec,ef,config.bars_per_year)))
                    ec.to_csv(output/f"fold{fold.number}_seed{seed}_ensemble_{include_ichi}_equity.csv")
                    checkpoint(ensemble_name)
    pd.DataFrame(records).to_csv(output/"metrics.csv",index=False)
    pd.DataFrame(regimes).to_csv(output/"regimes.csv",index=False)
    pd.DataFrame(costs).to_csv(output/"cost_sensitivity.csv",index=False)
    write_json(output/"bootstrap_intervals.json",intervals)
    manifest["status"] = "completed"
    write_json(output/"manifest.json",manifest)
    # No selection of a winning strategy from the held-out results.
    return pd.DataFrame(records)
