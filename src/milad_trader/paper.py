"""Checkpointed forward paper simulation. No authenticated order capability."""
from pathlib import Path
import hashlib
import json
import os
import joblib
import numpy as np
import pandas as pd
from .config import RiskConfig,ExperimentConfig
from .engine import Broker
from .experiment import digest_candles
from .features import make_features


def load_bundle(path):
    # Pickle/joblib is executable: load only locally created/trusted experiment artifacts.
    bundle = joblib.load(path)
    cfg = dict(bundle["config"])
    cfg["risk"] = RiskConfig(**cfg["risk"])
    bundle["configuration"] = ExperimentConfig(**cfg)
    if bundle["kind"] == "ppo":
        from stable_baselines3 import PPO
        bundle["agent"] = PPO.load(Path(path).parent/bundle["agent_file"],device="cpu")
    return bundle


def intention(bundle, frame, broker):
    x = frame[bundle["columns"]].to_numpy(float)
    cfg = bundle["configuration"]
    if bundle["kind"] == "ppo":
        from .rl import observation
        obs = observation(bundle["scaler"].transform(x),len(x)-1,cfg.lookback,broker,frame.close.iloc[-1])
        action,_ = bundle["agent"].predict(obs,deterministic=True)
        return int(action),None
    probability = float(bundle["predictor"].predict(x,[len(x)-1])[0])
    return int(probability>=bundle["threshold"]),probability


def paper_step(bundle_path, raw, state_path, now=None):
    """Initialize a next-candle intent, or settle exactly one new closed candle.

    Simulated fills use the prior committed intent and the next candle's open;
    settlement waits for that candle to close so OHLC stop handling is possible.
    Missing candles after a restart require review, not retrospective signal invention.
    """
    state_path = Path(state_path)
    state_path.parent.mkdir(parents=True,exist_ok=True)
    lock = state_path.with_suffix(state_path.suffix+".lock")
    descriptor = os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    os.close(descriptor)
    try:
        bundle = load_bundle(bundle_path)
        cfg = bundle["configuration"]
        if raw.index[0] != pd.Timestamp(bundle["history_start"]):
            raise ValueError("Paper input must start at the training history origin")
        cutoff = pd.Timestamp(bundle["trained_through"])
        if digest_candles(raw.loc[:cutoff]) != bundle["history_prefix_sha256"]:
            raise ValueError("Historical input was revised or belongs to another market")
        frame = make_features(raw)
        current = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
        latest,delta = frame.index[-1],pd.Timedelta(cfg.timeframe)
        if current.tzinfo is None or latest+delta > current or latest+2*delta <= current:
            raise ValueError("Paper input is unfinished or stale")
        if latest <= cutoff:
            raise ValueError("Paper decisions must be after all training/validation information")
        identity = hashlib.sha256(Path(bundle_path).read_bytes()).hexdigest()
        if state_path.exists():
            state = json.loads(state_path.read_text())
            if state["artifact_sha256"] != identity:
                raise ValueError("Cannot change models inside an active paper account")
            last = pd.Timestamp(state["last_candle"])
            if latest == last:
                return {"status":"no_new_closed_candle","last_candle":str(last)}
            if latest != last+delta:
                raise ValueError("Missed or out-of-order paper candle; manual reconciliation required")
            broker = Broker.restore(state["broker"])
            broker.process(latest,frame.iloc[-1],state["pending_action"],state["pending_atr"])
            events = state["events"]
        else:
            broker,events = Broker(cfg.risk),[]
        action,probability = intention(bundle,frame,broker)
        if broker.halted or broker.daily_halted:
            action = 0
        event = dict(decision_after=str(latest+delta),action=action,probability=probability,
                     equity=broker.equity(frame.close.iloc[-1]),
                     execution="simulated next-candle open; settled at candle close")
        events.append(event)
        state = dict(artifact_sha256=identity,last_candle=str(latest),pending_action=action,
                     pending_atr=float(frame.atr.iloc[-1]),broker=broker.snapshot(),events=events,
                     symbol=cfg.symbol,timeframe=cfg.timeframe,mode="paper")
        temporary = state_path.with_suffix(state_path.suffix+".tmp")
        temporary.write_text(json.dumps(state,indent=2,allow_nan=False))
        temporary.replace(state_path)
        return dict(status="paper_checkpoint_saved",**event)
    finally:
        lock.unlink(missing_ok=True)
