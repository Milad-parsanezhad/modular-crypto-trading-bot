"""PPO uses the same broker, costs and risk rules as supervised strategies."""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .engine import Broker


def observation(x, j, lookback, broker, price):
    value = broker.equity(price)
    state = [broker.quantity*price/max(value,1e-12), value/broker.config.initial_cash-1,
             value/broker.peak-1, value/broker.day_start-1,
             float(broker.halted), float(broker.daily_halted),
             broker.stop/price-1 if broker.quantity else 0,
             broker.take/price-1 if broker.quantity else 0]
    return np.r_[x[j-lookback+1:j+1].flatten(),state].astype(np.float32)


class TradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, frame, normalized, risk, lookback=24, random_start=False, episode_bars=720):
        self.frame, self.x, self.risk, self.lookback = frame,normalized,risk,lookback
        if len(frame) <= lookback or episode_bars < 1:
            raise ValueError("Need history and at least one actionable training candle")
        self.random_start, self.episode_bars = random_start, episode_bars
        self.episode_end = len(frame)-1
        self.sampled_starts = []
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-np.inf,np.inf,shape=(lookback*normalized.shape[1]+8,),dtype=np.float32)
        self.broker = Broker(risk)
        self.j = lookback-1

    def _obs(self):
        return observation(self.x,self.j,self.lookback,self.broker,self.frame.close.iloc[self.j])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.broker = Broker(self.risk)
        self.j = self.lookback-1
        if self.random_start:
            minimum = min(128, len(self.frame)-self.lookback)
            self.j = int(self.np_random.integers(self.lookback-1, len(self.frame)-minimum))
        self.episode_end = min(len(self.frame)-1, self.j+self.episode_bars) if self.random_start else len(self.frame)-1
        self.sampled_starts.append(self.j)
        return self._obs(), {}

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError("Invalid action")
        if self.j >= self.episode_end or self.broker.halted:
            raise RuntimeError("Episode ended; call reset")
        previous = self.broker.last_equity
        atr = self.frame.atr.iloc[self.j]
        self.j += 1
        self.broker.process(self.frame.index[self.j], self.frame.iloc[self.j],int(action),atr)
        terminated = bool(self.broker.halted)
        truncated = self.j == self.episode_end
        if terminated or truncated:
            self.broker.sell(self.frame.close.iloc[self.j],self.frame.index[self.j],"terminal_liquidation")
            self.broker.last_equity = self.broker.cash
        reward = float(np.log(self.broker.last_equity/previous))
        return self._obs(),reward,terminated,truncated,{"equity":self.broker.last_equity}


def fit_ppo(frame, values, train_end, config, seed):
    from sklearn.preprocessing import StandardScaler
    from stable_baselines3 import PPO
    from .models import seed_everything
    seed_everything(seed)
    scaler = StandardScaler().fit(values[:train_end])
    env = TradingEnv(frame.iloc[:train_end],scaler.transform(values[:train_end]),config.risk,config.lookback,
                     random_start=True,episode_bars=720)
    agent = PPO("MlpPolicy",env,seed=seed,verbose=0,device="cpu",n_steps=128,batch_size=64,
                n_epochs=5,learning_rate=3e-4,gamma=.99,policy_kwargs={"net_arch":[32,32]})
    agent.learn(total_timesteps=config.ppo_steps)
    agent.research_sampling = dict(method="seeded uniform starts inside training only",
                                   maximum_episode_bars=720, sampled_starts=env.sampled_starts)
    return agent,scaler


def evaluate_ppo(frame, values, start, end, agent, scaler, config):
    import pandas as pd
    broker, rows = Broker(config.risk), []
    x = scaler.transform(values)
    rows.append(dict(timestamp=frame.index[start],equity=config.risk.initial_cash,net_return=0.))
    for j in range(start,end-1):
        obs = observation(x,j,config.lookback,broker,frame.close.iloc[j])
        action,_ = agent.predict(obs,deterministic=True)
        rows.append(broker.process(frame.index[j+1],frame.iloc[j+1],int(action),frame.atr.iloc[j]))
    if broker.quantity:
        broker.sell(frame.close.iloc[end-1],frame.index[end-1],"terminal_liquidation")
        rows[-1]["equity"] = broker.cash
        rows[-1]["net_return"] = broker.cash/rows[-2]["equity"]-1
    return pd.DataFrame(rows).set_index("timestamp"),pd.DataFrame(broker.fills)
