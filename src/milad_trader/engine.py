"""One execution/accounting engine for backtests, PPO and paper replay.

Decision at close t -> execution at open t+1. Long/flat spot only.
OHLC ambiguity is resolved stop-first. Cash includes both-side fees.
"""
from dataclasses import asdict
import numpy as np
import pandas as pd
from .config import RiskConfig


class Broker:
    def __init__(self, config: RiskConfig):
        self.config = config
        self.cash = config.initial_cash
        self.quantity = 0.0
        self.entry_cost = 0.0
        self.stop = self.take = 0.0
        self.peak = self.last_equity = config.initial_cash
        self.day_start = config.initial_cash
        self.day = None
        self.halted = self.daily_halted = False
        self.last_timestamp = None
        self.fills = []

    def equity(self, price):
        return self.cash + self.quantity * price

    def _gate(self, price):
        value = self.equity(price)
        if value <= self.peak * (1-self.config.max_drawdown):
            self.halted = True
        if value <= self.day_start * (1-self.config.daily_loss):
            self.daily_halted = True
        return self.halted or self.daily_halted

    def sell(self, price, timestamp, reason):
        if self.quantity <= 0:
            return
        fill_price = float(price) * (1-self.config.slippage_bps/10000)
        gross = self.quantity * fill_price
        fee = gross * self.config.fee
        pnl = gross-fee-self.entry_cost
        self.cash += gross-fee
        self.fills.append(dict(timestamp=str(timestamp), side="sell", price=fill_price,
                               quantity=self.quantity, fee=fee, pnl=pnl, reason=reason))
        self.quantity = self.entry_cost = self.stop = self.take = 0.0

    def buy(self, price, atr, timestamp):
        if not np.isfinite(atr) or atr <= 0:
            raise ValueError("Entry requires a positive, previously observed ATR")
        fill_price = float(price) * (1+self.config.slippage_bps/10000)
        distance = atr * self.config.stop_atr
        if distance >= fill_price:
            return
        # Budget estimated stop loss plus approximate round-trip friction.
        loss_per_unit = distance + fill_price * (2*self.config.fee + 2*self.config.slippage_bps/10000)
        quantity = min(self.cash*self.config.allocation/(fill_price*(1+self.config.fee)),
                       self.cash*self.config.risk_per_trade/loss_per_unit)
        if quantity <= 0:
            return
        gross, fee = quantity*fill_price, quantity*fill_price*self.config.fee
        self.entry_cost = gross+fee
        self.cash -= self.entry_cost
        self.quantity = quantity
        self.stop, self.take = fill_price-distance, fill_price+atr*self.config.take_atr
        self.fills.append(dict(timestamp=str(timestamp), side="buy", price=fill_price,
                               quantity=quantity, fee=fee, pnl=None, reason="signal"))

    def process(self, timestamp, bar, desired, previous_atr):
        """Process one complete candle using an intention fixed BEFORE its open."""
        timestamp = pd.Timestamp(timestamp)
        if desired not in (0, 1):
            raise ValueError("Desired position must be 0 (flat) or 1 (long)")
        if self.last_timestamp is not None and timestamp <= pd.Timestamp(self.last_timestamp):
            raise ValueError("Duplicate or out-of-order candle")
        o, h, l, c = (float(bar[k]) for k in ("open", "high", "low", "close"))
        if not all(np.isfinite([o, h, l, c])) or min(o,h,l,c) <= 0 or h < max(o,l,c) or l > min(o,h,c):
            raise ValueError("Invalid execution candle")
        previous = self.last_equity
        day = timestamp.tz_convert("UTC").date().isoformat()
        if self.day != day:
            self.day, self.day_start, self.daily_halted = day, previous, False
        exits = False
        if self.quantity and (o <= self.stop or o >= self.take):
            self.sell(o, timestamp, "gap_stop" if o <= self.stop else "gap_take")
            exits = True
        if self._gate(o):
            self.sell(o, timestamp, "risk_stop")
            exits = True
        elif self.quantity and desired == 0:
            self.sell(o, timestamp, "signal")
            exits = True
        if desired == 1 and not self.quantity and not exits and not self.halted and not self.daily_halted:
            self.buy(o, previous_atr, timestamp)
        if self.quantity:
            if l <= self.stop:
                self.sell(self.stop, timestamp, "stop_loss")
            elif h >= self.take:
                self.sell(self.take, timestamp, "take_profit")
        if self._gate(c):
            self.sell(c, timestamp, "risk_stop_close")
        self.last_equity = self.equity(c)
        self.peak = max(self.peak, self.last_equity)
        self.last_timestamp = timestamp.isoformat()
        return dict(timestamp=timestamp, equity=self.last_equity, cash=self.cash,
                    quantity=self.quantity, net_return=self.last_equity/previous-1,
                    drawdown=self.last_equity/self.peak-1, desired=int(desired),
                    halted=self.halted, daily_halted=self.daily_halted)

    def snapshot(self):
        return {**self.__dict__, "config": asdict(self.config)}

    @classmethod
    def restore(cls, state):
        broker = cls(RiskConfig(**state["config"]))
        for key, value in state.items():
            if key != "config":
                setattr(broker, key, value)
        return broker


def backtest(frame, decisions, config, start=0, end=None, liquidate=True):
    """decisions[j] is observed after candle j closes; evaluate start..end."""
    end = len(frame) if end is None else end
    if end-start < 2 or len(decisions) != len(frame):
        raise ValueError("Need at least two evaluation candles and aligned decisions")
    broker = Broker(config)
    rows = [dict(timestamp=frame.index[start], equity=config.initial_cash, cash=config.initial_cash,
                 quantity=0., net_return=0., drawdown=0., desired=0, halted=False, daily_halted=False)]
    for j in range(start+1, end):
        action = int(decisions[j-1])
        rows.append(broker.process(frame.index[j], frame.iloc[j], action, float(frame.atr.iloc[j-1])))
    if liquidate and broker.quantity:
        broker.sell(frame.close.iloc[end-1], frame.index[end-1], "terminal_liquidation")
        rows[-1].update(equity=broker.cash, cash=broker.cash, quantity=0.,
                        net_return=broker.cash/rows[-2]["equity"]-1,
                        drawdown=broker.cash/broker.peak-1)
    curve = pd.DataFrame(rows).set_index("timestamp")
    return curve, pd.DataFrame(broker.fills)
