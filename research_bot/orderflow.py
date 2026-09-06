from __future__ import annotations

import pandas as pd


def aggregate_order_flow(trades: pd.DataFrame, freq: str = "4h") -> pd.DataFrame:
    """Aggregate public taker-side trades into order-flow features."""
    if trades.empty:
        return pd.DataFrame(columns=["timestamp","trade_count","buy_count","sell_count","buy_amount","sell_amount","buy_notional","sell_notional","base_imbalance","quote_imbalance","count_imbalance","avg_trade_notional"])
    x = trades.copy()
    x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True)
    x["is_buy"] = (x["side"].str.lower() == "buy").astype(int)
    x["is_sell"] = (x["side"].str.lower() == "sell").astype(int)
    x["buy_amount"] = x["amount"].where(x["is_buy"].eq(1), 0.0)
    x["sell_amount"] = x["amount"].where(x["is_sell"].eq(1), 0.0)
    x["buy_notional"] = x["notional"].where(x["is_buy"].eq(1), 0.0)
    x["sell_notional"] = x["notional"].where(x["is_sell"].eq(1), 0.0)
    x = x.set_index("timestamp")
    agg = x.resample(freq).agg(
        trade_count=("deal_id", "count"), buy_count=("is_buy", "sum"), sell_count=("is_sell", "sum"),
        buy_amount=("buy_amount", "sum"), sell_amount=("sell_amount", "sum"),
        buy_notional=("buy_notional", "sum"), sell_notional=("sell_notional", "sum"),
        avg_trade_notional=("notional", "mean"),
    )
    eps = 1e-12
    agg["base_imbalance"] = (agg["buy_amount"] - agg["sell_amount"]) / (agg["buy_amount"] + agg["sell_amount"] + eps)
    agg["quote_imbalance"] = (agg["buy_notional"] - agg["sell_notional"]) / (agg["buy_notional"] + agg["sell_notional"] + eps)
    agg["count_imbalance"] = (agg["buy_count"] - agg["sell_count"]) / (agg["buy_count"] + agg["sell_count"] + eps)
    return agg.reset_index()


def order_flow_summary(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"trades": 0, "coverage_start": None, "coverage_end": None}
    buy = trades.loc[trades["side"].str.lower().eq("buy")]
    sell = trades.loc[trades["side"].str.lower().eq("sell")]
    buy_notional = float(buy["notional"].sum())
    sell_notional = float(sell["notional"].sum())
    denom = buy_notional + sell_notional
    return {
        "trades": int(len(trades)),
        "coverage_start": pd.to_datetime(trades["timestamp"], utc=True).min().isoformat(),
        "coverage_end": pd.to_datetime(trades["timestamp"], utc=True).max().isoformat(),
        "buy_trade_share": float(len(buy) / len(trades)),
        "buy_notional": buy_notional,
        "sell_notional": sell_notional,
        "total_notional": float(trades["notional"].sum()),
        "quote_imbalance": float((buy_notional - sell_notional) / denom) if denom else None,
    }
