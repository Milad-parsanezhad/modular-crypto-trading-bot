import pandas as pd

from research_bot.coinex_public import audit_kline_gaps, fetch_coinex_klines
from research_bot.orderflow import aggregate_order_flow


def test_kline_pagination_with_mock():
    calls = []
    def mock(path, params, timeout):
        calls.append((path, params))
        start = params["start_time"]
        step = 14_400_000
        rows = []
        t = start
        for _ in range(4):
            if t > params["end_time"]:
                break
            rows.append({
                "market": "BTCUSDT", "created_at": t,
                "open": "100", "high": "102", "low": "99", "close": "101",
                "volume": "10", "value": "1000",
            })
            t += step
        return {"code": 0, "data": rows, "message": "OK"}

    start = 1_700_000_000_000
    end = start + 9 * 14_400_000
    df = fetch_coinex_klines("BTC/USDT", "4hour", "spot", start_ms=start, end_ms=end, bars=10, request_fn=mock)
    assert len(df) >= 4
    assert df["timestamp"].is_monotonic_increasing
    audit = audit_kline_gaps(df, "4hour")
    assert audit["rows"] == len(df)


def test_order_flow_aggregation():
    ts = pd.to_datetime(["2026-01-01T00:01:00Z", "2026-01-01T00:02:00Z", "2026-01-01T01:00:00Z"])
    trades = pd.DataFrame({
        "timestamp": ts,
        "deal_id": [1, 2, 3],
        "side": ["buy", "sell", "buy"],
        "price": [100.0, 100.0, 101.0],
        "amount": [2.0, 1.0, 1.0],
        "notional": [200.0, 100.0, 101.0],
        "market": ["BTCUSDT"] * 3,
    })
    agg = aggregate_order_flow(trades, "4h")
    assert len(agg) == 1
    row = agg.iloc[0]
    assert row["trade_count"] == 3
    assert row["buy_count"] == 2
    assert row["sell_count"] == 1
    assert row["quote_imbalance"] > 0
