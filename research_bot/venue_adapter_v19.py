from __future__ import annotations


def normalized_orderbook_limit(exchange_id: str, requested: int) -> int:
    """Map a generic depth request onto exchange-specific CCXT constraints.

    The function is deliberately explicit instead of silently retrying arbitrary
    values so the data contract is deterministic and unit-testable.
    """
    exchange_id = str(exchange_id).lower().strip()
    requested = max(1, int(requested))
    if exchange_id == "kucoin":
        # CCXT KuCoin fetchOrderBook accepts 20 or 100.
        return 20 if requested <= 20 else 100
    return requested
