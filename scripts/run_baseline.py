"""Production baseline with enhanced reporting and error handling."""
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from research_bot.data import fetch_with_fallback
from research_bot.features import add_features
from research_bot.regime import add_regime_features
from research_bot.model import fit_predict_holdout, positions_from_probabilities
from research_bot.backtest import backtest_positions, performance_metrics
from research_bot.validators import DataValidator
from research_bot.monitoring import StructuredLogger

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _json_clean(obj: Any) -> Any:
    """Convert numpy types to JSON-serializable types."""
    if isinstance(obj, dict):
        return {k: _json_clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if pd.isna(obj):
        return None
    return obj


def run_baseline(
    exchange: str = "coinex",
    symbol: str = "BTC/USDT",
    timeframe: str = "4h",
    limit: int = 2000,
    fee_bps: float = 10.0,
    slippage_bps: float = 2.0,
    upper_threshold: float = 0.56,
    output: str = "results/baseline_report.json",
) -> Dict[str, Any]:
    """Run complete baseline strategy with validation."""
    
    logger.info(f"Starting baseline: {symbol} {timeframe} on {exchange}")
    
    # Fetch data with fallback
    fallback_exchanges = [exchange] + [x for x in ["coinex", "kraken", "okx"] if x != exchange]
    try:
        raw, used_exchange = fetch_with_fallback(
            fallback_exchanges,
            symbol=symbol,
            timeframe=timeframe,
            limit=limit
        )
        logger.info(f"Fetched {len(raw)} bars from {used_exchange}")
    except Exception as e:
        logger.error(f"Data fetch failed: {e}")
        raise
    
    # Validate OHLCV
    is_valid, issues = DataValidator.check_ohlcv_integrity(raw)
    if not is_valid:
        logger.warning(f"OHLCV validation issues: {issues}")
    
    # Feature engineering
    try:
        feat = add_regime_features(add_features(raw))
        logger.info(f"Features computed: {len(feat)} rows")
    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        raise
    
    # Model training
    try:
        model, train, test, cols = fit_predict_holdout(
            feat,
            train_fraction=0.70,
            hurdle_bps=fee_bps + slippage_bps
        )
        logger.info(f"Model trained: {len(train)} train, {len(test)} test")
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        raise
    
    # Position generation
    try:
        positions = positions_from_probabilities(
            test["prob_up"],
            upper=upper_threshold,
            allow_short=False
        )
        logger.info(f"Positions: {(positions > 0).sum()} long, {(positions == 0).sum()} flat")
    except Exception as e:
        logger.error(f"Position generation failed: {e}")
        raise
    
    # Backtesting
    try:
        # ML strategy
        _, ml_metrics = backtest_positions(
            test["future_return"],
            positions,
            timeframe=timeframe,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps
        )
        
        # Buy & Hold baseline
        bh_metrics = performance_metrics(test["future_return"], timeframe=timeframe)
        
        # Momentum baseline
        momentum_pos = (test["mom_24"] > 0).astype(float)
        _, mom_metrics = backtest_positions(
            test["future_return"],
            momentum_pos,
            timeframe=timeframe,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps
        )
        
        logger.info(f"Backtests complete | ML Sharpe: {ml_metrics.get('sharpe', np.nan):.2f}")
    except Exception as e:
        logger.error(f"Backtesting failed: {e}")
        raise
    
    # Compile report
    report = {
        "research_status": "baseline_v0.2_not_production",
        "timestamp": pd.Timestamp.now().isoformat(),
        "exchange": used_exchange,
        "symbol": symbol,
        "timeframe": timeframe,
        "data": {
            "bars_raw": int(len(raw)),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "feature_count": len(cols),
        },
        "costs": {
            "fee_bps": fee_bps,
            "slippage_bps": slippage_bps,
        },
        "model_threshold": upper_threshold,
        "metrics": {
            "ml_strategy": ml_metrics,
            "buy_and_hold": bh_metrics,
            "momentum_24h": mom_metrics,
        },
        "warnings": issues if not is_valid else [],
    }
    
    # Save report
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_json_clean(report), indent=2), encoding="utf-8")
    logger.info(f"Report saved: {out_path}")
    
    return report


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run modular crypto trading bot baseline"
    )
    parser.add_argument("--exchange", default="coinex", help="Exchange name (CCXT)")
    parser.add_argument("--symbol", default="BTC/USDT", help="Trading pair")
    parser.add_argument("--timeframe", default="4h", help="Candle timeframe")
    parser.add_argument("--limit", type=int, default=2000, help="Number of candles")
    parser.add_argument("--fee-bps", type=float, default=10.0, help="Fee in basis points")
    parser.add_argument("--slippage-bps", type=float, default=2.0, help="Slippage in basis points")
    parser.add_argument("--upper", type=float, default=0.56, help="Position threshold")
    parser.add_argument("--output", default="results/baseline_report.json", help="Output file")
    
    args = parser.parse_args()
    
    try:
        report = run_baseline(
            exchange=args.exchange,
            symbol=args.symbol,
            timeframe=args.timeframe,
            limit=args.limit,
            fee_bps=args.fee_bps,
            slippage_bps=args.slippage_bps,
            upper_threshold=args.upper,
            output=args.output,
        )
        
        print("\n" + "="*60)
        print("BASELINE REPORT")
        print("="*60)
        print(json.dumps(_json_clean(report), indent=2))
        print("="*60)
        
    except Exception as e:
        logger.exception(f"Baseline run failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()
