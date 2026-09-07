"""Daily and weekly performance reports - گزارش‌های عملکرد روزانه و هفتگی."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List

import pandas as pd

logger = logging.getLogger(__name__)


class PerformanceReporter:
    """Generate performance reports for monitoring."""

    def __init__(self, results_dir: str = "results"):
        """Initialize reporter."""
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)

    def load_trade_logs(self, days: int = 7) -> List[Dict[str, Any]]:
        """Load recent trade logs."""
        trades = []
        cutoff = datetime.now() - timedelta(days=days)
        
        for log_file in self.results_dir.glob("*.jsonl"):
            try:
                with open(log_file) as f:
                    for line in f:
                        event = json.loads(line)
                        if event.get("type") == "trade_executed":
                            timestamp = datetime.fromisoformat(event["timestamp"])
                            if timestamp > cutoff:
                                trades.append(event["data"])
            except Exception as e:
                logger.error(f"Error reading {log_file}: {e}")
        
        return trades

    def generate_daily_report(self) -> Dict[str, Any]:
        """Generate daily performance report."""
        trades = self.load_trade_logs(days=1)
        
        winning_trades = len([t for t in trades if t.get("pnl", 0) > 0])
        losing_trades = len([t for t in trades if t.get("pnl", 0) < 0])
        
        report = {
            "date": datetime.now().date().isoformat(),
            "total_trades": len(trades),
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "daily_pnl": sum(t.get("pnl", 0) for t in trades),
            "avg_win": sum(t.get("pnl", 0) for t in trades if t.get("pnl", 0) > 0) / max(winning_trades, 1),
            "avg_loss": sum(t.get("pnl", 0) for t in trades if t.get("pnl", 0) < 0) / max(losing_trades, 1),
            "trades": trades,
        }
        
        # Save
        report_file = self.results_dir / f"daily_report_{report['date']}.json"
        report_file.write_text(json.dumps(report, indent=2, default=str))
        logger.info(f"Daily report saved: {report_file}")
        
        return report

    def generate_weekly_report(self) -> Dict[str, Any]:
        """Generate weekly performance report."""
        trades = self.load_trade_logs(days=7)
        
        winning = len([t for t in trades if t.get("pnl", 0) > 0])
        total = len(trades)
        
        report = {
            "week_ending": datetime.now().date().isoformat(),
            "total_trades": total,
            "winning_trades": winning,
            "win_rate": winning / max(total, 1),
            "total_pnl": sum(t.get("pnl", 0) for t in trades),
            "avg_trade_pnl": sum(t.get("pnl", 0) for t in trades) / max(total, 1),
        }
        
        report_file = self.results_dir / f"weekly_report_{report['week_ending']}.json"
        report_file.write_text(json.dumps(report, indent=2, default=str))
        logger.info(f"Weekly report saved: {report_file}")
        
        return report

    def print_performance_summary(self) -> None:
        """Print current performance summary."""
        daily = self.generate_daily_report()
        
        print("\n" + "="*60)
        print("DAILY PERFORMANCE SUMMARY")
        print("="*60)
        print(f"Date: {daily['date']}")
        print(f"Trades: {daily['total_trades']} ({daily['winning_trades']} wins, {daily['losing_trades']} losses)")
        print(f"Daily PnL: ${daily['daily_pnl']:.2f}")
        print(f"Avg Win: ${daily['avg_win']:.2f} | Avg Loss: ${daily['avg_loss']:.2f}")
        print("="*60 + "\n")
