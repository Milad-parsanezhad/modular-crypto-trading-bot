"""Logging, monitoring, and alerting."""
from __future__ import annotations

import logging
import logging.handlers
from datetime import datetime
from pathlib import Path
import json
from typing import Any, Dict


class StructuredLogger:
    """Structured logging for reproducibility and debugging."""

    def __init__(self, name: str, log_dir: str = "logs"):
        """Initialize structured logger."""
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Create logs directory
        Path(log_dir).mkdir(exist_ok=True)
        
        # File handler
        fh = logging.FileHandler(f"{log_dir}/{name}.log")
        fh.setLevel(logging.DEBUG)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log structured event."""
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": event_type,
            "data": data,
        }
        self.logger.info(json.dumps(event))

    def log_error(self, error: Exception, context: Dict[str, Any]) -> None:
        """Log error with context."""
        self.logger.error(f"Error: {str(error)}", extra={"context": context})

    def log_backtest_result(self, symbol: str, metrics: Dict[str, Any]) -> None:
        """Log backtest completion."""
        self.log_event("backtest_complete", {
            "symbol": symbol,
            "metrics": metrics,
        })
