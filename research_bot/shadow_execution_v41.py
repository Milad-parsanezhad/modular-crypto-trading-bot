from __future__ import annotations

"""v0.41 shadow/paper execution engine.

This module intentionally contains no live broker implementation. It validates the
operational path—idempotency, pre-trade risk, kill switch, fills and audit events—
without transmitting a real order. LIVE execution is structurally forbidden.
"""

from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Any
import json


LIVE_EXECUTION = False


class LiveExecutionForbidden(RuntimeError):
    pass


class RiskRejected(RuntimeError):
    pass


@dataclass(frozen=True)
class RiskLimitsV41:
    max_order_notional: float = 1000.0
    max_open_positions: int = 10
    max_gross_notional: float = 5000.0
    max_daily_loss_fraction: float = 0.02


@dataclass(frozen=True)
class OrderIntentV41:
    symbol: str
    side: str
    quantity: float
    reference_price: float
    signal_time: str
    strategy: str

    @property
    def notional(self) -> float:
        return abs(float(self.quantity) * float(self.reference_price))


@dataclass(frozen=True)
class PaperFillV41:
    client_order_id: str
    symbol: str
    side: str
    quantity: float
    fill_price: float
    fee: float
    mode: str = "paper"


class ShadowExecutionEngineV41:
    def __init__(self, limits: RiskLimitsV41 | None = None, *, fee_bps: float = 10.0, slippage_bps: float = 5.0):
        self.limits = limits or RiskLimitsV41()
        self.fee_bps = float(fee_bps)
        self.slippage_bps = float(slippage_bps)
        self.kill_switch = False
        self.open_positions: dict[str, float] = {}
        self.processed_ids: set[str] = set()
        self.audit: list[dict[str, Any]] = []
        self.realized_day_pnl_fraction = 0.0

    @staticmethod
    def client_order_id(intent: OrderIntentV41) -> str:
        raw = f"{intent.strategy}|{intent.symbol}|{intent.side}|{intent.quantity:.12g}|{intent.reference_price:.12g}|{intent.signal_time}"
        return "v41-" + sha256(raw.encode("utf-8")).hexdigest()[:24]

    def _gross_notional(self, marks: dict[str, float]) -> float:
        total = 0.0
        for symbol, qty in self.open_positions.items():
            px = float(marks.get(symbol, 0.0))
            total += abs(float(qty) * px)
        return total

    def _risk_check(self, intent: OrderIntentV41, marks: dict[str, float]) -> None:
        if self.kill_switch:
            raise RiskRejected("KILL_SWITCH_ACTIVE")
        if intent.side not in {"buy", "sell"}:
            raise RiskRejected("INVALID_SIDE")
        if intent.quantity <= 0 or intent.reference_price <= 0:
            raise RiskRejected("INVALID_SIZE_OR_PRICE")
        if intent.notional > self.limits.max_order_notional + 1e-12:
            raise RiskRejected("MAX_ORDER_NOTIONAL")
        if intent.symbol not in self.open_positions and len(self.open_positions) >= self.limits.max_open_positions:
            raise RiskRejected("MAX_OPEN_POSITIONS")
        projected = self._gross_notional(marks) + intent.notional
        if projected > self.limits.max_gross_notional + 1e-12:
            raise RiskRejected("MAX_GROSS_NOTIONAL")
        if self.realized_day_pnl_fraction <= -self.limits.max_daily_loss_fraction:
            raise RiskRejected("DAILY_LOSS_LIMIT")

    def submit_paper(self, intent: OrderIntentV41, *, marks: dict[str, float] | None = None) -> PaperFillV41:
        marks = dict(marks or {intent.symbol: intent.reference_price})
        cid = self.client_order_id(intent)
        if cid in self.processed_ids:
            raise RiskRejected("DUPLICATE_CLIENT_ORDER_ID")
        self._risk_check(intent, marks)
        direction = 1.0 if intent.side == "buy" else -1.0
        fill_price = float(intent.reference_price) * (1.0 + direction * self.slippage_bps / 10000.0)
        fee = abs(float(intent.quantity) * fill_price) * self.fee_bps / 10000.0
        signed_qty = float(intent.quantity) if intent.side == "buy" else -float(intent.quantity)
        self.open_positions[intent.symbol] = self.open_positions.get(intent.symbol, 0.0) + signed_qty
        if abs(self.open_positions[intent.symbol]) < 1e-12:
            self.open_positions.pop(intent.symbol, None)
        self.processed_ids.add(cid)
        fill = PaperFillV41(cid, intent.symbol, intent.side, float(intent.quantity), fill_price, fee)
        self.audit.append({"event": "paper_fill", **asdict(fill), "signal_time": intent.signal_time, "strategy": intent.strategy})
        return fill

    def submit_shadow(self, intent: OrderIntentV41, *, marks: dict[str, float] | None = None) -> dict[str, Any]:
        marks = dict(marks or {intent.symbol: intent.reference_price})
        cid = self.client_order_id(intent)
        if cid in self.processed_ids:
            raise RiskRejected("DUPLICATE_CLIENT_ORDER_ID")
        self._risk_check(intent, marks)
        self.processed_ids.add(cid)
        event = {"event": "shadow_intent", "client_order_id": cid, **asdict(intent), "transmitted": False}
        self.audit.append(event)
        return event

    def submit_live(self, *args: Any, **kwargs: Any) -> None:
        raise LiveExecutionForbidden("LIVE_EXECUTION=false: real order transmission is forbidden in v0.41")

    def activate_kill_switch(self, reason: str) -> None:
        self.kill_switch = True
        self.audit.append({"event": "kill_switch", "active": True, "reason": str(reason)})

    def reconcile(self, broker_positions: dict[str, float]) -> dict[str, Any]:
        expected = {k: float(v) for k, v in sorted(self.open_positions.items())}
        observed = {k: float(v) for k, v in sorted(broker_positions.items())}
        ok = expected == observed
        event = {"event": "reconcile", "ok": ok, "expected": expected, "observed": observed}
        self.audit.append(event)
        if not ok:
            self.activate_kill_switch("POSITION_RECONCILIATION_MISMATCH")
        return event

    def audit_jsonl(self) -> str:
        return "\n".join(json.dumps(x, sort_keys=True, default=str) for x in self.audit) + ("\n" if self.audit else "")


def preregistration_manifest_v41() -> dict[str, Any]:
    return {
        "version": "v0.41",
        "experiment": "SHADOW_PAPER_EXECUTION_ENGINE",
        "network_order_transmission": False,
        "live_execution": False,
        "features": ["deterministic client order id", "idempotency", "pre-trade limits", "kill switch", "paper fills", "shadow intents", "reconciliation", "audit log"],
        "manual_human_approval_required_for_any_future_live_path": True,
        "kraken_touched": False,
    }


def decision_v41(parent_v40_valid: bool, engine_validated: bool) -> dict[str, Any]:
    if parent_v40_valid and engine_validated:
        state = "V41_SUPERVISED_PAPER_PATH_READY"
    elif engine_validated:
        state = "V41_SHADOW_ENGINE_VALIDATED_RESEARCH_BLOCKED"
    else:
        state = "V41_SHADOW_ENGINE_INVALID"
    return {
        "version": "v0.41",
        "decision": state,
        "parent_v40_valid": bool(parent_v40_valid),
        "engine_validated": bool(engine_validated),
        "paper_path_ready": bool(parent_v40_valid and engine_validated),
        "kraken_touched": False,
        "real_order_transmission": False,
        "paper_replacement_authorized": False,
        "live_execution_authorized": False,
    }
