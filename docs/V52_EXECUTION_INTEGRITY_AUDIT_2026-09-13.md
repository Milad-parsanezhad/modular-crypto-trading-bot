# v0.52 Execution Integrity Audit

Date: 2026-09-13

Status: **PAPER execution hardening only — LIVE remains prohibited**

## 1. Scope

This audit continues the execution defects identified during thesis-system review. It does not alter the trading hypothesis, model-selection evidence, or v0.51 economic protocol. It hardens the PAPER execution path against market-data staleness, crash inconsistency, duplicate settlement, and restart gaps.

## 2. Confirmed defects

### E1 — Candle-close price could be used as if executable

The v0.14 forward PAPER path generated a signal from the most recent closed 4h candle, fetched a live depth snapshot, but still passed `signal.reference_price` (the candle close) into the execution engine. A large move between candle close and current top-of-book could therefore create a phantom paper fill at a stale economic reference.

**Fix:** candle close is now signal evidence only. BUY uses current `best_ask`, SELL uses current `best_bid`; spread is therefore represented structurally by crossing the book. Configured slippage is then applied on top of that executable-side quote.

### E2 — Quote freshness was not an execution gate

A depth snapshot could be internally valid but old.

**Fix:** executable quotes now require a timezone-aware timestamp, positive finite bid/ask/mid, non-crossed book, maximum age (default 30 seconds), and bounded future clock skew (default 5 seconds). Failure is `STALE_OR_INVALID_QUOTE`; no fill or account mutation is allowed.

### E3 — Cash/position mutation and fill ledger were separate commits

The old path updated position and cash before recording the fill. A crash between those operations could leave economic state changed with no accounting evidence.

**Fix:** `commit_fill_atomic()` commits cash, position and fill ledger as one logical settlement. PostgreSQL uses one ACID transaction with row locking. The memory backend implements rollback semantics for deterministic tests.

### E4 — Process restart could bypass in-memory duplicate protection

`PaperExecutionEngine._seen_ids` is process-local. After restart the same deterministic `client_order_id` could be attempted again.

**Fix:** idempotency is now enforced at persistence level. Existing `client_order_id` is checked before state mutation; repeated settlement raises `DuplicateSettlementError` and changes neither cash nor position.

### E5 — Observation-only crash seam

Observation was committed before settlement. A crash after the observation insert but before a fill could cause the next run to stop at `ALREADY_OBSERVED_BAR`, permanently skipping an actionable paper order.

**Fix:** duplicate observation and duplicate settlement are now separate states. If an actionable observation exists but no persisted settlement exists, the runner may perform a bounded recovery attempt using a new fresh quote and the existing signal only while the signal-age gate remains valid. If settlement already exists, the result is `ALREADY_SETTLED_ORDER`.

## 3. Economic cost semantics

The hardened path avoids double-counting spread:

- signal reference = closed-candle close (research evidence only),
- execution reference = ask for BUY / bid for SELL,
- spread cost = embedded in crossing mid → bid/ask,
- modeled slippage = additional movement from executable quote,
- fee = charged on filled notional,
- risk pre-check still uses an all-in estimate of configured slippage + half-spread.

The fill metadata preserves `signal_reference_price`, `execution_reference_price`, `mark_price`, `spread_bps`, quote timestamp/age, and recovery status so realized paper costs can be reconstructed rather than inferred from one price field.

## 4. Tests added

The v0.52 suite includes:

- stale quote rejection;
- future timestamp rejection;
- BUY-at-ask / SELL-at-bid semantics;
- deliberate 10% candle-vs-book price divergence to prove no stale-candle fill;
- memory settlement rollback after injected ledger failure;
- PostgreSQL rollback after a deliberately failed final fill INSERT;
- persistent duplicate-order idempotency;
- restart recovery when observation exists but settlement does not;
- existing wide-spread risk rejection and PAPER-only contract.

The PostgreSQL test is intentionally failure-injection based: position/account SQL executes first, the final ledger INSERT is forced to fail, and the test verifies that the database transaction rolls back all prior state changes.

## 5. Research/engineering basis

### Stale market data

Stale quotes can remain transport-valid while no longer representing a tradable market. For research-grade execution, timing and agreement with the current market state must be explicit gates. The implementation therefore treats quote age as execution evidence, not metadata only.

Reference: Sonar Sciences, *How to detect stale quotes in a feed* (2026-08-07).

### Atomic state + event/ledger persistence

The dual-write problem is well known: committing business state and its event/accounting record separately creates an inconsistency window. The transactional-outbox literature recommends one local ACID transaction plus idempotent downstream processing. v0.52 applies the same invariant locally to paper settlement: **cash + position + fill record land together or none land**.

Reference: AWS Prescriptive Guidance, *Transactional outbox pattern*.

### Exchange order identity and reconciliation

CoinEx v2 supports a user-defined `client_id` and exposes order-management endpoints keyed by that identifier. This supports the future live/testnet design in which deterministic client IDs are persisted before submission and queried after reconnect rather than blindly re-submitting.

References: CoinEx API v2 `POST /spot/order`, cancel/query endpoints using `client_id`.

## 6. Remaining boundary before any private exchange execution

v0.52 does **not** authorize TESTNET or LIVE. A real exchange adapter still requires a persistent order-state machine and reconnect reconciliation against venue truth:

`CREATED -> SUBMITTED -> ACKNOWLEDGED -> PARTIAL -> FILLED/CANCELLED/REJECTED`

On an ambiguous network failure, the required rule is **query by deterministic client ID before any retry**. No private API order submission should be enabled until this state machine, exchange-specific reconciliation, fee-currency handling, partial-fill aggregation, and balance reconciliation have their own failure-injection tests.

## 7. Claim boundary

This audit improves software correctness and the realism of PAPER execution. It is **not evidence of strategy profitability, statistical alpha, or live readiness**. Existing empirical promotion gates remain unchanged and LIVE execution remains fail-closed.
