# v0.52 Nobitex Reconciliation & Real-Cost Contract

Date: 2026-09-13

Status: **design/test contract only — no private API call, no TESTNET/LIVE authorization**

## 1. Why this contract exists

A safe exchange adapter must assume that the network can fail after the exchange has accepted an order but before the client receives the response. Blindly resubmitting in that situation can duplicate economic exposure. The adapter therefore needs a durable local order identity and a reconciliation path against venue truth.

## 2. Nobitex facts used by this design

The current Nobitex API documentation exposes:

- `POST /market/orders/add` for spot order creation;
- a user-supplied `clientOrderId` (up to 32 characters; documented as experimental);
- `POST /market/orders/status` by numeric order id or `clientOrderId`;
- `GET /market/orders/list` with detailed order state, `matchedAmount`, `averagePrice` and `fee`;
- `GET /market/trades/list` for actual user trades including order id, execution price, amount, total and fee;
- order states including New/Active/Inactive/Done/Canceled.

Important limitation: the documentation states that status lookup by `clientOrderId` searches only open/active/inactive orders. Consequently, a production-grade adapter must persist the numeric venue order id as soon as it becomes known and must be able to fall back to order lists/trades for terminal reconciliation.

## 3. Frozen local lifecycle

The adapter state machine is:

`CREATED`
→ `SUBMITTING`
→ either `ACKNOWLEDGED` or `UNKNOWN_PENDING_RECONCILIATION`
→ `PARTIAL`
→ `FILLED` / `CANCELLED` / `REJECTED`

Rules:

1. An ambiguous timeout after submit is **not** a rejection.
2. `UNKNOWN_PENDING_RECONCILIATION` forbids a blind resubmit.
3. Reconciliation must query venue truth using the deterministic client id and, when known, venue order id.
4. `Canceled` is terminal but may still contain a non-zero `matchedAmount`; those fills must remain in positions/accounting.
5. `Done` is accepted as fully filled only when matched amount is consistent with requested amount.
6. A terminal venue state may not be overwritten by a weaker local assumption.

## 4. Identity

`research_bot.order_reconciliation_v52.deterministic_client_order_id()` generates a <=32-character stable identifier from strategy version, symbol, decision epoch and side. The original semantic tuple remains persisted locally; the external client ID is only its deterministic compact identity.

Both identifiers must be stored:

- `client_order_id` — deterministic client identity;
- `venue_order_id` — numeric/venue identity returned after acknowledgement.

A mismatch in either identity is fail-closed.

## 5. Reconnect algorithm

On process startup or reconnect:

1. load every non-terminal local order;
2. for `SUBMITTING` or `UNKNOWN_PENDING_RECONCILIATION`, query status before any new submission;
3. if the client-ID query finds an open order, persist the venue id and reconcile quantities;
4. if the client-ID query cannot find the order, do **not** infer rejection from that fact alone because terminal orders may not be searchable by that path;
5. search detailed order history/list and user trades within the bounded decision-time window;
6. reconcile cumulative `matchedAmount`, actual trade fills and fees;
7. only after venue absence has been established by the complete frozen procedure may the order become a candidate for controlled resubmission; such resubmission must create a new explicit attempt record rather than silently reusing an ambiguous state.

## 6. Realized cost accounting

Paper cost assumptions and realized venue costs must be stored separately.

For each actual trade fill:

- quantity = venue trade `amount`;
- price = venue trade `price`;
- gross quote notional = sum(quantity × price);
- VWAP = gross quote notional / total filled quantity;
- implementation shortfall for BUY = `(VWAP - decision_reference) × quantity`;
- implementation shortfall for SELL = `(decision_reference - VWAP) × quantity`;
- explicit fee = venue-reported fee.

The Nobitex trade/order schema exposes a fee value, but the cited object schema does not make fee denomination sufficiently explicit for a generic converter. The implementation therefore keeps `fee_reported` and `fee_currency` separate and refuses to report `total_cost_quote` unless the fee-to-quote conversion is known. **No guessed fee currency is allowed.**

## 7. Fixed precision requirement for a private adapter

The Nobitex documentation recommends fixed-precision numeric handling for important monetary calculations. The current v0.52 PAPER path still uses Python floats because it is a simulator inherited from v0.14. A future private Nobitex adapter must use `Decimal`/fixed precision at the API/accounting boundary, quantized to venue market precision, before TESTNET/LIVE review.

## 8. Required tests before private execution

A future adapter is blocked until all of the following are demonstrated with mocks/test environment or equivalent safe evidence:

- timeout after the venue accepted the order but before HTTP response;
- retry after ambiguous timeout does not duplicate an order;
- reconnect with Active order;
- reconnect with partial fill;
- partial fill followed by cancel;
- fully Done order;
- client-id mismatch and venue-id mismatch;
- order status unavailable by client id but found by order history/trades;
- fee aggregation across multiple trade fills;
- fee denomination unavailable → total quote cost remains unknown, not guessed;
- balance reconciliation after partial/terminal states;
- process crash between local submit intent and venue acknowledgement;
- process crash after venue acknowledgement but before local persistence.

## 9. Safety boundary

This specification does not submit an order, does not use exchange credentials, and does not authorize live trading. The project remains PAPER/research-only until the empirical strategy gates, prospective evidence, adapter integration tests and explicit live-readiness review are all satisfied.
