from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from typing import Any


@dataclass(frozen=True)
class PaperAccount:
    cash: float
    equity: float
    peak_equity: float


@dataclass(frozen=True)
class PaperPosition:
    symbol: str
    quantity: float
    avg_price: float


class MemoryPaperStore:
    """Deterministic in-memory store used by tests and local fallback."""

    def __init__(self, initial_cash: float = 10_000.0):
        self._account = PaperAccount(initial_cash, initial_cash, initial_cash)
        self._positions: dict[str, PaperPosition] = {}
        self._observations: list[dict] = []
        self._fills: list[dict] = []
        self._equity: list[dict] = []
        self._obs_keys: set[tuple[str, str, str]] = set()

    def ensure_schema(self) -> None:
        return None

    def record_observation(self, row: dict) -> bool:
        key = (str(row["bar_timestamp"]), str(row["symbol"]), str(row["strategy_version"]))
        if key in self._obs_keys:
            return False
        self._obs_keys.add(key)
        self._observations.append(dict(row))
        return True

    def get_account(self) -> PaperAccount:
        return self._account

    def set_account(self, account: PaperAccount) -> None:
        self._account = account

    def get_position(self, symbol: str) -> PaperPosition:
        return self._positions.get(symbol, PaperPosition(symbol, 0.0, 0.0))

    def set_position(self, position: PaperPosition) -> None:
        if abs(position.quantity) < 1e-15:
            self._positions.pop(position.symbol, None)
        else:
            self._positions[position.symbol] = position

    def list_positions(self) -> list[PaperPosition]:
        return list(self._positions.values())

    def record_fill(self, row: dict) -> None:
        self._fills.append(dict(row))

    def record_equity(self, row: dict) -> None:
        self._equity.append(dict(row))

    def recent_equity_returns(self, limit: int = 100) -> tuple[float, ...]:
        values = [float(x["equity"]) for x in self._equity[-max(2, limit + 1):]]
        if len(values) < 2:
            return ()
        out = []
        for a, b in zip(values[:-1], values[1:]):
            if a > 0:
                out.append(b / a - 1.0)
        return tuple(out)

    def recent_observations(self, limit: int = 50) -> list[dict]:
        return list(reversed(self._observations[-limit:]))

    def recent_fills(self, limit: int = 50) -> list[dict]:
        return list(reversed(self._fills[-limit:]))

    def summary(self) -> dict:
        return {
            "backend": "memory",
            "account": asdict(self._account),
            "positions": [asdict(x) for x in self.list_positions()],
            "observations": len(self._observations),
            "fills": len(self._fills),
            "equity_points": len(self._equity),
        }


class PostgresPaperStore:
    """Small PostgreSQL persistence layer for forward paper evidence.

    Secrets remain in DATABASE_URL. The schema stores only research/paper data.
    No exchange credential is required or persisted.
    """

    def __init__(self, database_url: str, initial_cash: float = 10_000.0):
        if not database_url:
            raise ValueError("database_url is required")
        self.database_url = database_url
        self.initial_cash = float(initial_cash)
        try:
            import psycopg  # noqa: F401
        except Exception as exc:
            raise RuntimeError("psycopg is required for PostgresPaperStore") from exc

    def _connect(self):
        import psycopg
        return psycopg.connect(self.database_url)

    def ensure_schema(self) -> None:
        ddl = """
        CREATE TABLE IF NOT EXISTS paper_account (
            id SMALLINT PRIMARY KEY,
            cash DOUBLE PRECISION NOT NULL,
            equity DOUBLE PRECISION NOT NULL,
            peak_equity DOUBLE PRECISION NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL
        );
        CREATE TABLE IF NOT EXISTS paper_positions (
            symbol TEXT PRIMARY KEY,
            quantity DOUBLE PRECISION NOT NULL,
            avg_price DOUBLE PRECISION NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL
        );
        CREATE TABLE IF NOT EXISTS paper_observations (
            id BIGSERIAL PRIMARY KEY,
            observed_at TIMESTAMPTZ NOT NULL,
            bar_timestamp TIMESTAMPTZ NOT NULL,
            symbol TEXT NOT NULL,
            strategy_version TEXT NOT NULL,
            action TEXT NOT NULL,
            rule_score DOUBLE PRECISION NOT NULL,
            reference_price DOUBLE PRECISION,
            spread_bps DOUBLE PRECISION,
            atr_pct DOUBLE PRECISION,
            features JSONB NOT NULL,
            reasons JSONB NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE(bar_timestamp, symbol, strategy_version)
        );
        CREATE TABLE IF NOT EXISTS paper_fills (
            id BIGSERIAL PRIMARY KEY,
            timestamp TIMESTAMPTZ NOT NULL,
            client_order_id TEXT NOT NULL UNIQUE,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            requested_quantity DOUBLE PRECISION NOT NULL,
            filled_quantity DOUBLE PRECISION NOT NULL,
            fill_price DOUBLE PRECISION NOT NULL,
            fee_paid DOUBLE PRECISION NOT NULL,
            slippage_paid DOUBLE PRECISION NOT NULL,
            status TEXT NOT NULL,
            strategy_version TEXT NOT NULL,
            metadata JSONB NOT NULL DEFAULT '{}'::jsonb
        );
        CREATE TABLE IF NOT EXISTS paper_equity (
            id BIGSERIAL PRIMARY KEY,
            timestamp TIMESTAMPTZ NOT NULL,
            equity DOUBLE PRECISION NOT NULL,
            cash DOUBLE PRECISION NOT NULL,
            gross_exposure DOUBLE PRECISION NOT NULL,
            metadata JSONB NOT NULL DEFAULT '{}'::jsonb
        );
        """
        now = datetime.now(timezone.utc)
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(ddl)
                cur.execute(
                    """
                    INSERT INTO paper_account(id,cash,equity,peak_equity,updated_at)
                    VALUES (1,%s,%s,%s,%s)
                    ON CONFLICT (id) DO NOTHING
                    """,
                    (self.initial_cash, self.initial_cash, self.initial_cash, now),
                )
            conn.commit()

    @staticmethod
    def _dt(value: Any) -> datetime:
        if isinstance(value, datetime):
            return value
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))

    def record_observation(self, row: dict) -> bool:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO paper_observations(
                        observed_at,bar_timestamp,symbol,strategy_version,action,rule_score,
                        reference_price,spread_bps,atr_pct,features,reasons
                    )
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb)
                    ON CONFLICT (bar_timestamp,symbol,strategy_version) DO NOTHING
                    RETURNING id
                    """,
                    (
                        self._dt(row["observed_at"]),
                        self._dt(row["bar_timestamp"]),
                        row["symbol"],
                        row["strategy_version"],
                        row["action"],
                        float(row["rule_score"]),
                        row.get("reference_price"),
                        row.get("spread_bps"),
                        row.get("atr_pct"),
                        json.dumps(row.get("features", {})),
                        json.dumps(row.get("reasons", [])),
                    ),
                )
                inserted = cur.fetchone() is not None
            conn.commit()
        return inserted

    def get_account(self) -> PaperAccount:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT cash,equity,peak_equity FROM paper_account WHERE id=1")
                row = cur.fetchone()
        if row is None:
            self.ensure_schema()
            return PaperAccount(self.initial_cash, self.initial_cash, self.initial_cash)
        return PaperAccount(float(row[0]), float(row[1]), float(row[2]))

    def set_account(self, account: PaperAccount) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO paper_account(id,cash,equity,peak_equity,updated_at)
                    VALUES (1,%s,%s,%s,%s)
                    ON CONFLICT (id) DO UPDATE SET
                        cash=EXCLUDED.cash,equity=EXCLUDED.equity,
                        peak_equity=EXCLUDED.peak_equity,updated_at=EXCLUDED.updated_at
                    """,
                    (account.cash, account.equity, account.peak_equity, datetime.now(timezone.utc)),
                )
            conn.commit()

    def get_position(self, symbol: str) -> PaperPosition:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT quantity,avg_price FROM paper_positions WHERE symbol=%s", (symbol,))
                row = cur.fetchone()
        return PaperPosition(symbol, 0.0, 0.0) if row is None else PaperPosition(symbol, float(row[0]), float(row[1]))

    def set_position(self, position: PaperPosition) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                if abs(position.quantity) < 1e-15:
                    cur.execute("DELETE FROM paper_positions WHERE symbol=%s", (position.symbol,))
                else:
                    cur.execute(
                        """
                        INSERT INTO paper_positions(symbol,quantity,avg_price,updated_at)
                        VALUES (%s,%s,%s,%s)
                        ON CONFLICT (symbol) DO UPDATE SET
                            quantity=EXCLUDED.quantity,avg_price=EXCLUDED.avg_price,
                            updated_at=EXCLUDED.updated_at
                        """,
                        (position.symbol, position.quantity, position.avg_price, datetime.now(timezone.utc)),
                    )
            conn.commit()

    def list_positions(self) -> list[PaperPosition]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT symbol,quantity,avg_price FROM paper_positions ORDER BY symbol")
                rows = cur.fetchall()
        return [PaperPosition(str(r[0]), float(r[1]), float(r[2])) for r in rows]

    def record_fill(self, row: dict) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO paper_fills(
                        timestamp,client_order_id,symbol,side,requested_quantity,filled_quantity,
                        fill_price,fee_paid,slippage_paid,status,strategy_version,metadata
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
                    ON CONFLICT (client_order_id) DO NOTHING
                    """,
                    (
                        self._dt(row["timestamp"]), row["client_order_id"], row["symbol"], row["side"],
                        row["requested_quantity"], row["filled_quantity"], row["fill_price"],
                        row["fee_paid"], row["slippage_paid"], row["status"],
                        row["strategy_version"], json.dumps(row.get("metadata", {})),
                    ),
                )
            conn.commit()

    def record_equity(self, row: dict) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO paper_equity(timestamp,equity,cash,gross_exposure,metadata)
                    VALUES (%s,%s,%s,%s,%s::jsonb)
                    """,
                    (
                        self._dt(row["timestamp"]), row["equity"], row["cash"],
                        row["gross_exposure"], json.dumps(row.get("metadata", {})),
                    ),
                )
            conn.commit()

    def recent_equity_returns(self, limit: int = 100) -> tuple[float, ...]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT equity FROM paper_equity ORDER BY id DESC LIMIT %s",
                    (int(max(2, limit + 1)),),
                )
                values = [float(r[0]) for r in cur.fetchall()][::-1]
        out = []
        for a, b in zip(values[:-1], values[1:]):
            if a > 0:
                out.append(b / a - 1.0)
        return tuple(out)

    def _json_rows(self, query: str, limit: int) -> list[dict]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (int(limit),))
                names = [x.name for x in cur.description]
                rows = cur.fetchall()
        out = []
        for row in rows:
            d = dict(zip(names, row))
            for k, v in list(d.items()):
                if isinstance(v, datetime):
                    d[k] = v.isoformat()
            out.append(d)
        return out

    def recent_observations(self, limit: int = 50) -> list[dict]:
        return self._json_rows(
            """
            SELECT observed_at,bar_timestamp,symbol,strategy_version,action,rule_score,
                   reference_price,spread_bps,atr_pct,features,reasons
            FROM paper_observations ORDER BY id DESC LIMIT %s
            """,
            limit,
        )

    def recent_fills(self, limit: int = 50) -> list[dict]:
        return self._json_rows(
            """
            SELECT timestamp,client_order_id,symbol,side,requested_quantity,filled_quantity,
                   fill_price,fee_paid,slippage_paid,status,strategy_version,metadata
            FROM paper_fills ORDER BY id DESC LIMIT %s
            """,
            limit,
        )

    def summary(self) -> dict:
        account = self.get_account()
        positions = self.list_positions()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM paper_observations")
                obs = int(cur.fetchone()[0])
                cur.execute("SELECT COUNT(*) FROM paper_fills")
                fills = int(cur.fetchone()[0])
                cur.execute("SELECT COUNT(*) FROM paper_equity")
                eq = int(cur.fetchone()[0])
        return {
            "backend": "postgres",
            "account": asdict(account),
            "positions": [asdict(x) for x in positions],
            "observations": obs,
            "fills": fills,
            "equity_points": eq,
        }


def store_from_environment(initial_cash: float = 10_000.0):
    url = os.getenv("DATABASE_URL", "").strip()
    if url:
        store = PostgresPaperStore(url, initial_cash=initial_cash)
        store.ensure_schema()
        return store
    store = MemoryPaperStore(initial_cash=initial_cash)
    store.ensure_schema()
    return store
