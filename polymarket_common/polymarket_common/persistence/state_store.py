"""SQLite-backed state persistence."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import aiosqlite

from polymarket_common.execution.position_tracker import Position

logger = logging.getLogger(__name__)

DB_PATH = Path("data/bot.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY,
    condition_id TEXT NOT NULL,
    token_id TEXT NOT NULL,
    side TEXT NOT NULL,
    price REAL NOT NULL,
    size REAL NOT NULL,
    filled_size REAL DEFAULT 0,
    status TEXT NOT NULL,
    edge REAL,
    kelly_fraction REAL,
    p_hat REAL,
    b_estimate REAL,
    confidence REAL,
    market_question TEXT,
    placed_at REAL NOT NULL,
    updated_at REAL
);

CREATE TABLE IF NOT EXISTS daily_pnl (
    date TEXT PRIMARY KEY,
    realized_pnl REAL DEFAULT 0,
    trades INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_orders_condition ON orders(condition_id);
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);

CREATE TABLE IF NOT EXISTS positions (
    token_id TEXT PRIMARY KEY,
    condition_id TEXT NOT NULL,
    side TEXT NOT NULL,
    size REAL NOT NULL,
    avg_price REAL NOT NULL,
    cost_basis REAL NOT NULL,
    entry_time REAL NOT NULL,
    high_water_mark REAL NOT NULL,
    realized_pnl REAL DEFAULT 0,
    partial_profit_taken INTEGER DEFAULT 0,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_positions_condition ON positions(condition_id);
"""


class StateStore:
    def __init__(self, db_path: Path | None = None):
        self._path = db_path or DB_PATH
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self._path))

        # Enable WAL mode for concurrent reader/writer access across services
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA busy_timeout=5000")

        await self._db.executescript(SCHEMA)
        await self._db.commit()

        # Migrate existing DBs: add columns that may not exist yet.
        # spot_open/spot_close are the asset-agnostic successors to
        # btc_open/btc_close (the old names were misleading and hid a bug
        # where ETH trades were accidentally resolved against BTC prices).
        for col, col_type in [
            ("confidence", "REAL"),
            ("market_question", "TEXT"),
            ("btc_open", "REAL"),
            ("btc_close", "REAL"),
            ("outcome", "TEXT"),
            ("pnl_usd", "REAL"),
            ("exit_price", "REAL"),
            ("estimated_price", "REAL"),
            ("spot_open", "REAL"),
            ("spot_close", "REAL"),
        ]:
            try:
                await self._db.execute(
                    f"ALTER TABLE orders ADD COLUMN {col} {col_type}"
                )
                await self._db.commit()
            except Exception:
                pass  # Column already exists

        # One-time backfill: copy legacy btc_open/btc_close into spot_open/spot_close
        # for any rows that haven't been copied yet. Safe to re-run.
        try:
            await self._db.execute(
                """UPDATE orders
                   SET spot_open = btc_open
                   WHERE spot_open IS NULL AND btc_open IS NOT NULL"""
            )
            await self._db.execute(
                """UPDATE orders
                   SET spot_close = btc_close
                   WHERE spot_close IS NULL AND btc_close IS NOT NULL"""
            )
            await self._db.commit()
        except Exception:
            logger.debug("spot_* backfill skipped", exc_info=True)

        # Migrate positions table
        try:
            await self._db.execute(
                "ALTER TABLE positions ADD COLUMN partial_profit_taken INTEGER DEFAULT 0"
            )
            await self._db.commit()
        except Exception:
            pass  # Column already exists

        logger.info("State store initialized at %s", self._path)

    async def log_trade(
        self,
        order_id: str,
        condition_id: str,
        token_id: str,
        side: str,
        price: float,
        size: float,
        status: str,
        edge: float,
        kelly_fraction: float,
        p_hat: float,
        b_estimate: float,
        placed_at: float,
        confidence: float = 0.0,
        market_question: str = "",
        btc_open: float | None = None,
        btc_close: float | None = None,
        outcome: str | None = None,
        pnl_usd: float | None = None,
        exit_price: float | None = None,
        estimated_price: float | None = None,
        spot_open: float | None = None,
        spot_close: float | None = None,
    ) -> None:
        if not self._db:
            return
        # Keep the old btc_* columns populated for backwards compatibility
        # when callers pass the new spot_* names. (New runners only pass spot_*.)
        if btc_open is None and spot_open is not None:
            btc_open = spot_open
        if btc_close is None and spot_close is not None:
            btc_close = spot_close
        if spot_open is None and btc_open is not None:
            spot_open = btc_open
        if spot_close is None and btc_close is not None:
            spot_close = btc_close

        await self._db.execute(
            """INSERT OR REPLACE INTO orders
               (order_id, condition_id, token_id, side, price, size,
                status, edge, kelly_fraction, p_hat, b_estimate,
                confidence, market_question, placed_at,
                btc_open, btc_close, outcome, pnl_usd, exit_price,
                estimated_price, spot_open, spot_close)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                order_id,
                condition_id,
                token_id,
                side,
                price,
                size,
                status,
                edge,
                kelly_fraction,
                p_hat,
                b_estimate,
                confidence,
                market_question,
                placed_at,
                btc_open,
                btc_close,
                outcome,
                pnl_usd,
                exit_price,
                estimated_price,
                spot_open,
                spot_close,
            ),
        )
        await self._db.commit()

    async def update_daily_pnl(self, date: str, pnl: float, trades: int) -> None:
        if not self._db:
            return
        await self._db.execute(
            """INSERT OR REPLACE INTO daily_pnl (date, realized_pnl, trades)
               VALUES (?, ?, ?)""",
            (date, pnl, trades),
        )
        await self._db.commit()

    async def get_recent_orders(self, limit: int = 50) -> list[dict]:
        if not self._db:
            return []
        cursor = await self._db.execute(
            "SELECT * FROM orders ORDER BY placed_at DESC LIMIT ?", (limit,)
        )
        rows = await cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        return [dict(zip(cols, row)) for row in rows]

    async def save_position(self, pos: Position) -> None:
        if not self._db:
            return
        await self._db.execute(
            """INSERT OR REPLACE INTO positions
               (token_id, condition_id, side, size, avg_price, cost_basis,
                entry_time, high_water_mark, realized_pnl, partial_profit_taken, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                pos.token_id,
                pos.condition_id,
                pos.side,
                pos.size,
                pos.avg_price,
                pos.cost_basis,
                pos.entry_time,
                pos.high_water_mark,
                pos.realized_pnl,
                1 if pos.partial_profit_taken else 0,
                time.time(),
            ),
        )
        await self._db.commit()

    async def delete_position(self, token_id: str) -> None:
        if not self._db:
            return
        await self._db.execute("DELETE FROM positions WHERE token_id = ?", (token_id,))
        await self._db.commit()

    async def load_all_positions(self) -> list[Position]:
        if not self._db:
            return []
        cursor = await self._db.execute(
            "SELECT token_id, condition_id, side, size, avg_price, cost_basis, "
            "entry_time, high_water_mark, realized_pnl, partial_profit_taken "
            "FROM positions WHERE size > 0"
        )
        rows = await cursor.fetchall()
        positions = []
        for row in rows:
            positions.append(
                Position(
                    token_id=row[0],
                    condition_id=row[1],
                    side=row[2],
                    size=row[3],
                    avg_price=row[4],
                    cost_basis=row[5],
                    entry_time=row[6],
                    high_water_mark=row[7],
                    realized_pnl=row[8],
                    partial_profit_taken=bool(row[9]) if len(row) > 9 else False,
                )
            )
        return positions

    async def close(self) -> None:
        if self._db:
            await self._db.close()
