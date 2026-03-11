"""SQLite-backed state persistence."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import aiosqlite

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
"""


class StateStore:
    def __init__(self, db_path: Path | None = None):
        self._path = db_path or DB_PATH
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self._path))
        await self._db.executescript(SCHEMA)
        await self._db.commit()
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
    ) -> None:
        if not self._db:
            return
        await self._db.execute(
            """INSERT OR REPLACE INTO orders
               (order_id, condition_id, token_id, side, price, size,
                status, edge, kelly_fraction, p_hat, b_estimate, placed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                placed_at,
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

    async def close(self) -> None:
        if self._db:
            await self._db.close()
