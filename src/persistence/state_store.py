"""SQLite-backed state persistence."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import aiosqlite

from src.execution.position_tracker import Position

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
        await self._db.executescript(SCHEMA)
        await self._db.commit()

        # Migrate existing DBs: add columns that may not exist yet
        for col, col_type in [
            ("confidence", "REAL"),
            ("market_question", "TEXT"),
        ]:
            try:
                await self._db.execute(
                    f"ALTER TABLE orders ADD COLUMN {col} {col_type}"
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
    ) -> None:
        if not self._db:
            return
        await self._db.execute(
            """INSERT OR REPLACE INTO orders
               (order_id, condition_id, token_id, side, price, size,
                status, edge, kelly_fraction, p_hat, b_estimate,
                confidence, market_question, placed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                entry_time, high_water_mark, realized_pnl, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
            "entry_time, high_water_mark, realized_pnl FROM positions WHERE size > 0"
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
                )
            )
        return positions

    async def close(self) -> None:
        if self._db:
            await self._db.close()
