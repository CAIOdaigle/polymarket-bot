"""Position state tracking."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Position:
    condition_id: str
    token_id: str
    side: str  # YES or NO
    size: float = 0.0  # shares
    avg_price: float = 0.0
    cost_basis: float = 0.0  # total USD spent
    entry_time: float = 0.0  # timestamp of first fill
    high_water_mark: float = 0.0  # highest favorable price observed
    realized_pnl: float = 0.0  # accumulated realized PnL from partial exits

    @property
    def market_value(self) -> float:
        return self.size * self.avg_price

    def update_high_water_mark(self, current_price: float) -> None:
        """Update HWM based on current price for this token."""
        if current_price > self.high_water_mark:
            self.high_water_mark = current_price


class PositionTracker:
    def __init__(self) -> None:
        self._positions: dict[str, Position] = {}  # token_id -> Position

    def update_from_fill(
        self,
        condition_id: str,
        token_id: str,
        side: str,
        fill_size: float,
        fill_price: float,
    ) -> Position:
        pos = self._positions.get(token_id)
        if pos is None:
            pos = Position(
                condition_id=condition_id,
                token_id=token_id,
                side=side,
                entry_time=time.time(),
                high_water_mark=fill_price,
            )
            self._positions[token_id] = pos

        new_cost = fill_size * fill_price
        total_size = pos.size + fill_size
        if total_size > 0:
            pos.avg_price = (pos.cost_basis + new_cost) / total_size
        pos.size = total_size
        pos.cost_basis += new_cost

        logger.info(
            "Position updated: %s %s size=%.2f avg=%.4f",
            condition_id[:8],
            side,
            pos.size,
            pos.avg_price,
        )
        return pos

    def reduce_position(
        self,
        token_id: str,
        size_sold: float,
        exit_price: float,
    ) -> float:
        """Reduce position by size_sold shares at exit_price.

        Returns realized PnL for this exit.
        avg_price stays constant (FIFO-like), cost_basis reduces proportionally.
        """
        pos = self._positions.get(token_id)
        if pos is None:
            logger.warning("reduce_position called for unknown token %s", token_id)
            return 0.0

        size_sold = min(size_sold, pos.size)
        if size_sold <= 0:
            return 0.0

        # Realized PnL = (exit_price - avg_price) * shares_sold
        realized = (exit_price - pos.avg_price) * size_sold
        pos.realized_pnl += realized

        # Reduce cost_basis proportionally
        fraction_sold = size_sold / pos.size if pos.size > 0 else 1.0
        pos.cost_basis -= pos.cost_basis * fraction_sold
        pos.size -= size_sold

        logger.info(
            "Position reduced: %s sold=%.2f @ %.4f pnl=%.4f remaining=%.2f",
            token_id[:8],
            size_sold,
            exit_price,
            realized,
            pos.size,
        )

        # Remove fully closed positions
        if pos.size <= 0.001:
            del self._positions[token_id]

        return realized

    def get_all_open(self) -> list[Position]:
        """Return all positions with size > 0."""
        return [p for p in self._positions.values() if p.size > 0]

    def get_position_usd(self, condition_id: str) -> float:
        total = 0.0
        for pos in self._positions.values():
            if pos.condition_id == condition_id:
                total += pos.cost_basis
        return total

    def get_total_deployed(self) -> float:
        return sum(pos.cost_basis for pos in self._positions.values())

    def get_position(self, token_id: str) -> Position | None:
        return self._positions.get(token_id)

    def restore_position(self, pos: Position) -> None:
        """Restore a position from persistence (crash recovery)."""
        self._positions[pos.token_id] = pos
        logger.info(
            "Restored position: %s %s size=%.2f avg=%.4f",
            pos.condition_id[:8],
            pos.side,
            pos.size,
            pos.avg_price,
        )

    @property
    def positions(self) -> dict[str, Position]:
        return self._positions
