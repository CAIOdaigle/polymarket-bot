"""Position state tracking."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Position:
    condition_id: str
    token_id: str
    side: str  # YES or NO
    size: float = 0.0  # shares
    avg_price: float = 0.0
    cost_basis: float = 0.0  # total USD spent

    @property
    def market_value(self) -> float:
        return self.size * self.avg_price


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
            pos = Position(condition_id=condition_id, token_id=token_id, side=side)
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

    @property
    def positions(self) -> dict[str, Position]:
        return self._positions
