"""Portfolio-level risk constraints."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from src.config import TradingConfig

logger = logging.getLogger(__name__)


@dataclass
class DailyPnL:
    date: str
    realized_pnl: float = 0.0
    trades: int = 0


class RiskManager:
    def __init__(self, config: TradingConfig):
        self._daily_loss_limit = config.daily_loss_limit_usd
        self._max_exposure = config.max_portfolio_exposure
        self._bankroll = config.total_bankroll_usd
        self._daily_pnl: float = 0.0
        self._daily_trades: int = 0
        self._day_start: float = time.time()
        self._positions_by_event: dict[str, float] = {}  # event_slug -> total usd

    def update_limits(self, bankroll: float, daily_loss_limit: float | None = None) -> None:
        """Sync risk limits with the latest bankroll (called after balance refresh)."""
        self._bankroll = bankroll
        if daily_loss_limit is not None:
            self._daily_loss_limit = daily_loss_limit
        logger.debug("Risk limits updated: bankroll=$%.2f loss_limit=$%.2f",
                      self._bankroll, self._daily_loss_limit)

    def check_can_trade(
        self, total_deployed_usd: float, unrealized_pnl: float = 0.0
    ) -> tuple[bool, str]:
        """Check portfolio-level constraints before a trade."""
        # Daily loss limit (realized + unrealized)
        total_pnl = self._daily_pnl + unrealized_pnl
        if total_pnl < -self._daily_loss_limit:
            return False, f"Daily loss limit hit: ${total_pnl:.2f} (realized=${self._daily_pnl:.2f} unrealized=${unrealized_pnl:.2f})"

        # Portfolio exposure limit
        if total_deployed_usd >= self._bankroll * self._max_exposure:
            return False, f"Portfolio exposure limit: ${total_deployed_usd:.2f}"

        return True, "OK"

    def record_pnl(self, amount: float) -> None:
        self._daily_pnl += amount
        self._daily_trades += 1

    def update_event_exposure(self, event_slug: str, position_usd: float) -> None:
        self._positions_by_event[event_slug] = position_usd

    def reset_daily(self) -> DailyPnL:
        from datetime import datetime, timezone

        result = DailyPnL(
            date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            realized_pnl=self._daily_pnl,
            trades=self._daily_trades,
        )
        self._daily_pnl = 0.0
        self._daily_trades = 0
        self._day_start = time.time()
        return result
