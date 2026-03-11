from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.analysis.bayesian_engine import SignalUpdate

if TYPE_CHECKING:
    from src.analysis.lmsr_engine import LMSRState
    from src.feed.order_book import OrderBookState
    from src.market.models import Market
    from src.signals.base import BaseSignal

logger = logging.getLogger(__name__)


class SignalRegistry:
    """Registry of pluggable signal sources."""

    def __init__(self) -> None:
        self._signals: dict[str, BaseSignal] = {}

    def register(self, signal: BaseSignal) -> None:
        self._signals[signal.name] = signal
        logger.info("Registered signal: %s", signal.name)

    def unregister(self, name: str) -> None:
        self._signals.pop(name, None)

    async def compute_all(
        self,
        condition_id: str,
        market: Market,
        order_book: OrderBookState,
        lmsr_state: LMSRState,
        **context,
    ) -> list[SignalUpdate]:
        updates: list[SignalUpdate] = []
        for signal in self._signals.values():
            if not signal.is_applicable(market):
                continue
            try:
                result = await signal.compute(
                    condition_id, market, order_book, lmsr_state, **context
                )
                if result is not None:
                    updates.append(result)
            except Exception:
                logger.exception("Signal %s failed for %s", signal.name, condition_id)
        return updates
