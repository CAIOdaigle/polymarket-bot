from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

from src.analysis.bayesian_engine import SignalUpdate

if TYPE_CHECKING:
    from src.analysis.lmsr_engine import LMSRState
    from src.feed.order_book import OrderBookState
    from src.market.models import Market


class BaseSignal(ABC):
    """
    Abstract base class for all signal sources.

    To add a new signal:
      1. Subclass BaseSignal
      2. Implement name, description, and compute()
      3. Register it in the SignalRegistry
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @abstractmethod
    async def compute(
        self,
        condition_id: str,
        market: Market,
        order_book: OrderBookState,
        lmsr_state: LMSRState,
        **context,
    ) -> Optional[SignalUpdate]:
        """
        Compute signal for a market.
        Return SignalUpdate with log-likelihoods, or None if no opinion.
        """
        ...

    def is_applicable(self, market: Market) -> bool:
        """Override to restrict signal to certain market types."""
        return True
