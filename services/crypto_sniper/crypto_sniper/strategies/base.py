"""Base strategy interface for all crypto sniper strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from crypto_sniper.feeds.feed_manager import FeedManager


@dataclass
class TradeSignal:
    """Output of a strategy evaluation — a recommendation to trade."""
    direction: str  # "UP" or "DOWN"
    confidence: float  # 0.0-1.0
    score: float  # raw composite score
    strategy_name: str
    window_ts: int  # 5-min window start timestamp
    ev_edge: Optional[float] = None  # Black-Scholes EV edge (if applicable)
    token_price: float = 0.50  # estimated token cost
    components: dict = field(default_factory=dict)  # indicator breakdown


class BaseStrategy(ABC):
    """Abstract base class for all crypto sniper strategies."""

    @abstractmethod
    async def initialize(self, feed_manager: FeedManager) -> None:
        """Initialize the strategy with access to shared feeds."""
        ...

    @abstractmethod
    async def evaluate(
        self, window_ts: int, seconds_remaining: float
    ) -> Optional[TradeSignal]:
        """Evaluate whether to trade this window.

        Called repeatedly during the entry window at the strategy's eval_interval.
        Return a TradeSignal to fire, or None to skip.
        """
        ...

    @abstractmethod
    async def shutdown(self) -> None:
        """Clean up resources."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy name."""
        ...

    @property
    @abstractmethod
    def eval_interval_seconds(self) -> float:
        """How often evaluate() should be called during the entry window."""
        ...

    @property
    @abstractmethod
    def entry_window_seconds(self) -> tuple[int, int]:
        """(earliest_seconds_before_close, latest_seconds_before_close).

        e.g. (55, 8) means fire between T-55s and T-8s.
        """
        ...
