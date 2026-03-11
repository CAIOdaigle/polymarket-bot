"""
Cross-Market Signal — detects pricing inconsistencies between related markets.

Markets within the same event (same event_slug) should have coherent prices:
  1. Mutually exclusive outcomes should sum to ~1.0
  2. Shorter-deadline markets should not price higher than longer-deadline ones
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import TYPE_CHECKING, Optional

import numpy as np

from src.analysis.bayesian_engine import SignalUpdate
from src.signals.base import BaseSignal

if TYPE_CHECKING:
    from src.analysis.lmsr_engine import LMSRState
    from src.feed.order_book import OrderBookState
    from src.market.models import Market

logger = logging.getLogger(__name__)


class CrossMarketSignal(BaseSignal):
    """
    Finds pricing inconsistencies between markets in the same event.

    If mutually exclusive outcomes sum to more than 1.0, individual markets
    are overpriced. The signal pushes the most overpriced ones toward NO.
    """

    def __init__(
        self, min_inconsistency: float = 0.05, max_event_markets: int = 20
    ):
        self._min_inconsistency = min_inconsistency
        self._max_event_markets = max_event_markets

    @property
    def name(self) -> str:
        return "cross_market"

    @property
    def description(self) -> str:
        return "Cross-market pricing inconsistency"

    def is_applicable(self, market: Market) -> bool:
        return bool(market.event_slug)

    async def compute(
        self,
        condition_id: str,
        market: Market,
        order_book: OrderBookState,
        lmsr_state: LMSRState,
        **context,
    ) -> Optional[SignalUpdate]:
        scanner = context.get("scanner")
        if scanner is None:
            return None

        # Find sibling markets in the same event
        siblings = [
            m
            for m in scanner.markets.values()
            if m.event_slug == market.event_slug and m.condition_id != condition_id
        ]

        if not siblings:
            return None

        if len(siblings) + 1 > self._max_event_markets:
            return None

        # Get YES prices for all event markets (including current)
        current_price = self._get_yes_price(market)
        if current_price is None:
            return None

        sibling_prices = []
        for sib in siblings:
            p = self._get_yes_price(sib)
            if p is not None:
                sibling_prices.append(p)

        if not sibling_prices:
            return None

        # Sum violation: mutually exclusive outcomes should sum to ~1.0
        price_sum = current_price + sum(sibling_prices)
        deviation = price_sum - 1.0

        if abs(deviation) < self._min_inconsistency:
            return None

        # If sum > 1.0: all markets are overpriced, push toward NO
        # If sum < 1.0: all markets are underpriced, push toward YES
        # Scale by how much this market contributes to the overpricing
        market_share = current_price / price_sum if price_sum > 0 else 0
        signal_strength = np.log(1.0 + abs(deviation)) * market_share

        if deviation > 0:
            # Overpriced — push NO
            log_ll_yes = float(-signal_strength)
            log_ll_no = float(signal_strength)
        else:
            # Underpriced — push YES
            log_ll_yes = float(signal_strength)
            log_ll_no = float(-signal_strength)

        # Confidence scales with number of siblings and total liquidity
        total_liq = market.liquidity + sum(s.liquidity for s in siblings)
        confidence = min(1.0, len(siblings) / 5.0) * min(1.0, total_liq / 5000.0)

        return SignalUpdate(
            signal_name=self.name,
            timestamp=time.time(),
            log_likelihood_yes=log_ll_yes,
            log_likelihood_no=log_ll_no,
            confidence=max(0.01, confidence),
            metadata={
                "sibling_count": len(siblings),
                "price_sum": round(price_sum, 4),
                "deviation": round(deviation, 4),
                "inconsistency_type": "sum_violation",
            },
        )

    @staticmethod
    def _get_yes_price(market: Market) -> Optional[float]:
        for t in market.tokens:
            if t.outcome.lower() == "yes" and t.price is not None:
                return t.price
        return None
