from __future__ import annotations

import time
from typing import TYPE_CHECKING, Optional

import numpy as np

from src.analysis.bayesian_engine import SignalUpdate
from src.signals.base import BaseSignal

if TYPE_CHECKING:
    from src.analysis.lmsr_engine import LMSRState
    from src.feed.order_book import OrderBookState
    from src.market.models import Market


class LMSRDeviationSignal(BaseSignal):
    """
    CLOB vs LMSR fair-value deviation signal (Inefficiency Signal).

    Compares the CLOB mid-price to the LMSR-implied fair price.
    When the CLOB price is below the LMSR fair value for YES,
    it suggests YES is underpriced — a buy signal.
    """

    def __init__(self, min_deviation: float = 0.01):
        self._min_deviation = min_deviation

    @property
    def name(self) -> str:
        return "lmsr_deviation"

    @property
    def description(self) -> str:
        return "CLOB price vs LMSR implied fair value"

    async def compute(
        self,
        condition_id: str,
        market: Market,
        order_book: OrderBookState,
        lmsr_state: LMSRState,
        **context,
    ) -> Optional[SignalUpdate]:
        mid = order_book.mid_price
        if mid is None:
            return None

        deviation = lmsr_state.implied_price_yes - mid

        if abs(deviation) < self._min_deviation:
            return None

        # Scale signal: larger deviation = stronger signal
        # Positive deviation means CLOB underprices YES relative to LMSR
        signal_strength = np.log(1.0 + abs(deviation) * 10.0)
        if deviation > 0:
            ll_yes = float(signal_strength)
            ll_no = float(-signal_strength)
        else:
            ll_yes = float(-signal_strength)
            ll_no = float(signal_strength)

        return SignalUpdate(
            signal_name=self.name,
            timestamp=time.time(),
            log_likelihood_yes=ll_yes,
            log_likelihood_no=ll_no,
            confidence=lmsr_state.confidence,
            metadata={
                "clob_mid": mid,
                "lmsr_fair": lmsr_state.implied_price_yes,
                "deviation": deviation,
                "b": lmsr_state.b,
            },
        )
