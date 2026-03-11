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


class OrderBookImbalanceSignal(BaseSignal):
    """
    Bid/ask volume imbalance signal.

    If bids significantly outweigh asks, this suggests buying pressure
    and shifts the posterior toward YES. The log-likelihood ratio is
    proportional to log(bid_volume / ask_volume).
    """

    def __init__(self, sensitivity: float = 1.0, min_depth_usd: float = 100.0):
        self._sensitivity = sensitivity
        self._min_depth = min_depth_usd

    @property
    def name(self) -> str:
        return "orderbook_imbalance"

    @property
    def description(self) -> str:
        return "Bid/ask volume imbalance"

    async def compute(
        self,
        condition_id: str,
        market: Market,
        order_book: OrderBookState,
        lmsr_state: LMSRState,
        **context,
    ) -> Optional[SignalUpdate]:
        bids = order_book.bids_as_tuples()
        asks = order_book.asks_as_tuples()

        bid_vol = sum(p * s for p, s in bids)
        ask_vol = sum(p * s for p, s in asks)

        if bid_vol < self._min_depth or ask_vol < self._min_depth:
            return None

        # Imbalance ratio — clamped to avoid extreme values
        ratio = max(0.1, min(10.0, bid_vol / ask_vol))
        log_ratio = np.log(ratio) * self._sensitivity

        return SignalUpdate(
            signal_name=self.name,
            timestamp=time.time(),
            log_likelihood_yes=float(log_ratio),
            log_likelihood_no=float(-log_ratio),
            confidence=min(1.0, (bid_vol + ask_vol) / 5000.0),
            metadata={"bid_vol": bid_vol, "ask_vol": ask_vol, "ratio": ratio},
        )
