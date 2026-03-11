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


class VolumeSignal(BaseSignal):
    """
    Volume/momentum signal.

    Analyzes recent trade history. A surge in buy volume suggests
    informed trading toward YES; sell volume suggests NO.
    """

    def __init__(self, lookback_trades: int = 50):
        self._lookback = lookback_trades

    @property
    def name(self) -> str:
        return "volume_momentum"

    @property
    def description(self) -> str:
        return "Recent trade volume momentum"

    async def compute(
        self,
        condition_id: str,
        market: Market,
        order_book: OrderBookState,
        lmsr_state: LMSRState,
        **context,
    ) -> Optional[SignalUpdate]:
        trades = order_book.recent_trades
        if len(trades) < 5:
            return None

        recent = trades[-self._lookback :]
        buy_vol = sum(t["size"] for t in recent if t.get("side") == "BUY")
        sell_vol = sum(t["size"] for t in recent if t.get("side") == "SELL")

        total = buy_vol + sell_vol
        if total < 1.0:
            return None

        # Volume imbalance
        imbalance = (buy_vol - sell_vol) / total  # -1 to +1
        signal_strength = abs(imbalance) * 0.5  # Conservative scaling

        if imbalance > 0:
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
            confidence=min(1.0, total / 500.0),
            metadata={"buy_vol": buy_vol, "sell_vol": sell_vol, "imbalance": imbalance},
        )
