"""
Whale Tracker Signal — detects informed money flow from large trades.

Large trades (>$500) carry disproportionate information content.
This signal analyzes trade size distribution and sequential momentum
independently from the aggregate VolumeSignal.
"""

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


class WhaleTrackerSignal(BaseSignal):
    """
    Monitors large trades for informed-money directional signals.

    Independent from VolumeSignal: focuses on size distribution tail
    and sequential patterns rather than aggregate buy/sell imbalance.
    """

    def __init__(
        self,
        whale_threshold_usd: float = 500.0,
        volume_spike_multiplier: float = 3.0,
        lookback_trades: int = 100,
        min_whale_trades: int = 1,
    ):
        self._threshold = whale_threshold_usd
        self._spike_mult = volume_spike_multiplier
        self._lookback = lookback_trades
        self._min_whales = min_whale_trades

    @property
    def name(self) -> str:
        return "whale_tracker"

    @property
    def description(self) -> str:
        return "Large trade / whale activity tracker"

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

        # Compute trade USD values
        whale_buy_vol = 0.0
        whale_sell_vol = 0.0
        whale_count = 0
        largest_trade = 0.0

        sizes = []
        for t in recent:
            price = float(t.get("price", 0))
            size = float(t.get("size", 0))
            usd_val = price * size
            sizes.append(usd_val)

            if usd_val >= self._threshold:
                whale_count += 1
                side = t.get("side", "").upper()
                if side == "BUY":
                    whale_buy_vol += usd_val
                else:
                    whale_sell_vol += usd_val
                largest_trade = max(largest_trade, usd_val)

        if whale_count < self._min_whales:
            return None

        # Net whale direction
        net_whale = whale_buy_vol - whale_sell_vol
        total_whale = whale_buy_vol + whale_sell_vol
        if total_whale == 0:
            return None

        # Sequential momentum: count consecutive same-direction trades at tail
        consecutive = 1
        consecutive_dir = ""
        if recent:
            last_side = recent[-1].get("side", "").upper()
            consecutive_dir = last_side
            for t in reversed(recent[:-1]):
                if t.get("side", "").upper() == last_side:
                    consecutive += 1
                else:
                    break

        # Combine: whale direction is primary, sequential momentum is secondary
        whale_ratio = net_whale / total_whale  # -1 to +1
        momentum_bonus = 0.0
        if consecutive >= 5:
            momentum_bonus = 0.1 * min(consecutive / 10.0, 1.0)
            if consecutive_dir == "SELL":
                momentum_bonus = -momentum_bonus

        raw_signal = whale_ratio * 0.5 + momentum_bonus
        signal_strength = abs(raw_signal)

        if raw_signal > 0:
            log_ll_yes = float(signal_strength)
            log_ll_no = float(-signal_strength)
        else:
            log_ll_yes = float(-signal_strength)
            log_ll_no = float(signal_strength)

        # Confidence from whale count and recency
        confidence = min(1.0, whale_count / 5.0) * min(1.0, total_whale / 2000.0)

        return SignalUpdate(
            signal_name=self.name,
            timestamp=time.time(),
            log_likelihood_yes=log_ll_yes,
            log_likelihood_no=log_ll_no,
            confidence=max(0.01, confidence),
            metadata={
                "whale_count": whale_count,
                "whale_buy_vol": round(whale_buy_vol, 2),
                "whale_sell_vol": round(whale_sell_vol, 2),
                "largest_trade_usd": round(largest_trade, 2),
                "consecutive_dir": consecutive_dir,
                "consecutive_count": consecutive,
            },
        )
