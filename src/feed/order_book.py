"""In-memory order book state per market."""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

MAX_RECENT_TRADES = 200
STALE_TRADE_SECONDS = 300  # 5 minutes


@dataclass
class OrderBookState:
    token_id: str
    _bids: dict[float, float] = field(default_factory=dict)  # price -> size
    _asks: dict[float, float] = field(default_factory=dict)
    recent_trades: list[dict] = field(default_factory=list)
    last_trade_price: Optional[float] = None
    last_trade_time: Optional[float] = None

    def update_from_snapshot(self, bids: list[dict], asks: list[dict]) -> None:
        self._bids.clear()
        self._asks.clear()
        for b in bids:
            price = float(b.get("price", 0))
            size = float(b.get("size", 0))
            if size > 0:
                self._bids[price] = size
        for a in asks:
            price = float(a.get("price", 0))
            size = float(a.get("size", 0))
            if size > 0:
                self._asks[price] = size

    def update_price_level(self, side: str, price: float, size: float) -> None:
        book = self._bids if side.upper() == "BID" else self._asks
        if size <= 0:
            book.pop(price, None)
        else:
            book[price] = size

    def record_trade(self, trade: dict) -> None:
        self.recent_trades.append(trade)
        if len(self.recent_trades) > MAX_RECENT_TRADES:
            self.recent_trades = self.recent_trades[-MAX_RECENT_TRADES:]
        self.last_trade_price = float(trade.get("price", 0))
        self.last_trade_time = time.time()

    def bids_as_tuples(self) -> list[tuple[float, float]]:
        """Returns [(price, size), ...] sorted by price descending."""
        return sorted(self._bids.items(), key=lambda x: x[0], reverse=True)

    def asks_as_tuples(self) -> list[tuple[float, float]]:
        """Returns [(price, size), ...] sorted by price ascending."""
        return sorted(self._asks.items(), key=lambda x: x[0])

    @property
    def best_bid(self) -> Optional[float]:
        return max(self._bids.keys()) if self._bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return min(self._asks.keys()) if self._asks else None

    @property
    def mid_price(self) -> Optional[float]:
        bb = self.best_bid
        ba = self.best_ask
        if bb is not None and ba is not None:
            return (bb + ba) / 2
        # Only use last trade price if it's recent (< 5 min)
        if (
            self.last_trade_price is not None
            and self.last_trade_time is not None
            and (time.time() - self.last_trade_time) < STALE_TRADE_SECONDS
        ):
            return self.last_trade_price
        return None

    @property
    def spread(self) -> Optional[float]:
        bb = self.best_bid
        ba = self.best_ask
        if bb is not None and ba is not None:
            return ba - bb
        return None

    @property
    def has_data(self) -> bool:
        return bool(self._bids or self._asks)
