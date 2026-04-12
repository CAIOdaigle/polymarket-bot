"""Real-time Binance BTC price feed for 5-minute sniper strategy.

Provides:
- REST candle fetcher (1-min candles for TA)
- 2-second price polling for tick-level micro-trend detection
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
BINANCE_PRICE_URL = "https://api.binance.com/api/v3/ticker/price"
SYMBOL = "BTCUSDT"


@dataclass
class Candle:
    open_time: int  # ms
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: int  # ms


@dataclass
class BinanceFeed:
    """Async Binance data provider for BTC price + candles."""

    _session: Optional[aiohttp.ClientSession] = field(default=None, repr=False)
    _tick_prices: list[tuple[float, float]] = field(default_factory=list)  # (timestamp, price)
    _running: bool = False
    _poll_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        self._session = aiohttp.ClientSession()
        self._running = True
        self._poll_task = asyncio.create_task(self._price_poll_loop(), name="binance_poll")

    async def stop(self) -> None:
        self._running = False
        if self._poll_task:
            self._poll_task.cancel()
        if self._session:
            await self._session.close()

    async def _price_poll_loop(self) -> None:
        """Poll BTC price every 2 seconds for tick-level trend detection."""
        while self._running:
            try:
                price = await self.get_current_price()
                if price is not None:
                    self._tick_prices.append((time.time(), price))
                    # Keep last 5 minutes of ticks
                    cutoff = time.time() - 300
                    self._tick_prices = [
                        (t, p) for t, p in self._tick_prices if t >= cutoff
                    ]
            except asyncio.CancelledError:
                return
            except Exception:
                logger.debug("Binance price poll failed", exc_info=True)
            await asyncio.sleep(2)

    async def get_current_price(self) -> Optional[float]:
        if not self._session:
            return None
        try:
            async with self._session.get(
                BINANCE_PRICE_URL,
                params={"symbol": SYMBOL},
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return float(data["price"])
        except Exception:
            logger.debug("Failed to fetch BTC price", exc_info=True)
        return None

    async def get_candles(self, interval: str = "1m", limit: int = 30) -> list[Candle]:
        """Fetch recent 1-minute candles from Binance REST API."""
        if not self._session:
            return []
        try:
            async with self._session.get(
                BINANCE_KLINES_URL,
                params={"symbol": SYMBOL, "interval": interval, "limit": limit},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status != 200:
                    return []
                raw = await resp.json()
                return [
                    Candle(
                        open_time=int(c[0]),
                        open=float(c[1]),
                        high=float(c[2]),
                        low=float(c[3]),
                        close=float(c[4]),
                        volume=float(c[5]),
                        close_time=int(c[6]),
                    )
                    for c in raw
                ]
        except Exception:
            logger.debug("Failed to fetch candles", exc_info=True)
            return []

    def get_tick_prices(self, since: float = 0) -> list[tuple[float, float]]:
        """Return (timestamp, price) ticks since given timestamp."""
        return [(t, p) for t, p in self._tick_prices if t >= since]

    def get_latest_price(self) -> Optional[float]:
        """Return most recent polled price."""
        return self._tick_prices[-1][1] if self._tick_prices else None
