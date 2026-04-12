"""Real-time Binance BTC price feed — WebSocket + REST.

Merged from existing src/btc_sniper/binance_feed.py (REST polling, Candle dataclass)
and downloaded signals/binance_feed.py (WebSocket trade stream).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)


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
    """Async Binance data provider — WebSocket for real-time price, REST for candles."""

    ws_url: str = "wss://stream.binance.com:9443/ws/btcusdt@trade"
    rest_url: str = "https://api.binance.com/api/v3"
    symbol: str = "BTCUSDT"

    _session: Optional[aiohttp.ClientSession] = field(default=None, repr=False, init=False)
    _tick_prices: list[tuple[float, float]] = field(default_factory=list, init=False)
    _latest_price: float = field(default=0.0, init=False)
    _running: bool = field(default=False, init=False)
    _ws_task: Optional[asyncio.Task] = field(default=None, init=False)

    async def start(self) -> None:
        self._session = aiohttp.ClientSession()
        self._running = True

        # Get initial price via REST
        price = await self._fetch_price_rest()
        if price:
            self._latest_price = price
            self._tick_prices.append((time.time(), price))
            logger.info("Binance initial BTC price: $%.2f", price)

        # Start WebSocket listener
        self._ws_task = asyncio.create_task(
            self._ws_listener(), name="binance_ws"
        )

    async def stop(self) -> None:
        self._running = False
        if self._ws_task:
            self._ws_task.cancel()
        if self._session:
            await self._session.close()

    async def _ws_listener(self) -> None:
        """WebSocket trade stream for real-time price updates."""
        while self._running:
            try:
                async with self._session.ws_connect(
                    self.ws_url,
                    heartbeat=10,
                    timeout=aiohttp.ClientTimeout(total=None),
                ) as ws:
                    logger.info("Binance WebSocket connected")
                    async for msg in ws:
                        if not self._running:
                            break
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            data = json.loads(msg.data)
                            price = float(data.get("p", 0))
                            if price > 0:
                                self._latest_price = price
                                now = time.time()
                                self._tick_prices.append((now, price))
                                # Keep last 5 minutes
                                cutoff = now - 300
                                self._tick_prices = [
                                    (t, p) for t, p in self._tick_prices
                                    if t >= cutoff
                                ]
                        elif msg.type in (
                            aiohttp.WSMsgType.CLOSED,
                            aiohttp.WSMsgType.ERROR,
                        ):
                            break
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.warning("Binance WS error: %s — reconnecting in 3s", e)
                # Fallback to REST while reconnecting
                try:
                    price = await self._fetch_price_rest()
                    if price:
                        self._latest_price = price
                except Exception:
                    pass
                await asyncio.sleep(3)

    async def _fetch_price_rest(self) -> Optional[float]:
        """REST fallback for current price."""
        if not self._session:
            return None
        try:
            url = f"{self.rest_url}/ticker/price"
            async with self._session.get(
                url,
                params={"symbol": self.symbol},
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return float(data["price"])
        except Exception:
            logger.debug("REST price fetch failed", exc_info=True)
        return None

    async def get_candles(self, interval: str = "1m", limit: int = 30) -> list[Candle]:
        """Fetch recent candles from Binance REST API."""
        if not self._session:
            return []
        try:
            url = f"{self.rest_url}/klines"
            async with self._session.get(
                url,
                params={"symbol": self.symbol, "interval": interval, "limit": limit},
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
        """Return most recent price."""
        return self._latest_price if self._latest_price > 0 else None
