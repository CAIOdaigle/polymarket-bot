"""Feed manager — multiplexes shared exchange feeds for strategies."""

from __future__ import annotations

import logging
from typing import Optional

from crypto_sniper.config import SniperConfig
from crypto_sniper.feeds.binance_feed import BinanceFeed, Candle

logger = logging.getLogger(__name__)


class FeedManager:
    """Central hub for all exchange data feeds.

    Strategies access market data through this manager rather than
    connecting directly, ensuring a single WebSocket per exchange.
    """

    def __init__(self, config: SniperConfig):
        self._config = config
        self._binance = BinanceFeed(
            ws_url=config.feeds.binance_ws_url,
            rest_url=config.feeds.binance_rest_url,
            symbol=config.feeds.binance_symbol,
        )
        self._chainlink_feed = None  # Lazy-initialized if oracle strategy is enabled

    async def start(self) -> None:
        await self._binance.start()
        logger.info("FeedManager started — Binance feed active")

    async def stop(self) -> None:
        await self._binance.stop()
        if self._chainlink_feed is not None:
            await self._chainlink_feed.close()
        logger.info("FeedManager stopped")

    # --- Binance data ---

    def get_latest_price(self) -> Optional[float]:
        return self._binance.get_latest_price()

    async def get_candles(self, interval: str = "1m", limit: int = 30) -> list[Candle]:
        return await self._binance.get_candles(interval=interval, limit=limit)

    def get_tick_prices(self, since: float = 0) -> list[tuple[float, float]]:
        return self._binance.get_tick_prices(since=since)

    # --- Chainlink data (lazy init) ---

    async def get_chainlink_feed(self):
        """Lazy-initialize the Chainlink oracle feed."""
        if self._chainlink_feed is None:
            from crypto_sniper.feeds.chainlink_feed import ChainlinkOracleFeed
            self._chainlink_feed = ChainlinkOracleFeed(
                rtds_url=self._config.oracle_sniper.chainlink_rtds_url,
                rpc_url=self._config.oracle_sniper.polygon_rpc_url,
            )
            await self._chainlink_feed.connect()
            logger.info("Chainlink oracle feed initialized")
        return self._chainlink_feed
