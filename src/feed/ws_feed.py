"""WebSocket market data feed for Polymarket CLOB."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Callable, Optional

import websockets

from src.config import FeedConfig

logger = logging.getLogger(__name__)

MARKET_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"


class MarketFeed:
    """
    Persistent WebSocket connection to Polymarket market data.

    Subscribes to token IDs and dispatches book, price_change,
    and last_trade_price events via registered callbacks.
    """

    def __init__(self, config: FeedConfig):
        self._config = config
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._subscribed_ids: set[str] = set()
        self._callbacks: dict[str, list[Callable]] = {
            "book": [],
            "price_change": [],
            "last_trade_price": [],
            "tick_size_change": [],
        }
        self._running = False
        self._reconnect_delay = 1.0

    def on_book_update(self, callback: Callable) -> None:
        self._callbacks["book"].append(callback)

    def on_price_change(self, callback: Callable) -> None:
        self._callbacks["price_change"].append(callback)

    def on_trade(self, callback: Callable) -> None:
        self._callbacks["last_trade_price"].append(callback)

    async def start(self, token_ids: list[str]) -> None:
        self._running = True
        while self._running:
            try:
                async with websockets.connect(
                    MARKET_WS_URL,
                    ping_interval=None,  # We handle heartbeat manually
                    close_timeout=10,
                ) as ws:
                    self._ws = ws
                    self._reconnect_delay = 1.0
                    logger.info("WebSocket connected")

                    await self._subscribe(token_ids)

                    await asyncio.gather(
                        self._heartbeat_loop(),
                        self._message_loop(),
                    )
            except asyncio.CancelledError:
                break
            except Exception:
                if self._running:
                    logger.warning(
                        "WebSocket disconnected, reconnecting in %.1fs",
                        self._reconnect_delay,
                        exc_info=True,
                    )
                    await asyncio.sleep(self._reconnect_delay)
                    self._reconnect_delay = min(
                        self._reconnect_delay * 2,
                        self._config.ws_reconnect_max_delay,
                    )

    async def _subscribe(self, token_ids: list[str]) -> None:
        if not token_ids:
            return
        # Subscribe in batches of 10
        for i in range(0, len(token_ids), 10):
            batch = token_ids[i : i + 10]
            msg = json.dumps({"assets_ids": batch, "type": "market"})
            await self._ws.send(msg)
            self._subscribed_ids.update(batch)
        logger.info("Subscribed to %d token IDs", len(token_ids))

    async def add_markets(self, new_token_ids: list[str]) -> None:
        if not self._ws:
            return
        try:
            for i in range(0, len(new_token_ids), 10):
                batch = new_token_ids[i : i + 10]
                msg = json.dumps({"assets_ids": batch, "type": "market"})
                await self._ws.send(msg)
                self._subscribed_ids.update(batch)
        except Exception:
            pass

    async def remove_markets(self, token_ids: list[str]) -> None:
        # Polymarket WS doesn't have an explicit unsubscribe;
        # we track it locally and ignore messages for removed IDs
        self._subscribed_ids -= set(token_ids)

    async def _heartbeat_loop(self) -> None:
        interval = self._config.heartbeat_interval_seconds
        while self._running and self._ws:
            try:
                await self._ws.send("PING")
            except Exception:
                break
            await asyncio.sleep(interval)

    async def _message_loop(self) -> None:
        async for raw_msg in self._ws:
            if raw_msg == "PONG":
                continue

            try:
                msgs = json.loads(raw_msg)
                if not isinstance(msgs, list):
                    msgs = [msgs]

                for msg in msgs:
                    event_type = msg.get("event_type")
                    asset_id = msg.get("asset_id", msg.get("market", ""))

                    if asset_id and asset_id not in self._subscribed_ids:
                        continue

                    if event_type in self._callbacks:
                        for cb in self._callbacks[event_type]:
                            try:
                                asyncio.create_task(cb(msg))
                            except Exception:
                                logger.exception("Callback error for %s", event_type)
            except json.JSONDecodeError:
                pass

    async def stop(self) -> None:
        self._running = False
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
        logger.info("WebSocket feed stopped")
