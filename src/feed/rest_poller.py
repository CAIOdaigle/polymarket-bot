"""REST API fallback poller for when WebSocket is down."""

from __future__ import annotations

import asyncio
import logging
from typing import Callable, Optional

import aiohttp

from src.config import FeedConfig

logger = logging.getLogger(__name__)


class RestPoller:
    """Polls order books via REST API as a fallback."""

    def __init__(self, clob_host: str, config: FeedConfig):
        self._host = clob_host
        self._interval = config.rest_poll_interval_seconds
        self._session: Optional[aiohttp.ClientSession] = None
        self._callback: Optional[Callable] = None
        self._running = False

    def on_book_update(self, callback: Callable) -> None:
        self._callback = callback

    async def start(self, token_ids: list[str]) -> None:
        self._running = True
        self._session = aiohttp.ClientSession()
        logger.info("REST poller started for %d tokens", len(token_ids))

        while self._running:
            for tid in token_ids:
                if not self._running:
                    break
                try:
                    url = f"{self._host}/book"
                    async with self._session.get(url, params={"token_id": tid}) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            if self._callback:
                                await self._callback(
                                    {
                                        "event_type": "book",
                                        "asset_id": tid,
                                        "market": tid,
                                        "bids": data.get("bids", []),
                                        "asks": data.get("asks", []),
                                    }
                                )
                except Exception:
                    logger.debug("REST poll failed for %s", tid[:8], exc_info=True)

            await asyncio.sleep(self._interval)

    async def stop(self) -> None:
        self._running = False
        if self._session and not self._session.closed:
            await self._session.close()
