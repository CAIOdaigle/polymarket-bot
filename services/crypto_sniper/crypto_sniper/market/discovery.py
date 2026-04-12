"""Deterministic market discovery for BTC 5-min binary markets.

Slug format: btc-updown-5m-{window_ts}
Uses Gamma API to fetch market details (token IDs, prices, condition ID).
"""

from __future__ import annotations

import logging
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)

GAMMA_API = "https://gamma-api.polymarket.com"
WINDOW_SECONDS = 300


def get_current_window_ts() -> int:
    """Return the Unix timestamp of the current 5-min window start."""
    import time
    now = int(time.time())
    return now - (now % WINDOW_SECONDS)


def seconds_until_close(window_ts: int) -> float:
    """How many seconds remain in the window."""
    import time
    close_time = window_ts + WINDOW_SECONDS
    return close_time - time.time()


class MarketDiscovery:
    """Fetches and caches Polymarket BTC binary market data."""

    def __init__(self):
        self._cache: dict[int, dict] = {}
        self._session: Optional[aiohttp.ClientSession] = None

    async def _ensure_session(self) -> None:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()

    async def get_market(self, window_ts: int) -> Optional[dict]:
        """Fetch market for a given window timestamp.

        Returns dict with: condition_id, up_token, down_token, up_price,
        down_price, neg_risk, tick_size, slug
        """
        if window_ts in self._cache:
            return self._cache[window_ts]

        await self._ensure_session()

        slug = f"btc-updown-5m-{window_ts}"
        try:
            url = f"{GAMMA_API}/events"
            async with self._session.get(
                url,
                params={"slug": slug},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
        except Exception:
            logger.debug("Failed to discover market %s", slug, exc_info=True)
            return None

        if not data or not isinstance(data, list) or len(data) == 0:
            return None

        event = data[0]
        markets = event.get("markets", [])
        if not markets:
            return None

        market = markets[0]
        tokens = market.get("tokens", [])

        up_token = next(
            (t for t in tokens if t.get("outcome", "").upper() in ("UP", "YES")),
            None,
        )
        down_token = next(
            (t for t in tokens if t.get("outcome", "").upper() in ("DOWN", "NO")),
            None,
        )

        if not up_token or not down_token:
            return None

        result = {
            "condition_id": market.get("conditionId", market.get("condition_id", "")),
            "up_token": up_token.get("token_id"),
            "down_token": down_token.get("token_id"),
            "up_price": float(up_token.get("price", 0.5)),
            "down_price": float(down_token.get("price", 0.5)),
            "neg_risk": market.get("neg_risk", False),
            "tick_size": float(market.get("minimum_tick_size", 0.01)),
            "slug": slug,
            "window_ts": window_ts,
        }

        self._cache[window_ts] = result
        return result

    async def close(self) -> None:
        if self._session:
            await self._session.close()
