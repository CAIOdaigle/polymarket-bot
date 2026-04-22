"""Deterministic market discovery for BTC 5-min binary markets.

Slug format: btc-updown-5m-{window_ts}
Uses Gamma API to fetch market details (token IDs, prices, condition ID).
"""

from __future__ import annotations

import json
import logging
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"
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
            logger.debug("Gamma returned no events for slug %s", slug)
            return None

        event = data[0]
        markets = event.get("markets", [])
        if not markets:
            return None

        market = markets[0]

        # Gamma returns these as JSON-encoded STRINGS, not native lists:
        #   outcomes:       '["Up", "Down"]'
        #   clobTokenIds:   '["<token_up>", "<token_down>"]'
        #   outcomePrices:  '["0.485", "0.515"]'  (mid/last, NOT best ask)
        try:
            outcomes = json.loads(market.get("outcomes", "[]"))
            token_ids = json.loads(market.get("clobTokenIds", "[]"))
            outcome_prices = json.loads(market.get("outcomePrices", "[]"))
        except (json.JSONDecodeError, TypeError):
            logger.warning("Gamma market has malformed outcomes/tokens: %s", slug)
            return None

        if len(outcomes) != len(token_ids) or not outcomes:
            return None

        # Build outcome -> (token_id, gamma_price) map
        outcome_map: dict[str, tuple[str, float]] = {}
        for i, oc in enumerate(outcomes):
            tid = token_ids[i] if i < len(token_ids) else ""
            try:
                px = float(outcome_prices[i]) if i < len(outcome_prices) else 0.5
            except (ValueError, TypeError):
                px = 0.5
            outcome_map[oc.upper()] = (tid, px)

        up_entry = outcome_map.get("UP") or outcome_map.get("YES")
        down_entry = outcome_map.get("DOWN") or outcome_map.get("NO")
        if not up_entry or not down_entry:
            logger.warning(
                "Gamma market missing UP/DOWN outcomes: %s (got %s)",
                slug, list(outcome_map.keys()),
            )
            return None

        result = {
            "condition_id": market.get("conditionId", ""),
            "up_token": up_entry[0],
            "down_token": down_entry[0],
            "up_price_mid": up_entry[1],      # Gamma mid-price (reference only)
            "down_price_mid": down_entry[1],  # Gamma mid-price (reference only)
            "neg_risk": market.get("negRisk", market.get("neg_risk", False)),
            "tick_size": float(market.get("orderPriceMinTickSize",
                                market.get("minimum_tick_size", 0.01)) or 0.01),
            "slug": slug,
            "window_ts": window_ts,
        }

        self._cache[window_ts] = result
        return result

    async def get_best_ask(
        self, token_id: str, min_size: float = 1.0
    ) -> Optional[float]:
        """Fetch the current best ASK price from the Polymarket CLOB order book.

        The ASK is the lowest price someone is willing to SELL at — i.e., the
        price you'd pay to market-buy right now. Derived directly from the
        `/book` endpoint (the `/price` endpoint has opaque semantics and
        returns synthetic values we cannot trust).

        Returns None if:
          - The book query fails
          - The ask side is empty (no one is selling — cannot market-buy)
          - The top ask has size < min_size (not enough depth to fill a small bet)

        The strategy MUST skip the trade on None. No estimation fallback.
        """
        if not token_id:
            return None
        await self._ensure_session()
        try:
            url = f"{CLOB_API}/book"
            async with self._session.get(
                url,
                params={"token_id": token_id},
                timeout=aiohttp.ClientTimeout(total=3),
            ) as resp:
                if resp.status != 200:
                    logger.debug("CLOB /book %s returned %d", token_id[:10], resp.status)
                    return None
                book = await resp.json()
        except Exception:
            logger.debug("Failed to fetch book for %s", token_id[:10], exc_info=True)
            return None

        asks = book.get("asks") or []
        if not asks:
            # Empty ask side — nobody's selling. We can't market-buy this token.
            return None

        # Polymarket returns asks in DESCENDING price order (highest first).
        # Best ask (lowest price to buy) is the LAST entry.
        # Scan from lowest to highest for the first level with sufficient size.
        try:
            sorted_asks = sorted(asks, key=lambda a: float(a["price"]))
        except (KeyError, ValueError, TypeError):
            return None

        for level in sorted_asks:
            try:
                price = float(level["price"])
                size = float(level["size"])
            except (KeyError, ValueError, TypeError):
                continue
            if price <= 0 or price >= 1.0:
                continue
            if size < min_size:
                continue
            return price
        return None

    async def get_live_token_price(
        self, window_ts: int, direction: str
    ) -> Optional[float]:
        """Convenience: fetch real best-ask for the UP or DOWN token of a window.

        Returns None if market not yet available or CLOB query fails.
        The strategy MUST skip the trade on None — no estimation fallback.
        """
        market = await self.get_market(window_ts)
        if market is None:
            return None
        token_key = "up_token" if direction.upper() == "UP" else "down_token"
        token_id = market.get(token_key)
        if not token_id:
            return None
        return await self.get_best_ask(token_id)

    async def close(self) -> None:
        if self._session:
            await self._session.close()
