from __future__ import annotations

import json
import logging
from typing import Optional

import aiohttp

from src.config import BotConfig
from src.market.filters import MarketFilters
from src.market.models import Event, Market, MarketToken

logger = logging.getLogger(__name__)


class MarketScanner:
    """Fetches and filters active markets from the Gamma API."""

    def __init__(self, config: BotConfig):
        self._gamma_host = config.polymarket.gamma_host
        self._filters = MarketFilters(config.scanner, config.market_filter)
        self._session: Optional[aiohttp.ClientSession] = None
        self._markets: dict[str, Market] = {}

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def scan(self) -> dict[str, Market]:
        """Full scan of all active markets. Returns condition_id -> Market."""
        session = await self._ensure_session()
        all_markets: list[Market] = []
        offset = 0
        limit = 100
        scan_complete = True

        while True:
            url = f"{self._gamma_host}/markets"
            params = {
                "closed": "false",
                "active": "true",
                "limit": str(limit),
                "offset": str(offset),
            }
            try:
                async with session.get(url, params=params) as resp:
                    if resp.status != 200:
                        logger.warning(
                            "Gamma API returned %d at offset %d — partial scan",
                            resp.status, offset,
                        )
                        scan_complete = False
                        break
                    data = await resp.json()
            except Exception:
                logger.exception("Failed to fetch markets at offset %d — partial scan", offset)
                scan_complete = False
                break

            if not data:
                break

            for raw in data:
                market = self._parse_market(raw)
                if market is not None:
                    all_markets.append(market)

            if len(data) < limit:
                break
            offset += limit

        if not scan_complete and not all_markets:
            # Total failure — keep existing markets rather than wiping them
            logger.warning("Scan failed completely, keeping %d existing markets", len(self._markets))
            return self._markets

        filtered = self._filters.apply(all_markets)
        self._markets = {m.condition_id: m for m in filtered}
        logger.info(
            "Scanned %d total, tracking %d markets%s",
            len(all_markets), len(self._markets),
            "" if scan_complete else " (partial scan)",
        )
        return self._markets

    def get_market(self, condition_id: str) -> Optional[Market]:
        return self._markets.get(condition_id)

    def get_all_token_ids(self) -> list[str]:
        ids: list[str] = []
        for m in self._markets.values():
            ids.extend(m.all_token_ids)
        return ids

    def get_token_to_market_map(self) -> dict[str, str]:
        """Returns token_id -> condition_id mapping."""
        mapping: dict[str, str] = {}
        for m in self._markets.values():
            for t in m.tokens:
                mapping[t.token_id] = m.condition_id
        return mapping

    @property
    def markets(self) -> dict[str, Market]:
        return self._markets

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    @staticmethod
    def _parse_market(raw: dict) -> Optional[Market]:
        try:
            condition_id = raw.get("conditionId") or raw.get("condition_id", "")
            if not condition_id:
                return None

            # Parse clobTokenIds — may be a JSON string or a list
            token_ids_raw = raw.get("clobTokenIds")
            if isinstance(token_ids_raw, str):
                token_ids = json.loads(token_ids_raw)
            elif isinstance(token_ids_raw, list):
                token_ids = token_ids_raw
            else:
                return None

            if not token_ids:
                return None

            # Parse outcome prices
            outcome_prices_raw = raw.get("outcomePrices")
            if isinstance(outcome_prices_raw, str):
                outcome_prices = json.loads(outcome_prices_raw)
            elif isinstance(outcome_prices_raw, list):
                outcome_prices = outcome_prices_raw
            else:
                outcome_prices = []

            outcomes_raw = raw.get("outcomes")
            if isinstance(outcomes_raw, str):
                outcomes = json.loads(outcomes_raw)
            elif isinstance(outcomes_raw, list):
                outcomes = outcomes_raw
            else:
                outcomes = ["Yes", "No"]

            tokens: list[MarketToken] = []
            for i, tid in enumerate(token_ids):
                outcome = outcomes[i] if i < len(outcomes) else f"Outcome {i}"
                price = float(outcome_prices[i]) if i < len(outcome_prices) else None
                tokens.append(MarketToken(token_id=tid, outcome=outcome, price=price))

            # Parse tick size
            min_tick = raw.get("minimum_tick_size")
            tick_size = float(min_tick) if min_tick else 0.01

            tags_raw = raw.get("tags", [])
            tags = tags_raw if isinstance(tags_raw, list) else []

            return Market(
                condition_id=condition_id,
                question=raw.get("question", ""),
                slug=raw.get("slug", ""),
                tokens=tokens,
                end_date=raw.get("endDate", raw.get("end_date_iso", "")),
                volume_24h=float(raw.get("volume24hr", 0) or 0),
                liquidity=float(raw.get("liquidityNum", 0) or 0),
                neg_risk=bool(raw.get("negRisk", False)),
                tick_size=tick_size,
                active=bool(raw.get("active", True)),
                tags=[t.get("label", t) if isinstance(t, dict) else str(t) for t in tags],
                event_slug=raw.get("eventSlug", raw.get("event_slug", "")),
            )
        except Exception:
            logger.debug("Failed to parse market: %s", raw.get("question", "unknown"), exc_info=True)
            return None
