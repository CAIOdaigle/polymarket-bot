from __future__ import annotations

import logging
from datetime import datetime, timezone

from src.config import MarketFilterConfig, ScannerConfig
from src.market.models import Market

logger = logging.getLogger(__name__)


class MarketFilters:
    def __init__(self, scanner_cfg: ScannerConfig, filter_cfg: MarketFilterConfig):
        self._scanner = scanner_cfg
        self._filter = filter_cfg

    def apply(self, markets: list[Market]) -> list[Market]:
        result = []
        for m in markets:
            if not self._passes(m):
                continue
            result.append(m)
        logger.info("Filtered %d -> %d markets", len(markets), len(result))
        return result

    def _passes(self, m: Market) -> bool:
        # Slug include/exclude
        if self._filter.include_slugs and m.slug not in self._filter.include_slugs:
            return False
        if m.slug in self._filter.exclude_slugs:
            return False

        # Tag include/exclude
        if self._filter.include_tags:
            if not any(t in self._filter.include_tags for t in m.tags):
                return False
        if any(t in self._filter.exclude_tags for t in m.tags):
            return False

        # Must be active
        if not m.active:
            return False

        # Minimum volume
        if m.volume_24h < self._scanner.min_volume_24h_usd:
            return False

        # Minimum liquidity
        if m.liquidity < self._scanner.min_liquidity_usd:
            return False

        # Minimum time to expiry
        try:
            end = datetime.fromisoformat(m.end_date.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            hours_left = (end - now).total_seconds() / 3600
            if hours_left < self._scanner.min_hours_to_expiry:
                return False
        except (ValueError, TypeError):
            pass  # If we can't parse, let it through

        return True
