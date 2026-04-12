"""Slack webhook notifications — compatibility wrapper.

The actual SlackNotifier implementation lives in polymarket_common.
This wrapper maintains backward compatibility with the old interface
that accepted SizingResult and Market objects directly.
"""

from __future__ import annotations

import logging

from polymarket_common.notifications.slack_notifier import SlackNotifier as _BaseNotifier
from polymarket_common.config import SlackConfig

from src.sizing.kelly_sizer import SizingResult
from src.market.models import Market
from src.execution.order_manager import OrderRecord

logger = logging.getLogger(__name__)


class SlackNotifier(_BaseNotifier):
    """Backward-compatible wrapper that unpacks SizingResult/Market for the base notifier."""

    def __init__(self, slack_config: SlackConfig, trading_config):
        super().__init__(slack_config, dry_run=trading_config.dry_run)

    async def notify_trade(
        self,
        sizing: SizingResult,
        order: OrderRecord,
        market: Market,
    ) -> None:
        """Unpack SizingResult/Market and delegate to the primitive-based base method."""
        await super().notify_trade(
            side=sizing.side,
            price=order.price,
            size_shares=sizing.position_size_shares,
            size_usd=sizing.position_size_usd,
            edge=sizing.edge,
            half_kelly=sizing.half_kelly_fraction,
            confidence=sizing.confidence,
            order_id=order.order_id,
            market_question=market.question,
            market_slug=market.event_slug or market.slug,
        )
