"""Slack webhook notifications."""

from __future__ import annotations

import asyncio
import logging

from slack_sdk.webhook import WebhookClient

from src.config import SlackConfig, TradingConfig
from src.execution.order_manager import OrderRecord
from src.market.models import Market
from src.notifications.formatters import (
    daily_summary_blocks,
    error_blocks,
    startup_blocks,
    trade_blocks,
)
from src.sizing.kelly_sizer import SizingResult

logger = logging.getLogger(__name__)


class SlackNotifier:
    def __init__(self, slack_config: SlackConfig, trading_config: TradingConfig):
        self._config = slack_config
        self._dry_run = trading_config.dry_run
        self._webhook: WebhookClient | None = None
        if slack_config.webhook_url:
            self._webhook = WebhookClient(slack_config.webhook_url)

    async def notify_trade(
        self,
        sizing: SizingResult,
        order: OrderRecord,
        market: Market,
    ) -> None:
        if not self._config.notify_on_trade or not self._webhook:
            return

        # Only notify on BUY_YES entries (skip BUY_NO)
        if "YES" not in sizing.side:
            return

        blocks = trade_blocks(
            market_question=market.question,
            side=sizing.side,
            price=order.price,
            size_shares=sizing.position_size_shares,
            size_usd=sizing.position_size_usd,
            edge=sizing.edge,
            half_kelly=sizing.half_kelly_fraction,
            confidence=sizing.confidence,
            order_id=order.order_id,
            market_slug=market.event_slug or market.slug,
            dry_run=self._dry_run,
        )
        await self._send(blocks, f"Trade: {sizing.side} {market.question[:50]}")

    async def notify_error(self, error: Exception, context: str) -> None:
        if not self._config.notify_on_error or not self._webhook:
            return

        blocks = error_blocks(str(error), context)
        await self._send(blocks, f"Error: {context}")

    async def notify_startup(self, market_count: int, bankroll: float) -> None:
        if not self._config.notify_on_startup or not self._webhook:
            return

        blocks = startup_blocks(market_count, bankroll, self._dry_run)
        await self._send(blocks, "Bot started")

    async def notify_daily_summary(
        self,
        positions_count: int,
        total_deployed: float,
        daily_pnl: float,
        trades_today: int,
        bankroll: float,
    ) -> None:
        if not self._webhook:
            return

        blocks = daily_summary_blocks(
            positions_count, total_deployed, daily_pnl, trades_today, bankroll
        )
        await self._send(blocks, "Daily summary")

    async def _send(self, blocks: list[dict], fallback_text: str) -> None:
        if not self._webhook:
            return

        loop = asyncio.get_event_loop()
        for attempt in range(3):
            try:
                resp = await loop.run_in_executor(
                    None,
                    lambda: self._webhook.send(blocks=blocks, text=fallback_text),
                )
                if resp.status_code == 200:
                    return
                if resp.status_code == 429:
                    retry_after = 2 ** (attempt + 1)
                    await asyncio.sleep(retry_after)
                    continue
                logger.warning("Slack returned %d: %s", resp.status_code, resp.body)
                return
            except Exception:
                logger.exception("Slack send failed (attempt %d)", attempt + 1)
                await asyncio.sleep(2**attempt)
