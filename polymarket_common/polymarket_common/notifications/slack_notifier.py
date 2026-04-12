"""Slack webhook notifications."""

from __future__ import annotations

import asyncio
import logging

from slack_sdk.webhook import WebhookClient

from polymarket_common.config import SlackConfig
from polymarket_common.notifications.formatters import (
    daily_summary_blocks,
    error_blocks,
    exit_blocks,
    startup_blocks,
    trade_blocks,
)

logger = logging.getLogger(__name__)


class SlackNotifier:
    def __init__(self, slack_config: SlackConfig, dry_run: bool = True):
        self._config = slack_config
        self._dry_run = dry_run
        self._webhook: WebhookClient | None = None
        if slack_config.webhook_url:
            self._webhook = WebhookClient(slack_config.webhook_url)

    async def notify_trade(
        self,
        side: str,
        price: float,
        size_shares: float,
        size_usd: float,
        edge: float,
        half_kelly: float,
        confidence: float,
        order_id: str,
        market_question: str,
        market_slug: str,
    ) -> None:
        if not self._config.notify_on_trade or not self._webhook:
            return

        # Only notify on BUY_YES entries (skip BUY_NO)
        if "YES" not in side:
            return

        blocks = trade_blocks(
            market_question=market_question,
            side=side,
            price=price,
            size_shares=size_shares,
            size_usd=size_usd,
            edge=edge,
            half_kelly=half_kelly,
            confidence=confidence,
            order_id=order_id,
            market_slug=market_slug,
            dry_run=self._dry_run,
        )
        await self._send(blocks, f"Trade: {side} {market_question[:50]}")

    async def notify_exit(
        self,
        market_question: str,
        reason: str,
        entry_price: float,
        exit_price: float,
        size_shares: float,
        pnl_pct: float,
        realized_pnl: float,
        edge_at_exit: float,
        confidence: float,
        market_slug: str,
    ) -> None:
        """Notify on ALL exits (not filtered to YES-only like entries)."""
        if not self._config.notify_on_trade or not self._webhook:
            return

        blocks = exit_blocks(
            market_question=market_question,
            reason=reason,
            entry_price=entry_price,
            exit_price=exit_price,
            size_shares=size_shares,
            pnl_pct=pnl_pct,
            realized_pnl=realized_pnl,
            edge_at_exit=edge_at_exit,
            confidence=confidence,
            market_slug=market_slug,
            dry_run=self._dry_run,
        )
        await self._send(blocks, f"Exit: {reason} {market_question[:50]}")

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
