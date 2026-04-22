"""Crypto Sniper Service — entry point.

Runs fast crypto-linked trading strategies against Polymarket binary markets.
"""

from __future__ import annotations

import asyncio
import logging
import os

from polymarket_common.execution.order_manager import OrderManager
from polymarket_common.execution.position_tracker import PositionTracker
from polymarket_common.persistence.state_store import StateStore
from polymarket_common.notifications.slack_notifier import SlackNotifier
from polymarket_common.utils.logging_config import setup_logging

from crypto_sniper.config import load_sniper_config
from crypto_sniper.feeds.feed_manager import FeedManager
from crypto_sniper.runner import StrategyRunner
from crypto_sniper.strategies.ta_sniper import TASniperStrategy
from crypto_sniper.strategies.oracle_sniper import OracleSniperStrategy

logger = logging.getLogger(__name__)


async def run() -> None:
    config = load_sniper_config()
    setup_logging(config.logging)

    logger.info(
        "Crypto Sniper starting — asset=%s, slug_prefix=%s, binance=%s",
        config.asset, config.slug_prefix, config.feeds.binance_symbol,
    )

    # Safety check
    if not config.trading.dry_run:
        confirm = os.environ.get("CONFIRM_LIVE_TRADING", "").lower()
        if confirm != "true":
            logger.critical(
                "LIVE TRADING enabled but CONFIRM_LIVE_TRADING not set. "
                "Set CONFIRM_LIVE_TRADING=true to confirm."
            )
            raise SystemExit(1)
        logger.warning("*** LIVE TRADING MODE — real orders ***")

    # Shared infrastructure
    order_mgr = OrderManager(config.polymarket, config.trading)
    if config.polymarket.private_key:
        await order_mgr.initialize()
    else:
        logger.warning("No private key — order placement disabled")

    positions = PositionTracker()
    state_store = StateStore()
    await state_store.initialize()
    slack = SlackNotifier(config.slack, dry_run=config.trading.dry_run)

    # Feed manager
    feed_mgr = FeedManager(config)
    await feed_mgr.start()

    # Strategy runner
    runner = StrategyRunner(
        config=config,
        order_mgr=order_mgr,
        positions=positions,
        state_store=state_store,
        slack=slack,
        feed_mgr=feed_mgr,
    )

    # Register enabled strategies
    if config.strategies.ta_sniper:
        runner.register(TASniperStrategy(config.ta_sniper))

    if config.strategies.oracle_sniper:
        runner.register(OracleSniperStrategy(config.oracle_sniper))

    if not runner._strategies:
        logger.error("No strategies enabled — exiting")
        raise SystemExit(1)

    await slack.notify_startup(
        market_count=len(runner._strategies),
        bankroll=config.trading.total_bankroll_usd,
    )

    try:
        await runner.run()
    finally:
        await feed_mgr.stop()
        await state_store.close()
        await order_mgr.cancel_all()


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
