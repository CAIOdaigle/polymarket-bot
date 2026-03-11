"""
Polymarket Bayesian Trading Bot — Main Orchestrator.

Pipeline: WebSocket → OrderBook → LMSR → Signals → Bayesian → Kelly → Execute → Notify
"""

from __future__ import annotations

import asyncio
import logging
import signal
import time

from src.analysis.bayesian_engine import BayesianEngine
from src.analysis.lmsr_engine import LMSREngine
from src.analysis.signal_registry import SignalRegistry
from src.config import BotConfig, load_config
from src.execution.order_manager import OrderManager, TradeRequest
from src.execution.position_tracker import PositionTracker
from src.feed.order_book import OrderBookState
from src.feed.ws_feed import MarketFeed
from src.market.scanner import MarketScanner
from src.notifications.slack_notifier import SlackNotifier
from src.persistence.state_store import StateStore
from src.persistence.trade_log import TradeLog
from src.signals.cross_market_signal import CrossMarketSignal
from src.signals.lmsr_deviation import LMSRDeviationSignal
from src.signals.orderbook_signal import OrderBookImbalanceSignal
from src.signals.related_market_signal import RelatedMarketSignal
from src.signals.volume_signal import VolumeSignal
from src.signals.whale_tracker_signal import WhaleTrackerSignal
from src.sizing.kelly_sizer import KellySizer
from src.sizing.risk_manager import RiskManager
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


class TradingBot:
    def __init__(self, config: BotConfig):
        self.config = config
        self.scanner = MarketScanner(config)
        self.feed = MarketFeed(config.feed)
        self.lmsr = LMSREngine(config.lmsr)
        self.bayesian = BayesianEngine(config.bayesian)
        self.sizer = KellySizer(config.trading)
        self.risk_mgr = RiskManager(config.trading)
        self.order_mgr = OrderManager(config.polymarket, config.trading)
        self.positions = PositionTracker()
        self.slack = SlackNotifier(config.slack, config.trading)
        self.state_store = StateStore()
        self.trade_log = TradeLog()

        self.signal_registry = SignalRegistry()
        sig_cfg = config.signals
        if sig_cfg.orderbook_imbalance.get("enabled", True):
            self.signal_registry.register(
                OrderBookImbalanceSignal(
                    sensitivity=sig_cfg.orderbook_imbalance.get("sensitivity", 1.0),
                    min_depth_usd=sig_cfg.orderbook_imbalance.get("min_depth_usd", 100.0),
                )
            )
        if sig_cfg.lmsr_deviation.get("enabled", True):
            self.signal_registry.register(
                LMSRDeviationSignal(
                    min_deviation=sig_cfg.lmsr_deviation.get("min_deviation", 0.01),
                )
            )
        if sig_cfg.volume.get("enabled", True):
            self.signal_registry.register(
                VolumeSignal(
                    lookback_trades=sig_cfg.volume.get("lookback_trades", 50),
                )
            )
        if sig_cfg.cross_market.get("enabled", True):
            self.signal_registry.register(
                CrossMarketSignal(
                    min_inconsistency=sig_cfg.cross_market.get("min_inconsistency", 0.05),
                    max_event_markets=sig_cfg.cross_market.get("max_event_markets", 20),
                )
            )
        if sig_cfg.whale_tracker.get("enabled", True):
            self.signal_registry.register(
                WhaleTrackerSignal(
                    whale_threshold_usd=sig_cfg.whale_tracker.get("whale_threshold_usd", 500.0),
                    volume_spike_multiplier=sig_cfg.whale_tracker.get("volume_spike_multiplier", 3.0),
                    lookback_trades=sig_cfg.whale_tracker.get("lookback_trades", 100),
                    min_whale_trades=sig_cfg.whale_tracker.get("min_whale_trades", 1),
                )
            )
        if sig_cfg.related_market.get("enabled", True):
            self.signal_registry.register(
                RelatedMarketSignal(
                    min_similarity=sig_cfg.related_market.get("min_similarity", 0.3),
                    max_related=sig_cfg.related_market.get("max_related", 5),
                    min_price_divergence=sig_cfg.related_market.get("min_price_divergence", 0.10),
                )
            )

        # token_id -> OrderBookState
        self._books: dict[str, OrderBookState] = {}
        # token_id -> condition_id
        self._token_to_market: dict[str, str] = {}
        self._shutdown_event = asyncio.Event()
        self._analysis_lock = asyncio.Lock()

        # Per-market trade cooldown (condition_id -> last trade timestamp)
        self._last_trade_time: dict[str, float] = {}
        self._trade_cooldown_seconds = 60.0

        # Track update events per market (condition_id -> count of book updates)
        self._update_event_count: dict[str, int] = {}

    async def initialize(self) -> None:
        await self.state_store.initialize()

        if self.config.polymarket.private_key:
            await self.order_mgr.initialize()
        else:
            logger.warning("No private key configured — order placement disabled")

        markets = await self.scanner.scan()
        self._token_to_market = self.scanner.get_token_to_market_map()

        # Initialize order books and Bayesian beliefs
        for cid, market in markets.items():
            for token in market.tokens:
                self._books[token.token_id] = OrderBookState(token_id=token.token_id)
            initial_price = market.tokens[0].price or 0.5
            self.bayesian.initialize_market(cid, initial_price)

        # Register WebSocket callbacks
        self.feed.on_book_update(self._on_book_update)
        self.feed.on_price_change(self._on_price_change)
        self.feed.on_trade(self._on_trade)

        logger.info(
            "Bot initialized: %d markets, dry_run=%s, bankroll=$%.2f",
            len(markets),
            self.config.trading.dry_run,
            self.config.trading.total_bankroll_usd,
        )

        await self.slack.notify_startup(
            len(markets), self.config.trading.total_bankroll_usd
        )

    async def _on_book_update(self, msg: dict) -> None:
        asset_id = msg.get("asset_id", msg.get("market", ""))
        book = self._books.get(asset_id)
        if book is None:
            return

        book.update_from_snapshot(msg.get("bids", []), msg.get("asks", []))

        condition_id = self._token_to_market.get(asset_id)
        if condition_id:
            self._update_event_count[condition_id] = (
                self._update_event_count.get(condition_id, 0) + 1
            )
            await self._analyze_and_maybe_trade(condition_id)

    async def _on_price_change(self, msg: dict) -> None:
        asset_id = msg.get("asset_id", msg.get("market", ""))
        book = self._books.get(asset_id)
        if book is None:
            return

        # Price change messages contain updated levels
        for change in msg.get("changes", []):
            side = change.get("side", "")
            price = float(change.get("price", 0))
            size = float(change.get("size", 0))
            book.update_price_level(side, price, size)

        condition_id = self._token_to_market.get(asset_id)
        if condition_id:
            self._update_event_count[condition_id] = (
                self._update_event_count.get(condition_id, 0) + 1
            )
            await self._analyze_and_maybe_trade(condition_id)

    async def _on_trade(self, msg: dict) -> None:
        asset_id = msg.get("asset_id", msg.get("market", ""))
        book = self._books.get(asset_id)
        if book is None:
            return

        book.record_trade(
            {
                "price": float(msg.get("price", 0)),
                "size": float(msg.get("size", 0)),
                "side": msg.get("side", ""),
                "timestamp": msg.get("timestamp", time.time()),
            }
        )

    async def _analyze_and_maybe_trade(self, condition_id: str) -> None:
        async with self._analysis_lock:
            market = self.scanner.get_market(condition_id)
            if market is None:
                return

            # Trade cooldown — don't re-trade same market within cooldown period
            now = time.time()
            last_trade = self._last_trade_time.get(condition_id, 0)
            if now - last_trade < self._trade_cooldown_seconds:
                return

            # Require enough independent update events (not just signal types)
            update_count = self._update_event_count.get(condition_id, 0)
            min_updates = self.bayesian.min_signals
            if update_count < min_updates:
                return

            # Get the YES token's order book
            yes_book = self._books.get(market.yes_token_id)
            if yes_book is None or not yes_book.has_data:
                return

            mid = yes_book.mid_price
            if mid is None:
                return

            bids = yes_book.bids_as_tuples()
            asks = yes_book.asks_as_tuples()

            # 1. LMSR analysis
            lmsr_state = self.lmsr.compute_state(bids, asks, mid)

            # 2. Run signals (pass scanner/books for cross-market signals)
            signals = await self.signal_registry.compute_all(
                condition_id, market, yes_book, lmsr_state,
                scanner=self.scanner, books=self._books,
            )

            # 3. Bayesian update
            for sig in signals:
                self.bayesian.update(condition_id, sig)

            # 4. Get posterior
            p_hat = self.bayesian.get_estimate(condition_id)
            if p_hat is None:
                return

            if not self.bayesian.has_sufficient_signals(condition_id):
                return

            logger.info(
                "Analysis: %s mid=%.3f p_hat=%.3f b=%.1f conf=%.2f updates=%d",
                condition_id[:16],
                mid,
                p_hat,
                lmsr_state.b,
                lmsr_state.confidence,
                self._update_event_count.get(condition_id, 0),
            )

            # 5. Risk check
            total_deployed = self.positions.get_total_deployed()
            can_trade, reason = self.risk_mgr.check_can_trade(total_deployed)
            if not can_trade:
                return

            # 6. Kelly sizing
            current_pos = self.positions.get_position_usd(condition_id)
            sizing = self.sizer.compute(
                p_hat=p_hat,
                market_price_yes=mid,
                market_price_no=1 - mid,
                current_position_usd=current_pos,
                total_deployed_usd=total_deployed,
                lmsr_confidence=lmsr_state.confidence,
                signal_count=len(signals),
            )

            if not sizing.should_trade:
                return

            # 7. Execute
            token_id = (
                market.yes_token_id if "YES" in sizing.side else market.no_token_id
            )
            side = "BUY" if "BUY" in sizing.side else "SELL"

            request = TradeRequest(
                condition_id=condition_id,
                token_id=token_id,
                side=side,
                price=mid if side == "BUY" else mid,
                size=sizing.position_size_shares,
                order_type=self.config.trading.order_type,
                edge=sizing.edge,
                kelly_fraction=sizing.half_kelly_fraction,
                neg_risk=market.neg_risk,
                tick_size=market.tick_size,
            )

            order = await self.order_mgr.place_order(request)

            # Record trade time for cooldown
            self._last_trade_time[condition_id] = time.time()

            # 8. Track position
            if order.status not in ("failed",):
                self.positions.update_from_fill(
                    condition_id=condition_id,
                    token_id=token_id,
                    side="YES" if "YES" in sizing.side else "NO",
                    fill_size=sizing.position_size_shares,
                    fill_price=order.price,
                )

            # 9. Notify
            await self.slack.notify_trade(sizing, order, market)

            # 10. Persist
            await self.state_store.log_trade(
                order_id=order.order_id,
                condition_id=condition_id,
                token_id=token_id,
                side=sizing.side,
                price=order.price,
                size=sizing.position_size_shares,
                status=order.status,
                edge=sizing.edge,
                kelly_fraction=sizing.half_kelly_fraction,
                p_hat=p_hat,
                b_estimate=lmsr_state.b,
                placed_at=order.placed_at,
            )

            self.trade_log.log(
                order_id=order.order_id,
                condition_id=condition_id,
                market_question=market.question,
                side=sizing.side,
                price=order.price,
                size_shares=sizing.position_size_shares,
                size_usd=sizing.position_size_usd,
                edge=sizing.edge,
                kelly_fraction=sizing.half_kelly_fraction,
                p_hat=p_hat,
                market_price=mid,
                b_estimate=lmsr_state.b,
                confidence=sizing.confidence,
                status=order.status,
                dry_run=self.config.trading.dry_run,
            )

    # ---- Periodic tasks ----

    async def _periodic_market_rescan(self) -> None:
        interval = self.config.scanner.rescan_interval_seconds
        while not self._shutdown_event.is_set():
            await asyncio.sleep(interval)
            try:
                old_ids = set(self.scanner.get_all_token_ids())
                new_markets = await self.scanner.scan()
                self._token_to_market = self.scanner.get_token_to_market_map()

                new_ids = set(self.scanner.get_all_token_ids())
                added = new_ids - old_ids
                removed = old_ids - new_ids

                for tid in added:
                    self._books[tid] = OrderBookState(token_id=tid)
                for cid, m in new_markets.items():
                    if cid not in self.bayesian.beliefs:
                        price = m.tokens[0].price or 0.5
                        self.bayesian.initialize_market(cid, price)

                if added:
                    await self.feed.add_markets(list(added))
                if removed:
                    await self.feed.remove_markets(list(removed))

                logger.info(
                    "Rescan: +%d -%d tokens, %d markets total",
                    len(added),
                    len(removed),
                    len(new_markets),
                )
            except Exception:
                logger.exception("Market rescan failed")

    async def _periodic_order_cleanup(self) -> None:
        while not self._shutdown_event.is_set():
            await asyncio.sleep(60)
            try:
                await self.order_mgr.cancel_stale_orders()
            except Exception:
                logger.exception("Order cleanup failed")

    async def _periodic_signal_decay(self) -> None:
        while not self._shutdown_event.is_set():
            await asyncio.sleep(300)
            for cid in list(self.bayesian.beliefs.keys()):
                self.bayesian.recompute_with_decay(cid)

    async def _periodic_daily_summary(self) -> None:
        while not self._shutdown_event.is_set():
            await asyncio.sleep(3600)
            from datetime import datetime, timezone

            now = datetime.now(timezone.utc)
            if now.hour == self.config.slack.daily_summary_hour_utc:
                daily = self.risk_mgr.reset_daily()
                await self.slack.notify_daily_summary(
                    positions_count=len(self.positions.positions),
                    total_deployed=self.positions.get_total_deployed(),
                    daily_pnl=daily.realized_pnl,
                    trades_today=daily.trades,
                    bankroll=self.config.trading.total_bankroll_usd,
                )
                await self.state_store.update_daily_pnl(
                    daily.date, daily.realized_pnl, daily.trades
                )

    # ---- Lifecycle ----

    async def run(self) -> None:
        await self.initialize()

        token_ids = self.scanner.get_all_token_ids()

        tasks = [
            asyncio.create_task(self.feed.start(token_ids), name="ws_feed"),
            asyncio.create_task(self._periodic_market_rescan(), name="rescan"),
            asyncio.create_task(self._periodic_order_cleanup(), name="cleanup"),
            asyncio.create_task(self._periodic_signal_decay(), name="decay"),
            asyncio.create_task(self._periodic_daily_summary(), name="daily"),
        ]

        logger.info("Bot running — %d markets, waiting for data...", len(self.scanner.markets))

        await self._shutdown_event.wait()

        logger.info("Shutting down...")
        await self.order_mgr.cancel_all()
        await self.feed.stop()
        for task in tasks:
            task.cancel()
        await self.scanner.close()
        await self.state_store.close()
        logger.info("Shutdown complete")

    def request_shutdown(self) -> None:
        self._shutdown_event.set()


def main() -> None:
    config = load_config()
    setup_logging(config.logging)

    bot = TradingBot(config)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, bot.request_shutdown)

    try:
        loop.run_until_complete(bot.run())
    finally:
        loop.close()


if __name__ == "__main__":
    main()
