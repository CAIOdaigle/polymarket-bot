"""
Polymarket Bayesian Trading Bot — Main Orchestrator.

Pipeline: WebSocket → OrderBook → LMSR → Signals → Bayesian → Kelly → Execute → Notify
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import time

from src.analysis.bayesian_engine import BayesianEngine
from src.analysis.lmsr_engine import LMSREngine
from src.analysis.signal_registry import SignalRegistry
from src.config import BotConfig, load_config
from src.execution.exit_manager import ExitManager
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
from src.sizing.entry_liquidity_gate import EntryLiquidityGate, EntryLiquidityConfig
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
        self.sizer = KellySizer(config.kelly)
        self.entry_gate = EntryLiquidityGate(EntryLiquidityConfig(
            min_fill_coverage=config.entry_liquidity.min_fill_coverage,
            max_slippage_pct=config.entry_liquidity.max_slippage_pct,
            min_absolute_depth=config.entry_liquidity.min_absolute_depth,
            check_exit_viability=config.entry_liquidity.check_exit_viability,
        ))
        self.risk_mgr = RiskManager(config.trading)
        self.order_mgr = OrderManager(config.polymarket, config.trading)
        self.positions = PositionTracker()
        self.slack = SlackNotifier(config.slack, config.trading)
        self.state_store = StateStore()
        self.trade_log = TradeLog()
        self.exit_mgr = ExitManager(
            config.exit, self.order_mgr, self.positions, self.state_store, self.slack,
        )

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
        self._trade_cooldown_seconds = 300.0  # 5 min — prevent hammering same market

        # Track update events per market (condition_id -> count of book updates)
        self._update_event_count: dict[str, int] = {}

    async def initialize(self) -> None:
        await self.state_store.initialize()

        # Restore positions from SQLite (crash recovery)
        persisted = await self.state_store.load_all_positions()
        for pos in persisted:
            self.positions.restore_position(pos)
        if persisted:
            logger.info("Restored %d positions from database", len(persisted))

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
            await self._check_exits_for_market(condition_id)

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
            await self._check_exits_for_market(condition_id)

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

            # Set cooldown immediately to prevent rapid-fire duplicates
            # (other coroutines waiting on the lock will see this timestamp)
            self._last_trade_time[condition_id] = now

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

            # 3. Update Bayesian with market price as prior anchor
            self.bayesian.update_market_price(condition_id, mid)
            for sig in signals:
                self.bayesian.update(condition_id, sig)

            # Reset update counter so next analysis requires fresh data
            self._update_event_count[condition_id] = 0

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

            # 5b. Hedge prevention — don't buy opposite side of same market
            for existing_pos in self.positions.get_all_open():
                if existing_pos.condition_id == condition_id:
                    # We already have a position on this market.
                    # Block any new entry (Kelly will pick a side; if it picks
                    # the opposite side we'd be self-hedging).
                    logger.info(
                        "Hedge prevention: already hold %s %s on %s — skipping new entry",
                        existing_pos.side,
                        f"{existing_pos.size:.1f} shares",
                        condition_id[:16],
                    )
                    return

            # 5c. Event exposure cap — prevent concentration in one event cluster
            event_slug = market.event_slug
            if event_slug:
                max_event_usd = (
                    self.config.trading.max_event_exposure_pct
                    * self.config.trading.total_bankroll_usd
                )
                event_exposure = 0.0
                for ep in self.positions.get_all_open():
                    ep_market = self.scanner.get_market(ep.condition_id)
                    if ep_market is not None and ep_market.event_slug == event_slug:
                        event_exposure += ep.cost_basis
                if event_exposure >= max_event_usd:
                    logger.info(
                        "Event exposure cap: %s has $%.2f deployed (max $%.2f) — skipping",
                        event_slug[:30],
                        event_exposure,
                        max_event_usd,
                    )
                    return

            # 6. Kelly sizing (now uses ask prices, not mid)
            no_book = self._books.get(market.no_token_id)
            current_pos = self.positions.get_position_usd(condition_id)
            sizing = self.sizer.compute(
                p_hat=p_hat,
                market_price_yes=mid,
                market_price_no=1 - mid,
                current_position_usd=current_pos,
                total_deployed_usd=total_deployed,
                lmsr_confidence=lmsr_state.confidence,
                signal_count=len(signals),
                book_yes=yes_book,
                book_no=no_book,
                market=market,
            )

            if not sizing.should_trade:
                return

            # 7. Entry liquidity gate — check both sides before committing
            token_id = (
                market.yes_token_id if "YES" in sizing.side else market.no_token_id
            )
            entry_book = self._books.get(token_id)
            if entry_book is not None:
                bid_price = entry_book.best_bid or 0.0
                ask_price = sizing.ask_price if sizing.ask_price > 0 else mid
                gate_ok, gate_reason = self.entry_gate.check_both_sides(
                    book_entry=entry_book,
                    book_exit=entry_book,  # same token — check bid side for exit
                    size_usd=sizing.position_size_usd,
                    ask_price=ask_price,
                    bid_price=bid_price,
                )
                if not gate_ok:
                    logger.info(
                        "Entry blocked by liquidity gate: %s — %s",
                        condition_id[:16], gate_reason,
                    )
                    return

            side = "BUY" if "BUY" in sizing.side else "SELL"

            # Use ask price (true fill cost) instead of mid for order placement
            order_price = sizing.ask_price if sizing.ask_price > 0 else mid

            request = TradeRequest(
                condition_id=condition_id,
                token_id=token_id,
                side=side,
                price=order_price,
                size=sizing.position_size_shares,
                order_type=self.config.kelly.order_type,
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
                pos = self.positions.update_from_fill(
                    condition_id=condition_id,
                    token_id=token_id,
                    side="YES" if "YES" in sizing.side else "NO",
                    fill_size=sizing.position_size_shares,
                    fill_price=order.price,
                )
                await self.state_store.save_position(pos)

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
                confidence=lmsr_state.confidence,
                market_question=market.question if market else "",
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

    # ---- Exit management ----

    async def _check_exits_for_market(self, condition_id: str) -> None:
        """Check if any positions in this market should be exited."""
        if not self.config.exit.enabled:
            return

        market = self.scanner.get_market(condition_id)
        if market is None:
            return

        for token in market.tokens:
            pos = self.positions.get_position(token.token_id)
            if pos is None or pos.size <= 0:
                continue

            book = self._books.get(token.token_id)
            if book is None or not book.has_data:
                continue

            # Update high-water mark on every tick
            if book.best_bid is not None:
                pos.update_high_water_mark(book.best_bid)

            p_hat = self.bayesian.get_estimate(condition_id)
            if p_hat is None:
                continue

            # For NO positions, pass (1 - p_hat) as the NO probability
            pos_p_hat = p_hat if pos.side == "YES" else (1 - p_hat)

            # Get LMSR confidence
            yes_book = self._books.get(market.yes_token_id)
            confidence = 0.5
            if yes_book and yes_book.has_data and yes_book.mid_price is not None:
                lmsr_state = self.lmsr.compute_state(
                    yes_book.bids_as_tuples(), yes_book.asks_as_tuples(), yes_book.mid_price
                )
                confidence = lmsr_state.confidence

            signal = self.exit_mgr.evaluate_position(
                pos, book, pos_p_hat, confidence, market
            )
            if signal is None:
                continue

            executed = self.exit_mgr.execute_exit(
                signal, market, dry_run=self.config.trading.dry_run
            )
            if executed:
                pnl = (signal.current_price - pos.avg_price) * signal.size_to_sell
                self.risk_mgr.record_pnl(pnl)
                await self._log_exit(signal, market, p_hat)

    async def _log_exit(self, signal, market, p_hat: float) -> None:
        """Log an exit to Slack, SQLite, and CSV."""
        realized_pnl = signal.pnl_pct * signal.entry_price * signal.size_to_sell

        # Slack
        await self.slack.notify_exit(
            market_question=market.question if market else "Unknown",
            reason=signal.reason.value,
            entry_price=signal.entry_price,
            exit_price=signal.current_price,
            size_shares=signal.size_to_sell,
            pnl_pct=signal.pnl_pct,
            realized_pnl=realized_pnl,
            edge_at_exit=signal.edge_at_exit,
            confidence=signal.confidence,
            market_slug=market.event_slug or market.slug if market else "",
        )

        # SQLite
        import uuid
        exit_order_id = f"exit_{uuid.uuid4().hex[:12]}"
        await self.state_store.log_trade(
            order_id=exit_order_id,
            condition_id=signal.condition_id,
            token_id=signal.token_id,
            side=f"SELL_{signal.reason.value.upper()}",
            price=signal.current_price,
            size=signal.size_to_sell,
            status="dry_run" if self.config.trading.dry_run else "filled",
            edge=signal.edge_at_exit,
            kelly_fraction=0.0,
            p_hat=p_hat or 0.0,
            b_estimate=0.0,
            placed_at=time.time(),
            confidence=signal.confidence,
            market_question=market.question if market else "Unknown",
        )

        # CSV
        self.trade_log.log(
            order_id=exit_order_id,
            condition_id=signal.condition_id,
            market_question=market.question if market else "Unknown",
            side=f"SELL_{signal.reason.value.upper()}",
            price=signal.current_price,
            size_shares=signal.size_to_sell,
            size_usd=signal.size_to_sell * signal.current_price,
            edge=signal.edge_at_exit,
            kelly_fraction=0.0,
            p_hat=p_hat or 0.0,
            market_price=signal.current_price,
            b_estimate=0.0,
            confidence=signal.confidence,
            status="dry_run" if self.config.trading.dry_run else "filled",
            dry_run=self.config.trading.dry_run,
        )

        # Delete position from persistence if fully closed
        remaining = self.positions.get_position(signal.token_id)
        if remaining is None:
            await self.state_store.delete_position(signal.token_id)

    async def _periodic_exit_sweep(self) -> None:
        """Periodic sweep for time-based exits and position monitoring."""
        interval = self.config.exit.check_interval_seconds
        while not self._shutdown_event.is_set():
            await asyncio.sleep(interval)
            if not self.config.exit.enabled:
                continue
            try:
                signals = self.exit_mgr.check_all_positions(
                    books=self._books, bayesian=self.bayesian, scanner=self.scanner,
                )
                for signal in signals:
                    market = self.scanner.get_market(signal.condition_id)
                    pos = self.positions.get_position(signal.token_id)
                    executed = self.exit_mgr.execute_exit(
                        signal, market, dry_run=self.config.trading.dry_run
                    )
                    if executed:
                        pnl = (signal.current_price - signal.entry_price) * signal.size_to_sell
                        self.risk_mgr.record_pnl(pnl)
                        p_hat = self.bayesian.get_estimate(signal.condition_id)
                        await self._log_exit(signal, market, p_hat or 0.0)
            except Exception:
                logger.exception("Exit sweep failed")

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
            asyncio.create_task(self._periodic_exit_sweep(), name="exit_sweep"),
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

    # Safety check: live trading requires explicit confirmation
    if not config.trading.dry_run:
        confirm = os.environ.get("CONFIRM_LIVE_TRADING", "").lower()
        if confirm != "true":
            logger.critical(
                "LIVE TRADING is enabled (dry_run=false) but CONFIRM_LIVE_TRADING "
                "env var is not set to 'true'. Refusing to start. Set "
                "CONFIRM_LIVE_TRADING=true to confirm you want live orders."
            )
            raise SystemExit(1)
        logger.warning(
            "*** LIVE TRADING MODE — real orders will be placed ***"
        )

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
