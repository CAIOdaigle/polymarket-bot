"""Strategy runner — manages the 5-min window loop and strategy dispatch."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import time
from pathlib import Path
from typing import Optional

from polymarket_common.execution.order_manager import OrderManager, TradeRequest
from polymarket_common.execution.position_tracker import PositionTracker
from polymarket_common.persistence.state_store import StateStore
from polymarket_common.notifications.slack_notifier import SlackNotifier

from crypto_sniper.config import SniperConfig
from crypto_sniper.feeds.feed_manager import FeedManager
from crypto_sniper.strategies.base import BaseStrategy, TradeSignal
from crypto_sniper.sizing.kelly import kelly_bet, calculate_pnl
from crypto_sniper.market.discovery import MarketDiscovery

logger = logging.getLogger(__name__)

WINDOW_SECONDS = 300  # 5-minute windows


def _current_window_ts() -> int:
    """Current 5-minute window start timestamp."""
    now = int(time.time())
    return now - (now % WINDOW_SECONDS)


class StrategyRunner:
    """Orchestrates strategy evaluation and order execution per 5-min window."""

    def __init__(
        self,
        config: SniperConfig,
        order_mgr: OrderManager,
        positions: PositionTracker,
        state_store: StateStore,
        slack: SlackNotifier,
        feed_mgr: FeedManager,
    ):
        self.config = config
        self.order_mgr = order_mgr
        self.positions = positions
        self.state_store = state_store
        self.slack = slack
        self.feed_mgr = feed_mgr
        self.discovery = MarketDiscovery()

        self._strategies: list[BaseStrategy] = []
        self._shutdown_event = asyncio.Event()
        self._bankroll = config.trading.total_bankroll_usd
        self._original_bankroll = self._bankroll
        self._trade_count = 0
        self._win_count = 0
        self._loss_count = 0
        self._traded_windows: set[int] = set()
        self._stats_path = Path(os.environ.get("SNIPER_DATA_DIR", "/app/data")) / "sniper_stats.json"
        self._write_stats_snapshot()

    def register(self, strategy: BaseStrategy) -> None:
        self._strategies.append(strategy)
        logger.info("Registered strategy: %s", strategy.name)

    async def run(self) -> None:
        """Main loop — runs until shutdown signal."""
        # Initialize all strategies
        for strategy in self._strategies:
            await strategy.initialize(self.feed_mgr)

        # Setup signal handlers
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._shutdown_event.set)

        logger.info(
            "StrategyRunner started — %d strategies, bankroll=$%.2f, dry_run=%s",
            len(self._strategies),
            self._bankroll,
            self.config.trading.dry_run,
        )

        # Periodic stats snapshot refresher (keeps btc_price fresh)
        stats_task = asyncio.create_task(self._periodic_stats_refresh())

        while not self._shutdown_event.is_set():
            try:
                await self._run_window()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Window loop error")
                await asyncio.sleep(10)

        # Shutdown
        stats_task.cancel()
        logger.info("StrategyRunner shutting down...")
        for strategy in self._strategies:
            await strategy.shutdown()
        logger.info(
            "Shutdown complete — %d trades, W/L=%d/%d, PnL=$%.2f",
            self._trade_count,
            self._win_count,
            self._loss_count,
            self._bankroll - self._original_bankroll,
        )

    async def _run_window(self) -> None:
        """Wait for the next window entry point, evaluate strategies, maybe trade."""
        window_ts = _current_window_ts()
        close_time = window_ts + WINDOW_SECONDS

        # Skip if we already traded this window
        if window_ts in self._traded_windows:
            next_window = close_time + 1
            wait = next_window - time.time()
            if wait > 0:
                await asyncio.sleep(min(wait, 5.0))
            return

        # Determine the earliest entry point across all strategies
        earliest_entry = max(
            s.entry_window_seconds[0] for s in self._strategies
        )
        latest_exit = min(
            s.entry_window_seconds[1] for s in self._strategies
        )

        # Wait until we're inside the entry window
        entry_time = close_time - earliest_entry
        now = time.time()
        if now < entry_time:
            wait = entry_time - now
            if wait > 60:
                # Sleep in chunks to stay responsive to shutdown
                await asyncio.sleep(min(wait, 5.0))
                return
            await asyncio.sleep(wait)

        if self._shutdown_event.is_set():
            return

        # Get window open price
        window_open = await self._get_window_open_price(window_ts)
        if window_open is None:
            logger.warning("Could not determine window open price, skipping")
            self._traded_windows.add(window_ts)
            return

        # Poll strategies until one fires or deadline
        hard_deadline = close_time - latest_exit
        best_signal: Optional[TradeSignal] = None

        while time.time() < hard_deadline and not self._shutdown_event.is_set():
            for strategy in self._strategies:
                secs_remaining = close_time - time.time()
                entry_early, entry_late = strategy.entry_window_seconds
                if not (entry_late <= secs_remaining <= entry_early):
                    continue

                signal = await strategy.evaluate(window_ts, secs_remaining)
                if signal is not None:
                    if best_signal is None or signal.confidence > best_signal.confidence:
                        best_signal = signal

            if best_signal is not None and best_signal.confidence >= 0.30:
                break

            await asyncio.sleep(
                min(s.eval_interval_seconds for s in self._strategies)
            )

        if best_signal is None:
            self._traded_windows.add(window_ts)
            return

        # Size the bet
        mode = self.config.ta_sniper.mode
        if best_signal.strategy_name == "oracle_sniper":
            mode = "oracle"

        bet_size = kelly_bet(
            bankroll=self._bankroll,
            model_prob=best_signal.confidence,
            token_price=best_signal.token_price,
            confidence=best_signal.confidence,
            mode=mode,
            kelly_config=self.config.kelly,
        )

        if bet_size <= 0:
            logger.info(
                "Kelly returned 0 for %s (conf=%.2f, token=$%.3f) — skipping",
                best_signal.strategy_name,
                best_signal.confidence,
                best_signal.token_price,
            )
            self._traded_windows.add(window_ts)
            return

        # Execute trade
        order_status = await self._execute_trade(
            window_ts, best_signal, bet_size
        )

        self._traded_windows.add(window_ts)
        self._trade_count += 1

        logger.info(
            "%s TRADE: %s conf=%.1f%% score=%.2f token=$%.3f bet=$%.2f status=%s",
            best_signal.strategy_name.upper(),
            best_signal.direction,
            best_signal.confidence * 100,
            best_signal.score,
            best_signal.token_price,
            bet_size,
            order_status,
        )

        # Wait for resolution and check outcome
        remaining = close_time - time.time() + 2
        if remaining > 0:
            await asyncio.sleep(remaining)

        await self._check_resolution(
            best_signal, window_ts, window_open, bet_size
        )

    async def _get_window_open_price(self, window_ts: int) -> Optional[float]:
        """Get BTC price at window open from candle data."""
        candles = await self.feed_mgr.get_candles(limit=10)
        window_open_ms = window_ts * 1000
        for c in candles:
            if c.open_time <= window_open_ms <= c.close_time:
                return c.open
        if candles:
            return candles[-5].open if len(candles) > 5 else candles[0].open
        return None

    async def _execute_trade(
        self,
        window_ts: int,
        signal: TradeSignal,
        bet_size: float,
    ) -> str:
        """Place order on Polymarket for this signal."""
        if self.config.trading.dry_run:
            shares = bet_size / signal.token_price if signal.token_price > 0 else 0
            logger.info(
                "[DRY RUN] %s %s $%.2f (%.1f shares @ $%.3f)",
                signal.strategy_name,
                signal.direction,
                bet_size,
                shares,
                signal.token_price,
            )
            return "dry_run"

        # Discover market
        market = await self.discovery.get_market(window_ts)
        if market is None:
            logger.warning("Market not found for window %d", window_ts)
            return "failed"

        token_id = market.get(
            "up_token" if signal.direction == "UP" else "down_token"
        )
        condition_id = market.get("condition_id", "")
        neg_risk = market.get("neg_risk", False)
        tick_size = float(market.get("tick_size", 0.01))

        if not token_id:
            logger.warning("No token_id for %s direction", signal.direction)
            return "failed"

        shares = bet_size / signal.token_price if signal.token_price > 0 else 0

        request = TradeRequest(
            condition_id=condition_id,
            token_id=token_id,
            side="BUY",
            price=min(signal.token_price + 0.02, 0.99),
            size=shares,
            order_type="FOK",
            edge=signal.ev_edge or 0.0,
            kelly_fraction=0.0,
            neg_risk=neg_risk,
            tick_size=tick_size,
        )

        order = await self.order_mgr.place_order(request)
        return order.status

    async def _check_resolution(
        self,
        signal: TradeSignal,
        window_ts: int,
        window_open: float,
        bet_size: float,
    ) -> None:
        """Check actual BTC outcome after window closes."""
        candles = await self.feed_mgr.get_candles(limit=10)
        close_time_ms = (window_ts + WINDOW_SECONDS) * 1000

        btc_close = None
        for c in candles:
            if c.open_time <= close_time_ms <= c.close_time + 60000:
                btc_close = c.close
                break

        if btc_close is None:
            btc_close = self.feed_mgr.get_latest_price() or window_open

        actual = "UP" if btc_close >= window_open else "DOWN"
        won = signal.direction == actual
        pnl = calculate_pnl(bet_size, signal.token_price, won=won)

        self._bankroll += pnl
        if won:
            self._win_count += 1
        else:
            self._loss_count += 1

        total = self._win_count + self._loss_count
        win_rate = self._win_count / total * 100 if total > 0 else 0

        logger.info(
            "RESULT: %s predicted=%s actual=%s %s $%.2f | "
            "bankroll=$%.2f W/L=%d/%d (%.1f%%)",
            signal.strategy_name,
            signal.direction,
            actual,
            "WIN" if won else "LOSS",
            pnl,
            self._bankroll,
            self._win_count,
            self._loss_count,
            win_rate,
        )

        # Update stats snapshot for dashboard
        self._write_stats_snapshot()

        # Log to state store with full resolution details
        await self.state_store.log_trade(
            order_id=f"sniper-{window_ts}-{signal.strategy_name}",
            condition_id=f"btc-5m-{window_ts}",
            token_id="",
            side=f"BUY_{signal.direction}",
            price=signal.token_price,
            size=bet_size / signal.token_price if signal.token_price > 0 else 0,
            status="dry_run" if self.config.trading.dry_run else "filled",
            edge=signal.ev_edge or 0.0,
            kelly_fraction=0.0,
            p_hat=signal.confidence,
            b_estimate=0.0,
            placed_at=time.time(),
            confidence=signal.confidence,
            market_question=f"BTC 5-min {signal.direction} ({signal.strategy_name})",
            btc_open=window_open,
            btc_close=btc_close,
            outcome="WIN" if won else "LOSS",
            pnl_usd=pnl,
            exit_price=1.0 if won else 0.0,
        )

    @property
    def stats(self) -> dict:
        total = self._win_count + self._loss_count
        return {
            "trades": total,
            "wins": self._win_count,
            "losses": self._loss_count,
            "win_rate": self._win_count / total if total > 0 else 0,
            "bankroll": round(self._bankroll, 2),
            "starting_bankroll": round(self._original_bankroll, 2),
            "pnl": round(self._bankroll - self._original_bankroll, 2),
        }

    async def _periodic_stats_refresh(self) -> None:
        """Refresh stats snapshot every 15s so the dashboard sees live BTC price."""
        while not self._shutdown_event.is_set():
            try:
                self._write_stats_snapshot()
            except Exception:
                pass
            try:
                await asyncio.wait_for(self._shutdown_event.wait(), timeout=15.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    def _write_stats_snapshot(self) -> None:
        """Persist runtime stats to JSON for the dashboard."""
        try:
            self._stats_path.parent.mkdir(parents=True, exist_ok=True)
            snapshot = self.stats
            snapshot["btc_price"] = self.feed_mgr.get_latest_price()
            snapshot["dry_run"] = self.config.trading.dry_run
            snapshot["updated_at"] = time.time()
            self._stats_path.write_text(json.dumps(snapshot, indent=2))
        except Exception:
            logger.debug("Failed to write stats snapshot", exc_info=True)
