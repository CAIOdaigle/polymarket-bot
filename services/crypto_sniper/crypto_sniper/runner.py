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
from crypto_sniper.sizing.calibration import (
    ConfidenceCalibrator, build_calibration_from_db,
)
from crypto_sniper.market.discovery import MarketDiscovery

logger = logging.getLogger(__name__)

WINDOW_SECONDS = 300  # 5-minute windows

# Slippage model: live orders submit at `best_ask + SLIPPAGE_BUMP` to raise
# fill probability. This matches what _execute_trade sends to Polymarket.
# In the new walk-the-book regime, this is the maximum price level we're
# willing to walk to in order to fill the full size.
SLIPPAGE_BUMP = 0.02  # +2c over the quoted best ask

# Token-price floor — refuse trades when the underdog token costs less than
# this. Three reasons: (1) deep-underdog regime is where book depth is
# thinnest and slippage hurts most, (2) calibration data shows we lose
# money in this bucket anyway, (3) the apparent "wins" at $0.03 entries
# are exactly the trades whose dry-run PnL is most overstated.
TOKEN_PRICE_FLOOR = 0.10

# Minimum top-3-level liquidity in USD. If the cheapest 3 ask levels add
# up to less than this much in dollars, we skip — the book is too thin to
# trust the quoted price as something a real order could fill against.
MIN_TOP3_DEPTH_USD = 5.0


def compute_fok_fill(
    asks: list[tuple[float, float]],
    shares_wanted: float,
    limit_price: float,
) -> tuple[float | None, bool]:
    """Simulate a Polymarket FOK BUY by walking the ask book.

    Polymarket FOK orders are all-or-nothing: the entire size must fill at
    or below the limit price, or the order is killed and nothing happens.

    Args:
        asks: List of (price, size) tuples, sorted ascending by price.
        shares_wanted: Total shares we need.
        limit_price: Highest price we're willing to walk to.

    Returns:
        (avg_fill_price, True)  if the full size fills at <= limit
        (None,            False) if the FOK would kill (insufficient depth
                                  at acceptable prices)
    """
    if not asks or shares_wanted <= 0:
        return None, False
    total_cost = 0.0
    total_shares = 0.0
    for price, size in asks:
        if price > limit_price:
            break
        take = min(size, shares_wanted - total_shares)
        if take <= 0:
            break
        total_cost += take * price
        total_shares += take
        if total_shares >= shares_wanted - 1e-9:
            return total_cost / total_shares, True
    return None, False


def top_n_depth_usd(asks: list[tuple[float, float]], n: int = 3) -> float:
    """Sum (price × size) of the top-N ask levels — total fillable USD."""
    return sum(p * s for p, s in asks[:n])


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
        self.discovery = MarketDiscovery(slug_prefix=config.slug_prefix)

        self._strategies: list[BaseStrategy] = []
        self._shutdown_event = asyncio.Event()
        self._bankroll = config.trading.total_bankroll_usd
        self._original_bankroll = self._bankroll
        self._trade_count = 0
        self._win_count = 0
        self._loss_count = 0
        self._traded_windows: set[int] = set()
        self._asset = config.asset.lower()  # e.g. "btc", "eth"
        stats_dir = Path(os.environ.get("SNIPER_DATA_DIR", "/app/data"))
        self._stats_path = stats_dir / f"sniper_stats_{self._asset}.json"

        # Empirical confidence calibrator — maps the TA "confidence" score to
        # an observed win rate. Without this, Kelly treats conf=1.0 as 100%
        # win probability, which our data shows is dramatically miscalibrated.
        calibration_path = stats_dir / "calibration.json"
        self._calibrator = ConfidenceCalibrator(
            calibration_path=calibration_path,
            default_cap=0.55,  # until proven otherwise, no signal exceeds this
        )
        self._calibration_db_path = stats_dir / "bot.db"
        self._write_stats_snapshot()

    def register(self, strategy: BaseStrategy) -> None:
        self._strategies.append(strategy)
        logger.info("Registered strategy: %s", strategy.name)

    async def run(self) -> None:
        """Main loop — runs until shutdown signal."""
        # Rebuild calibration from the latest trade history on startup so
        # restarts don't lose ground. Safe if the DB is empty or sparse.
        try:
            build_calibration_from_db(
                self._calibration_db_path,
                self._stats_path.parent / "calibration.json",
            )
            self._calibrator.reload()
        except Exception:
            logger.warning("Calibration rebuild failed on startup", exc_info=True)

        # Initialize all strategies — pass both feed manager and market discovery
        # so they can fetch REAL Polymarket prices (no estimation fallback).
        for strategy in self._strategies:
            await strategy.initialize(self.feed_mgr, self.discovery)

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

        # Periodic calibration rebuild — recomputes empirical bucket win-rates
        # from the DB every 30 min so the calibrator keeps learning.
        calibration_task = asyncio.create_task(self._periodic_calibration_rebuild())

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
        calibration_task.cancel()
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

        # Calibrate stated confidence -> empirical win probability BEFORE sizing.
        # This is the single most important correction to make dry-run numbers
        # meaningful for live trading.
        calibrated_prob = self._calibrator.calibrate(
            self._asset, best_signal.confidence
        )
        logger.info(
            "Calibration: stated_conf=%.2f -> model_prob=%.2f (asset=%s)",
            best_signal.confidence, calibrated_prob, self._asset.upper(),
        )

        bet_size = kelly_bet(
            bankroll=self._bankroll,
            model_prob=calibrated_prob,
            token_price=best_signal.token_price,
            confidence=calibrated_prob,  # pass calibrated to mode floors too
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

        # ── Execution-realism gates ───────────────────────────────────────
        # 1. Token-price floor: refuse pennies-on-the-dollar underdog bets.
        if best_signal.token_price < TOKEN_PRICE_FLOOR:
            logger.info(
                "Token price floor: %s %s @ $%.3f < $%.3f — skipping",
                best_signal.strategy_name, best_signal.direction,
                best_signal.token_price, TOKEN_PRICE_FLOOR,
            )
            self._traded_windows.add(window_ts)
            return

        # 2. Walk the book: confirm the FOK can actually fill at our limit.
        # Refresh the book here (not the cached strategy quote) because the
        # ask side moves second-to-second.
        asks = await self.discovery.get_live_token_book(
            window_ts, best_signal.direction
        )
        if not asks:
            logger.info(
                "Book empty for %s %s — skipping",
                best_signal.strategy_name, best_signal.direction,
            )
            self._traded_windows.add(window_ts)
            return

        # 3. Depth gate: top-3 levels must offer at least MIN_TOP3_DEPTH_USD.
        depth_usd = top_n_depth_usd(asks, n=3)
        if depth_usd < MIN_TOP3_DEPTH_USD:
            logger.info(
                "Insufficient depth: top-3 = $%.2f < $%.2f — skipping",
                depth_usd, MIN_TOP3_DEPTH_USD,
            )
            self._traded_windows.add(window_ts)
            return

        # 4. FOK simulation: shares_wanted at limit = quote + SLIPPAGE_BUMP.
        # If the book can't support the full fill at that limit, the order
        # would be killed in live trading — skip in dry-run too.
        shares_wanted = bet_size / best_signal.token_price
        limit_price = min(best_signal.token_price + SLIPPAGE_BUMP, 0.99)
        avg_fill, filled = compute_fok_fill(asks, shares_wanted, limit_price)
        if not filled:
            logger.info(
                "FOK would kill: want %.1f shares at limit $%.3f "
                "(top ask $%.3f, depth $%.2f) — skipping",
                shares_wanted, limit_price, asks[0][0], depth_usd,
            )
            self._traded_windows.add(window_ts)
            return

        # Replace the quoted price with the realistic walked fill price.
        # This is what shares + PnL accounting will use throughout.
        logger.info(
            "FOK fill OK: quote $%.3f → walked $%.3f (limit $%.3f, "
            "shares %.1f, depth $%.2f)",
            best_signal.token_price, avg_fill, limit_price,
            shares_wanted, depth_usd,
        )
        best_signal.token_price = avg_fill
        # ──────────────────────────────────────────────────────────────────

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
        """Place order on Polymarket for this signal.

        At entry, signal.token_price is the walked-book average fill price
        from compute_fok_fill() — already reflects realistic slippage.
        Shares + PnL are computed from this effective price, both in
        dry-run and live.
        """
        effective_price = signal.token_price
        if self.config.trading.dry_run:
            shares = bet_size / effective_price if effective_price > 0 else 0
            logger.info(
                "[DRY RUN] %s %s $%.2f (%.1f shares @ $%.3f walked)",
                signal.strategy_name,
                signal.direction,
                bet_size,
                shares,
                effective_price,
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

        # signal.token_price is the walked-book avg fill from
        # compute_fok_fill(); we already verified the FOK can fully fill
        # at limit = quote + SLIPPAGE_BUMP. Submit at that limit so the
        # exchange accepts the same range of price levels we walked.
        shares = bet_size / signal.token_price if signal.token_price > 0 else 0
        limit_price = min(signal.token_price + SLIPPAGE_BUMP, 0.99)

        request = TradeRequest(
            condition_id=condition_id,
            token_id=token_id,
            side="BUY",
            price=limit_price,
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
        """Check actual BTC outcome after window closes.

        Resolution rule: BTC close = the close of the LAST 1-min candle whose
        close_time_ms is at or just before window close. Using `c.close`
        of that candle (NOT the next candle's close) avoids off-by-one.
        """
        candles = await self.feed_mgr.get_candles(limit=10)
        close_time_ms = (window_ts + WINDOW_SECONDS) * 1000

        # Find the candle whose close_time is closest to (but at or before) window close
        btc_close = None
        best = None
        for c in candles:
            if c.close_time <= close_time_ms + 5000:  # allow small drift
                if best is None or c.close_time > best.close_time:
                    best = c
        if best is not None:
            btc_close = best.close

        if btc_close is None:
            # WS latest price as last resort (runner waits ~2s past close time)
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

        # Log to state store with full resolution details.
        # `price` is the REAL best-ask we would pay; `estimated_price` is the
        # old piecewise-linear model kept only for comparison/audit.
        asset_up = self._asset.upper()
        await self.state_store.log_trade(
            order_id=f"sniper-{self._asset}-{window_ts}-{signal.strategy_name}",
            condition_id=f"{self._asset}-5m-{window_ts}",
            token_id="",
            side=f"BUY_{signal.direction}",
            price=signal.token_price,  # REAL Polymarket best-ask
            size=bet_size / signal.token_price if signal.token_price > 0 else 0,
            status="dry_run" if self.config.trading.dry_run else "filled",
            edge=signal.ev_edge or 0.0,
            kelly_fraction=0.0,
            p_hat=signal.confidence,
            b_estimate=0.0,
            placed_at=time.time(),
            confidence=signal.confidence,
            market_question=f"{asset_up} 5-min {signal.direction} ({signal.strategy_name})",
            spot_open=window_open,
            spot_close=btc_close,
            outcome="WIN" if won else "LOSS",
            pnl_usd=pnl,
            exit_price=1.0 if won else 0.0,
            estimated_price=signal.estimated_price,
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

    async def _periodic_calibration_rebuild(self) -> None:
        """Recompute calibration.json every 30 minutes from logged trades."""
        INTERVAL = 1800.0  # 30 min
        while not self._shutdown_event.is_set():
            try:
                await asyncio.wait_for(self._shutdown_event.wait(), timeout=INTERVAL)
            except asyncio.TimeoutError:
                pass
            except asyncio.CancelledError:
                break
            try:
                build_calibration_from_db(
                    self._calibration_db_path,
                    self._stats_path.parent / "calibration.json",
                )
                self._calibrator.reload()
            except Exception:
                logger.debug("Calibration rebuild failed", exc_info=True)

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
            snapshot["asset"] = self._asset.upper()
            snapshot["spot_price"] = self.feed_mgr.get_latest_price()
            # Keep btc_price for backward compatibility with older dashboards
            snapshot["btc_price"] = snapshot["spot_price"]
            snapshot["dry_run"] = self.config.trading.dry_run
            snapshot["updated_at"] = time.time()
            self._stats_path.write_text(json.dumps(snapshot, indent=2))
        except Exception:
            logger.debug("Failed to write stats snapshot", exc_info=True)
