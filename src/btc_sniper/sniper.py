"""BTC 5-minute up/down sniper engine.

Handles:
- Clock-based window timing (next 5-min boundary)
- Market discovery via deterministic slug
- TA polling loop at T-10s
- Order execution (FOK primary, GTC $0.95 fallback)
- Dry run simulation with delta-based token pricing
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import aiohttp

from src.btc_sniper.binance_feed import BinanceFeed
from src.btc_sniper.ta_strategy import TAResult, analyze
from src.config import BTCSniperConfig

logger = logging.getLogger(__name__)

GAMMA_API = "https://gamma-api.polymarket.com"


@dataclass
class SniperTradeResult:
    window_ts: int
    direction: str  # "UP" or "DOWN"
    confidence: float
    score: float
    token_price: float  # what we paid (or simulated)
    bet_size_usd: float
    order_status: str  # "matched", "dry_run", "failed", "skipped"
    pnl: float = 0.0  # filled in after resolution
    actual_outcome: str = ""  # "UP" or "DOWN" after resolution
    window_delta_pct: float = 0.0
    components: dict = field(default_factory=dict)


def _next_window_ts() -> int:
    """Calculate the current 5-minute window start timestamp."""
    now = int(time.time())
    return now - (now % 300)


def _estimate_token_price(delta_pct: float) -> float:
    """Estimate token price based on how decisive the window delta is.

    Larger move = market has priced it in = token costs more.
    This prevents unrealistic backtests at fixed $0.50.
    """
    abs_delta = abs(delta_pct)
    if abs_delta < 0.005:
        return 0.50
    elif abs_delta < 0.02:
        return 0.55
    elif abs_delta < 0.05:
        return 0.65
    elif abs_delta < 0.10:
        return 0.80
    elif abs_delta < 0.15:
        return 0.92
    else:
        return min(0.97, 0.92 + (abs_delta - 0.15) * 0.5)


class BTCSniper:
    """Main sniper engine for BTC 5-minute binary markets."""

    def __init__(
        self,
        config: BTCSniperConfig,
        binance_feed: BinanceFeed,
        order_mgr,
        positions,
        state_store,
        dry_run: bool = True,
    ):
        self.cfg = config
        self.feed = binance_feed
        self.order_mgr = order_mgr
        self.positions = positions
        self.state_store = state_store
        self.dry_run = dry_run
        self._bankroll = config.starting_bankroll
        self._original_bankroll = config.starting_bankroll
        self._session: Optional[aiohttp.ClientSession] = None
        self._running = False
        self._trade_count = 0
        self._win_count = 0
        self._loss_count = 0

    async def start(self) -> None:
        self._session = aiohttp.ClientSession()
        self._running = True

    async def stop(self) -> None:
        self._running = False
        if self._session:
            await self._session.close()

    async def run_single_window(self) -> Optional[SniperTradeResult]:
        """Wait for the next window, run TA at T-10s, place trade, check result."""
        window_ts = _next_window_ts()
        close_time = window_ts + 300

        # Wait until T-10 seconds before close
        snipe_time = close_time - self.cfg.entry_seconds_before_close
        now = time.time()
        wait_secs = snipe_time - now

        if wait_secs > 0:
            logger.info(
                "BTC Sniper: waiting %.0fs for window %d (closes at %d)",
                wait_secs, window_ts, close_time,
            )
            await asyncio.sleep(wait_secs)

        if not self._running:
            return None

        # Get the window open price from Binance candle data
        window_open = await self._get_window_open_price(window_ts)
        if window_open is None:
            logger.warning("BTC Sniper: could not determine window open price, skipping")
            return SniperTradeResult(
                window_ts=window_ts, direction="", confidence=0,
                score=0, token_price=0, bet_size_usd=0,
                order_status="skipped",
            )

        # TA polling loop: check every 2s, fire when confident or at T-5s deadline
        best_result: Optional[TAResult] = None
        prev_score: Optional[float] = None
        fired = False
        hard_deadline = close_time - 5  # T-5s: must fire by here

        while time.time() < hard_deadline and self._running:
            current_price = self.feed.get_latest_price()
            if current_price is None:
                await asyncio.sleep(1)
                continue

            candles = await self.feed.get_candles(limit=30)
            tick_prices = self.feed.get_tick_prices(since=window_ts)

            result = analyze(candles, window_open, current_price, tick_prices)

            # Track best signal
            if best_result is None or abs(result.score) > abs(best_result.score):
                best_result = result

            # Spike detection: score jumped >= 1.5 since last check
            if prev_score is not None and abs(result.score - prev_score) >= 1.5:
                logger.info(
                    "BTC Sniper: SPIKE detected (%.2f -> %.2f), firing immediately",
                    prev_score, result.score,
                )
                best_result = result
                fired = True
                break

            # Confidence threshold met
            if result.confidence >= self.cfg.min_confidence:
                best_result = result
                fired = True
                break

            prev_score = result.score
            await asyncio.sleep(2)

        if best_result is None:
            return SniperTradeResult(
                window_ts=window_ts, direction="", confidence=0,
                score=0, token_price=0, bet_size_usd=0,
                order_status="skipped",
            )

        # Calculate bet size based on mode
        bet_size = self._calculate_bet_size()
        if bet_size < self.cfg.min_bet_usd:
            logger.info("BTC Sniper: bankroll too low ($%.2f), skipping", self._bankroll)
            return SniperTradeResult(
                window_ts=window_ts, direction=best_result.direction,
                confidence=best_result.confidence, score=best_result.score,
                token_price=0, bet_size_usd=0, order_status="skipped",
            )

        # Estimate token price for dry run / log
        token_price = _estimate_token_price(best_result.window_delta_pct)

        # Execute trade
        if self.dry_run:
            order_status = "dry_run"
            shares = bet_size / token_price if token_price > 0 else 0
            logger.info(
                "BTC Sniper [DRY RUN]: %s conf=%.1f%% score=%.2f price=$%.3f "
                "bet=$%.2f shares=%.1f delta=%.4f%%",
                best_result.direction, best_result.confidence * 100,
                best_result.score, token_price, bet_size, shares,
                best_result.window_delta_pct,
            )
        else:
            order_status = await self._place_polymarket_order(
                window_ts, best_result.direction, bet_size, token_price,
            )

        self._trade_count += 1

        trade_result = SniperTradeResult(
            window_ts=window_ts,
            direction=best_result.direction,
            confidence=best_result.confidence,
            score=best_result.score,
            token_price=token_price,
            bet_size_usd=bet_size,
            order_status=order_status,
            window_delta_pct=best_result.window_delta_pct,
            components=best_result.components,
        )

        # Wait for resolution and check outcome
        await self._wait_for_resolution(close_time)
        await self._check_resolution(trade_result, window_ts, window_open)

        return trade_result

    def _calculate_bet_size(self) -> float:
        """Calculate bet size based on trading mode."""
        mode = self.cfg.mode
        if mode == "safe":
            return min(self._bankroll * 0.25, self._bankroll)
        elif mode == "aggressive":
            # Risk only profits above original bankroll
            profit = max(0, self._bankroll - self._original_bankroll)
            if profit > 0:
                return profit
            return min(self._bankroll * 0.25, self._bankroll)
        elif mode == "degen":
            return self._bankroll
        return min(self._bankroll * 0.25, self._bankroll)

    async def _get_window_open_price(self, window_ts: int) -> Optional[float]:
        """Get BTC price at window open from Binance 1-min candle."""
        candles = await self.feed.get_candles(limit=10)
        window_open_ms = window_ts * 1000
        for c in candles:
            if c.open_time <= window_open_ms <= c.close_time:
                return c.open
        # Fallback: use the candle closest to window_ts
        if candles:
            return candles[-5].open if len(candles) > 5 else candles[0].open
        return None

    async def _place_polymarket_order(
        self,
        window_ts: int,
        direction: str,
        bet_size: float,
        estimated_price: float,
    ) -> str:
        """Discover the Polymarket market and place order."""
        slug = f"btc-updown-5m-{window_ts}"
        market = await self._discover_market(slug)
        if market is None:
            logger.warning("BTC Sniper: market %s not found", slug)
            return "failed"

        # Find the right token (Up or Down)
        token_id = None
        condition_id = market.get("condition_id", "")
        neg_risk = market.get("neg_risk", False)
        tick_size = float(market.get("minimum_tick_size", 0.01))

        for outcome in market.get("tokens", []):
            outcome_name = outcome.get("outcome", "").upper()
            if (direction == "UP" and outcome_name in ("UP", "YES")) or \
               (direction == "DOWN" and outcome_name in ("DOWN", "NO")):
                token_id = outcome.get("token_id")
                break

        if not token_id:
            logger.warning("BTC Sniper: could not find %s token for %s", direction, slug)
            return "failed"

        from src.execution.order_manager import TradeRequest

        # Primary: FOK at market
        request = TradeRequest(
            condition_id=condition_id,
            token_id=token_id,
            side="BUY",
            price=min(estimated_price + 0.02, 0.99),  # slight premium for fill
            size=bet_size / estimated_price if estimated_price > 0 else 0,
            order_type="FOK",
            edge=0.0,
            kelly_fraction=0.0,
            neg_risk=neg_risk,
            tick_size=tick_size,
        )

        order = await self.order_mgr.place_order(request)
        if order.status in ("matched", "filled"):
            return "matched"

        # Fallback: GTC limit at $0.95
        logger.info("BTC Sniper: FOK failed, trying GTC at $0.95")
        min_shares = max(5.0, bet_size / 0.95)
        fallback = TradeRequest(
            condition_id=condition_id,
            token_id=token_id,
            side="BUY",
            price=0.95,
            size=min_shares,
            order_type="GTC",
            edge=0.0,
            kelly_fraction=0.0,
            neg_risk=neg_risk,
            tick_size=tick_size,
        )
        order2 = await self.order_mgr.place_order(fallback)
        return order2.status if order2 else "failed"

    async def _discover_market(self, slug: str) -> Optional[dict]:
        """Find market by slug via Gamma API."""
        if not self._session:
            return None
        try:
            url = f"{GAMMA_API}/events?slug={slug}"
            async with self._session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                if data and len(data) > 0:
                    event = data[0]
                    markets = event.get("markets", [])
                    return markets[0] if markets else None
        except Exception:
            logger.debug("Failed to discover market %s", slug, exc_info=True)
        return None

    async def _wait_for_resolution(self, close_time: float) -> None:
        """Wait until the 5-min window has closed."""
        remaining = close_time - time.time() + 2  # +2s buffer
        if remaining > 0:
            await asyncio.sleep(remaining)

    async def _check_resolution(
        self, trade: SniperTradeResult, window_ts: int, window_open: float,
    ) -> None:
        """Check actual BTC outcome after window closes."""
        candles = await self.feed.get_candles(limit=10)
        close_time_ms = (window_ts + 300) * 1000

        btc_close = None
        for c in candles:
            if c.open_time <= close_time_ms <= c.close_time + 60000:
                btc_close = c.close
                break

        if btc_close is None:
            # Use latest price as approximation
            btc_close = self.feed.get_latest_price() or window_open

        actual = "UP" if btc_close >= window_open else "DOWN"
        trade.actual_outcome = actual

        won = trade.direction == actual
        if won:
            profit = trade.bet_size_usd * (1.0 / trade.token_price - 1.0) if trade.token_price > 0 else 0
            trade.pnl = round(profit, 2)
            self._bankroll += profit
            self._win_count += 1
        else:
            trade.pnl = -trade.bet_size_usd
            self._bankroll -= trade.bet_size_usd
            self._loss_count += 1

        total = self._win_count + self._loss_count
        win_rate = self._win_count / total * 100 if total > 0 else 0

        logger.info(
            "BTC Sniper: window %d %s (predicted %s) — %s $%.2f | "
            "bankroll=$%.2f W/L=%d/%d (%.1f%%)",
            window_ts, actual, trade.direction,
            "WIN" if won else "LOSS", trade.pnl,
            self._bankroll, self._win_count, self._loss_count, win_rate,
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
            "pnl": round(self._bankroll - self._original_bankroll, 2),
        }
