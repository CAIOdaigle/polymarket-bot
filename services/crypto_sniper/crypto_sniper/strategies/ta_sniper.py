"""TA-based BTC 5-min sniper strategy.

Evaluates 7-indicator composite TA score at T-10s before window close.
Optionally gates trades through Black-Scholes EV filter (min 5% edge).
"""

from __future__ import annotations

import logging
from typing import Optional

from crypto_sniper.config import TASniperConfig
from crypto_sniper.feeds.feed_manager import FeedManager
from crypto_sniper.signals.technical import analyze, TAResult
from crypto_sniper.signals.black_scholes import should_trade as ev_should_trade
from crypto_sniper.strategies.base import BaseStrategy, TradeSignal
from polymarket_common.utils.pricing import estimate_token_price

logger = logging.getLogger(__name__)


class TASniperStrategy(BaseStrategy):
    """T-10s technical analysis sniper with optional EV gating."""

    def __init__(self, config: TASniperConfig):
        self._config = config
        self._feed_mgr: Optional[FeedManager] = None
        self._prev_score: Optional[float] = None

    async def initialize(self, feed_manager: FeedManager) -> None:
        self._feed_mgr = feed_manager
        logger.info(
            "TASniperStrategy initialized (mode=%s, min_conf=%.0f%%, ev_gate=%.0f%%)",
            self._config.mode,
            self._config.min_confidence * 100,
            self._config.min_ev_edge * 100,
        )

    async def evaluate(
        self, window_ts: int, seconds_remaining: float
    ) -> Optional[TradeSignal]:
        if self._feed_mgr is None:
            return None

        current_price = self._feed_mgr.get_latest_price()
        if current_price is None:
            return None

        candles = await self._feed_mgr.get_candles(limit=30)
        if len(candles) < 5:
            return None

        # Determine window open price
        window_open = self._get_window_open(candles, window_ts)
        if window_open is None:
            return None

        tick_prices = self._feed_mgr.get_tick_prices(since=window_ts)

        # Run TA composite
        result = analyze(candles, window_open, current_price, tick_prices)

        # Spike detection: score jumped >= 1.5 since last check
        fired_spike = False
        if self._prev_score is not None and abs(result.score - self._prev_score) >= 1.5:
            logger.info(
                "TA SPIKE: %.2f -> %.2f, firing",
                self._prev_score, result.score,
            )
            fired_spike = True
        self._prev_score = result.score

        # Confidence check
        if not fired_spike and result.confidence < self._config.min_confidence:
            return None

        # EV filter gate (Black-Scholes)
        closes = [c.close for c in candles]
        direction_lower = result.direction.lower()
        trade_ok, edge, model_prob = ev_should_trade(
            current_price=current_price,
            window_open_price=window_open,
            closes=closes,
            seconds_remaining=seconds_remaining,
            signal_direction=direction_lower,
            min_ev_edge=self._config.min_ev_edge,
        )

        if not trade_ok:
            logger.debug(
                "EV filter blocked: edge=%.3f < %.3f",
                edge, self._config.min_ev_edge,
            )
            return None

        # Estimate token price
        token_price = estimate_token_price(abs(result.window_delta_pct))

        return TradeSignal(
            direction=result.direction,
            confidence=result.confidence,
            score=result.score,
            strategy_name=self.name,
            window_ts=window_ts,
            ev_edge=edge,
            token_price=token_price,
            components=result.components,
        )

    def _get_window_open(self, candles, window_ts: int) -> Optional[float]:
        window_open_ms = window_ts * 1000
        for c in candles:
            if c.open_time <= window_open_ms <= c.close_time:
                return c.open
        if candles:
            return candles[-5].open if len(candles) > 5 else candles[0].open
        return None

    async def shutdown(self) -> None:
        pass

    @property
    def name(self) -> str:
        return "ta_sniper"

    @property
    def eval_interval_seconds(self) -> float:
        return self._config.eval_interval_seconds

    @property
    def entry_window_seconds(self) -> tuple[int, int]:
        return (self._config.entry_seconds_before_close, 5)
