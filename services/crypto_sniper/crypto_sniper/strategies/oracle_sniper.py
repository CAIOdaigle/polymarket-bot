"""Oracle lag sniper strategy.

Monitors Chainlink BTC/USD feed for confirmed direction changes,
then trades on Polymarket before the market reprices.

Edge: information latency (~55s average), not prediction.
"""

from __future__ import annotations

import logging
from typing import Optional

from crypto_sniper.config import OracleSniperConfig
from crypto_sniper.feeds.feed_manager import FeedManager
from crypto_sniper.signals.oracle_lag import evaluate_lag_opportunity
from crypto_sniper.strategies.base import BaseStrategy, TradeSignal

logger = logging.getLogger(__name__)


class OracleSniperStrategy(BaseStrategy):
    """Chainlink oracle lag arbitrage strategy."""

    def __init__(self, config: OracleSniperConfig):
        self._config = config
        self._feed_mgr: Optional[FeedManager] = None
        self._chainlink_feed = None

    async def initialize(self, feed_manager: FeedManager) -> None:
        self._feed_mgr = feed_manager
        self._chainlink_feed = await feed_manager.get_chainlink_feed()
        logger.info(
            "OracleSniperStrategy initialized (min_lag=%.2f, entry=%d-%ds)",
            self._config.min_lag_score,
            self._config.entry_window_max_seconds,
            self._config.entry_window_min_seconds,
        )

    async def evaluate(
        self, window_ts: int, seconds_remaining: float
    ) -> Optional[TradeSignal]:
        if self._chainlink_feed is None:
            return None

        # Get strike price for this window
        try:
            strike = await self._chainlink_feed.get_strike(window_ts)
        except Exception:
            cl_price, _ = self._chainlink_feed.get_chainlink_price()
            strike = cl_price
            if strike <= 0:
                return None

        # Get current Chainlink price
        cl_price, cl_ts = self._chainlink_feed.get_chainlink_price()
        if cl_price <= 0:
            return None

        # Get market prices (from market discovery, default to 0.50)
        from crypto_sniper.market.discovery import MarketDiscovery
        discovery = MarketDiscovery()
        market = await discovery.get_market(window_ts)
        up_price = market["up_price"] if market else 0.50
        down_price = market["down_price"] if market else 0.50

        # Get price velocity
        velocity = self._chainlink_feed.get_price_velocity(10)

        # Evaluate oracle lag
        opportunity = evaluate_lag_opportunity(
            chainlink_price=cl_price,
            chainlink_ts=cl_ts,
            strike=strike,
            market_up_price=up_price,
            market_down_price=down_price,
            seconds_remaining=seconds_remaining,
            velocity=velocity,
            min_seconds=self._config.entry_window_min_seconds,
            max_seconds=self._config.entry_window_max_seconds,
            confirmation_threshold=self._config.confirmation_threshold,
        )

        if not opportunity.trade:
            return None

        if opportunity.lag_score < self._config.min_lag_score:
            return None

        # Oracle confidence: high because this is confirmed, not predicted
        oracle_confidence = min(0.95, 0.60 + opportunity.lag_score * 0.35)

        return TradeSignal(
            direction=opportunity.direction,
            confidence=oracle_confidence,
            score=opportunity.lag_score,
            strategy_name=self.name,
            window_ts=window_ts,
            ev_edge=opportunity.edge,
            token_price=opportunity.market_price,
            components={
                "lag_score": opportunity.lag_score,
                "lag": opportunity.lag,
                "edge": opportunity.edge,
                "delta_pct": opportunity.delta_pct,
                "velocity": opportunity.velocity,
                "velocity_confirms": opportunity.velocity_confirms,
                "theoretical_fair": opportunity.theoretical_fair,
            },
        )

    async def shutdown(self) -> None:
        pass

    @property
    def name(self) -> str:
        return "oracle_sniper"

    @property
    def eval_interval_seconds(self) -> float:
        return self._config.poll_interval_ms / 1000

    @property
    def entry_window_seconds(self) -> tuple[int, int]:
        return (
            self._config.entry_window_max_seconds,
            self._config.entry_window_min_seconds,
        )
