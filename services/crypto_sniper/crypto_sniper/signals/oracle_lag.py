"""Oracle lag detection — finds windows where Chainlink confirms direction
but Polymarket hasn't repriced yet.

The edge is information latency, not prediction.
Traders on Polymarket take ~55s average to reprice after Chainlink updates.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Dynamic fee estimate (conservative)
BASE_FEE_PCT = 0.01  # 1% baseline taker fee
VOLATILITY_SURCHARGE = 0.01  # Additional 1% during high-vol windows
TOTAL_FEE_EST = BASE_FEE_PCT + VOLATILITY_SURCHARGE  # 2% round-trip
MIN_PROFIT_MARGIN = 0.03  # Minimum profit after fees
MIN_NET_EDGE = TOTAL_FEE_EST + MIN_PROFIT_MARGIN  # 5% total minimum

# Window parameters
MIN_SECONDS_REMAINING = 8
MAX_SECONDS_REMAINING = 55
CONFIRMATION_THRESHOLD = 0.003  # 0.3% delta minimum


@dataclass
class LagOpportunity:
    """Result of oracle lag evaluation."""
    trade: bool
    direction: Optional[str]  # "UP" or "DOWN"
    edge: float
    lag: float
    lag_score: float
    theoretical_fair: float
    market_price: float
    chainlink_price: float
    delta_pct: float
    velocity: float
    velocity_confirms: bool
    reason: str


def evaluate_lag_opportunity(
    chainlink_price: float,
    chainlink_ts: float,
    strike: float,
    market_up_price: float,
    market_down_price: float,
    seconds_remaining: float,
    velocity: float = 0.0,
    min_seconds: int = MIN_SECONDS_REMAINING,
    max_seconds: int = MAX_SECONDS_REMAINING,
    confirmation_threshold: float = CONFIRMATION_THRESHOLD,
) -> LagOpportunity:
    """Core oracle lag detection logic.

    Returns a LagOpportunity indicating whether to trade and with what confidence.
    """
    no_trade = lambda reason: LagOpportunity(
        trade=False, direction=None, edge=0.0, lag=0.0, lag_score=0.0,
        theoretical_fair=0.0, market_price=0.0, chainlink_price=chainlink_price,
        delta_pct=0.0, velocity=velocity, velocity_confirms=False, reason=reason,
    )

    if chainlink_price <= 0:
        return no_trade("No Chainlink price available")

    if seconds_remaining < min_seconds:
        return no_trade(f"Too close to close ({seconds_remaining:.1f}s)")
    if seconds_remaining > max_seconds:
        return no_trade(f"Too early ({seconds_remaining:.1f}s remaining)")

    # Data freshness
    data_age = time.time() - chainlink_ts
    if data_age > 30:
        return no_trade(f"Chainlink data stale ({data_age:.0f}s old)")

    # Direction confirmation
    delta = (chainlink_price - strike) / strike
    abs_delta = abs(delta)

    if abs_delta < confirmation_threshold:
        return no_trade(
            f"Delta {abs_delta:.4%} below threshold {confirmation_threshold:.4%}"
        )

    cl_direction = "UP" if delta > 0 else "DOWN"

    # Velocity confirmation
    velocity_confirms = (
        (velocity > 0 and cl_direction == "UP")
        or (velocity < 0 and cl_direction == "DOWN")
    )

    # Market lag detection
    if cl_direction == "UP":
        market_price = market_up_price
    else:
        market_price = market_down_price

    theoretical_fair = min(0.97, 0.50 + abs_delta * 150)
    lag = theoretical_fair - market_price

    # Net edge after fees
    net_edge = lag - TOTAL_FEE_EST

    if net_edge < MIN_PROFIT_MARGIN:
        return no_trade(
            f"Net edge {net_edge:.3f} below minimum {MIN_PROFIT_MARGIN:.3f} "
            f"after {TOTAL_FEE_EST:.0%} fees"
        )

    # Lag confidence score (0-1)
    lag_score = min(1.0, (
        min(abs_delta / 0.01, 1.0) * 0.40
        + min(lag / 0.15, 1.0) * 0.35
        + (0.25 if velocity_confirms else 0.0)
    ))

    logger.info(
        "LAG DETECTED: %s | CL: $%.2f vs strike $%.2f (%.4f%%) | "
        "Market: %.3f vs fair: %.3f | Net edge: %+.3f | Score: %.2f | %.1fs left",
        cl_direction,
        chainlink_price,
        strike,
        delta * 100,
        market_price,
        theoretical_fair,
        net_edge,
        lag_score,
        seconds_remaining,
    )

    return LagOpportunity(
        trade=True,
        direction=cl_direction,
        edge=round(net_edge, 4),
        lag=round(lag, 4),
        lag_score=round(lag_score, 3),
        theoretical_fair=round(theoretical_fair, 4),
        market_price=market_price,
        chainlink_price=chainlink_price,
        delta_pct=round(delta * 100, 4),
        velocity=round(velocity, 2),
        velocity_confirms=velocity_confirms,
        reason=f"Oracle lag confirmed: {abs_delta:.3%} delta, {lag:.3f} market lag",
    )
