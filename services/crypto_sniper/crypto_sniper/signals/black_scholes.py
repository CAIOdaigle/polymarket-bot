"""Black-Scholes binary option EV calculator.

Estimates true probability that BTC closes above/below window open price.
Compares model probability to Polymarket market price for EV edge.
"""

from __future__ import annotations

import math
from scipy.stats import norm


def ewma_volatility(closes: list[float], span: int = 20) -> float:
    """Exponentially weighted moving average volatility (per-minute)."""
    if len(closes) < 2:
        return 0.01

    returns = []
    for i in range(1, len(closes)):
        if closes[i - 1] > 0:
            returns.append(math.log(closes[i] / closes[i - 1]))

    if not returns:
        return 0.01

    alpha = 2.0 / (span + 1)
    var = returns[0] ** 2
    for r in returns[1:]:
        var = alpha * r**2 + (1 - alpha) * var

    return math.sqrt(var)


def binary_prob(
    current_price: float,
    strike_price: float,
    volatility_per_min: float,
    minutes_remaining: float,
    direction: str,
) -> float:
    """Probability BTC closes above (UP) or below (DOWN) the strike."""
    if minutes_remaining <= 0 or volatility_per_min <= 0 or strike_price <= 0:
        if direction.upper() == "UP":
            return 1.0 if current_price >= strike_price else 0.0
        else:
            return 1.0 if current_price <= strike_price else 0.0

    sigma = volatility_per_min * math.sqrt(minutes_remaining)
    if sigma <= 0:
        sigma = 1e-6

    d2 = math.log(current_price / strike_price) / sigma

    if direction.upper() == "UP":
        prob = norm.cdf(d2)
    else:
        prob = norm.cdf(-d2)

    return max(0.01, min(0.99, prob))


def ev_edge(model_prob: float, market_price: float) -> float:
    """EV edge = model probability - market implied probability."""
    return model_prob - market_price


def should_trade(
    current_price: float,
    window_open_price: float,
    closes: list[float],
    seconds_remaining: float,
    signal_direction: str,
    market_price: float = 0.50,
    min_ev_edge: float = 0.05,
) -> tuple[bool, float, float]:
    """Full EV check. Returns (should_trade, ev_edge, model_prob)."""
    minutes_remaining = max(seconds_remaining / 60, 0.01)
    vol = ewma_volatility(closes)

    model_prob = binary_prob(
        current_price=current_price,
        strike_price=window_open_price,
        volatility_per_min=vol,
        minutes_remaining=minutes_remaining,
        direction=signal_direction,
    )

    edge = ev_edge(model_prob, market_price)
    trade = edge >= min_ev_edge

    return trade, round(edge, 4), round(model_prob, 4)
