"""
Delta-based token price model.
Simulates what you'd actually pay on Polymarket given the BTC move from window open.
Based on observed live trading data (piecewise linear model from Archetapp).

In dry-run mode this prevents unrealistically optimistic backtests
(fixed $0.50 tokens = fake 2x on every win -- not real).
"""


def estimate_token_price(delta_pct: float) -> float:
    """
    Given delta_pct = (current_btc - window_open) / window_open * 100,
    return estimated cost of the winning-direction token.

    delta_pct should always be passed as abs() value --
    direction is handled by the strategy layer.

    Examples:
        delta_pct = 0.003 -> $0.50  (coin flip)
        delta_pct = 0.02  -> $0.55  (slight lean)
        delta_pct = 0.05  -> $0.65  (moderate)
        delta_pct = 0.10  -> $0.80  (strong)
        delta_pct = 0.15  -> $0.92  (nearly certain)
    """
    d = abs(delta_pct)

    if d < 0.005:
        return 0.50
    elif d < 0.02:
        # Linear: 0.50 -> 0.55
        t = (d - 0.005) / (0.02 - 0.005)
        return 0.50 + t * 0.05
    elif d < 0.05:
        # Linear: 0.55 -> 0.65
        t = (d - 0.02) / (0.05 - 0.02)
        return 0.55 + t * 0.10
    elif d < 0.10:
        # Linear: 0.65 -> 0.80
        t = (d - 0.05) / (0.10 - 0.05)
        return 0.65 + t * 0.15
    elif d < 0.15:
        # Linear: 0.80 -> 0.92
        t = (d - 0.10) / (0.15 - 0.10)
        return 0.80 + t * 0.12
    else:
        # Cap at 0.97
        return min(0.92 + (d - 0.15) * 0.5, 0.97)


def expected_payout(token_price: float) -> float:
    """Payout per dollar risked if correct (token resolves to $1.00)."""
    if token_price <= 0 or token_price >= 1.0:
        return 0.0
    return (1.0 - token_price) / token_price
