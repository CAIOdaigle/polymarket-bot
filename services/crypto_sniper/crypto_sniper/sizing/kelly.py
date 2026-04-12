"""Token-price-aware fractional Kelly criterion for bet sizing.

Kelly formula: f = (bp - q) / b
  b = odds (payout per dollar wagered)
  p = probability of winning (model_prob)
  q = probability of losing (1 - p)

Fractional Kelly: f_actual = f * kelly_fraction
Quarter Kelly (0.25) is standard for volatile markets.

Ported from downloaded strategies/kelly.py with config-driven mode caps.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from crypto_sniper.config import KellyConfig


def kelly_bet(
    bankroll: float,
    model_prob: float,
    token_price: float,
    confidence: float,
    mode: str = "safe",
    kelly_config: "KellyConfig | None" = None,
) -> float:
    """Calculate bet size in USDC using fractional Kelly.

    Returns bet_size, capped at mode-specific maximums.
    Returns 0.0 if no edge or below minimums.
    """
    # Config defaults
    kelly_fraction = 0.25
    min_bet = 4.75

    # Mode-specific caps
    max_bet_fractions = {
        "safe": 0.01,
        "aggressive": 0.02,
        "degen": 0.03,
        "oracle": 0.02,
    }
    max_token_prices = {
        "safe": 0.62,
        "aggressive": 0.70,
        "degen": 0.80,
        "oracle": 0.80,
    }
    confidence_floors = {
        "safe": 0.30,
        "aggressive": 0.20,
        "degen": 0.00,
        "oracle": 0.00,
    }

    if kelly_config is not None:
        kelly_fraction = kelly_config.fraction
        min_bet = kelly_config.min_bet_usd
        max_bet_fractions = {
            "safe": kelly_config.max_bet_fraction_safe,
            "aggressive": kelly_config.max_bet_fraction_aggressive,
            "degen": kelly_config.max_bet_fraction_degen,
            "oracle": kelly_config.max_bet_fraction_oracle,
        }
        max_token_prices = {
            "safe": kelly_config.max_token_price_safe,
            "aggressive": kelly_config.max_token_price_aggressive,
            "degen": kelly_config.max_token_price_degen,
            "oracle": kelly_config.max_token_price_oracle,
        }

    # Confidence floor check
    conf_floor = confidence_floors.get(mode, 0.30)
    if confidence < conf_floor:
        return 0.0

    # Token price ceiling — skip if too expensive for our win rate
    max_token = max_token_prices.get(mode, 0.65)
    if token_price > max_token:
        return 0.0

    # Payout odds: b = (1 - token_price) / token_price
    if token_price <= 0 or token_price >= 1.0:
        return 0.0
    b = (1.0 - token_price) / token_price

    # Kelly fraction: f = (bp - q) / b
    q = 1.0 - model_prob
    f = (b * model_prob - q) / b

    if f <= 0:
        return 0.0  # No edge

    # Apply fractional Kelly
    f_actual = f * kelly_fraction

    # Mode-specific bet size cap
    max_fraction = max_bet_fractions.get(mode, 0.01)
    fraction = min(f_actual, max_fraction)
    bet = bankroll * fraction

    # Enforce minimums and maximums
    if bet < min_bet:
        return 0.0
    bet = min(bet, bankroll)

    return round(bet, 2)


def calculate_pnl(bet_size: float, token_price: float, won: bool) -> float:
    """Calculate PnL for a resolved trade.

    Token resolves to $1.00 if correct, $0.00 if not.
    """
    if token_price <= 0:
        return 0.0
    shares = bet_size / token_price
    if won:
        return round(shares * (1.0 - token_price), 4)
    else:
        return round(-bet_size, 4)
