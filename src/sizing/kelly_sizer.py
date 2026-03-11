"""
Half-Kelly Position Sizing.

Implements the Kelly criterion for binary prediction markets:
  f* = (p_hat - p) / (1 - p)
  Half-Kelly = 0.5 * f*

From the documents:
  EV = p_hat * (1 - p) - (1 - p_hat) * p = p_hat - p  (Eq. 4)
  "NEVER full Kelly on 5min markets!" — hence half-Kelly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.config import TradingConfig

logger = logging.getLogger(__name__)


@dataclass
class SizingResult:
    should_trade: bool
    side: str  # BUY_YES, BUY_NO, SELL_YES, SELL_NO, HOLD
    edge: float
    kelly_fraction: float
    half_kelly_fraction: float
    position_size_usd: float
    position_size_shares: float
    confidence: float
    reason: str


class KellySizer:
    def __init__(self, config: TradingConfig):
        self.kelly_mult = config.kelly_fraction  # 0.5 = half-Kelly
        self.min_edge = config.min_edge_threshold
        self.max_position = config.max_position_usd
        self.bankroll = config.total_bankroll_usd
        self.max_exposure = config.max_portfolio_exposure

    def compute(
        self,
        p_hat: float,
        market_price_yes: float,
        market_price_no: float,
        current_position_usd: float,
        total_deployed_usd: float,
        lmsr_confidence: float,
        signal_count: int,
    ) -> SizingResult:
        edge_yes = p_hat - market_price_yes
        edge_no = (1 - p_hat) - market_price_no

        # Confidence-adjusted edge: raw edge is meaningless without confidence
        adj_edge_yes = edge_yes * lmsr_confidence
        adj_edge_no = edge_no * lmsr_confidence

        # Pick the side with the larger confidence-adjusted edge
        if adj_edge_yes >= adj_edge_no and adj_edge_yes > self.min_edge:
            side = "BUY_YES"
            edge = adj_edge_yes
            market_price = market_price_yes
            p_est = p_hat
        elif adj_edge_no > self.min_edge:
            side = "BUY_NO"
            edge = adj_edge_no
            market_price = market_price_no
            p_est = 1 - p_hat
        else:
            return SizingResult(
                should_trade=False,
                side="HOLD",
                edge=max(adj_edge_yes, adj_edge_no),
                kelly_fraction=0,
                half_kelly_fraction=0,
                position_size_usd=0,
                position_size_shares=0,
                confidence=lmsr_confidence,
                reason=f"Adj edge {max(adj_edge_yes, adj_edge_no):.4f} below threshold {self.min_edge}",
            )

        if market_price >= 0.99:
            return SizingResult(
                should_trade=False,
                side="HOLD",
                edge=edge,
                kelly_fraction=0,
                half_kelly_fraction=0,
                position_size_usd=0,
                position_size_shares=0,
                confidence=0,
                reason="Market price too close to 1.0",
            )

        # Kelly fraction: f* = (p_hat - p) / (1 - p)
        kelly_f = (p_est - market_price) / (1 - market_price)
        half_kelly_f = self.kelly_mult * kelly_f

        # Scale by LMSR confidence
        adjusted_f = half_kelly_f * lmsr_confidence

        # Available capital
        available_bankroll = self.bankroll - total_deployed_usd
        max_from_exposure = self.bankroll * self.max_exposure - total_deployed_usd
        available = min(available_bankroll, max_from_exposure)

        position_usd = min(
            adjusted_f * self.bankroll,
            self.max_position - current_position_usd,
            max(0, available),
        )

        if position_usd <= 0.10:  # Minimum order ~$0.10
            return SizingResult(
                should_trade=False,
                side=side,
                edge=edge,
                kelly_fraction=kelly_f,
                half_kelly_fraction=half_kelly_f,
                position_size_usd=0,
                position_size_shares=0,
                confidence=lmsr_confidence,
                reason="Position limit or bankroll constraint",
            )

        position_shares = position_usd / market_price

        logger.info(
            "Sizing: %s edge=%.4f kelly=%.4f half_kelly=%.4f size=$%.2f shares=%.2f",
            side,
            edge,
            kelly_f,
            half_kelly_f,
            position_usd,
            position_shares,
        )

        return SizingResult(
            should_trade=True,
            side=side,
            edge=edge,
            kelly_fraction=kelly_f,
            half_kelly_fraction=half_kelly_f,
            position_size_usd=round(position_usd, 2),
            position_size_shares=round(position_shares, 2),
            confidence=lmsr_confidence,
            reason=f"Edge={edge:.4f} HalfKelly={half_kelly_f:.4f} Size=${position_usd:.2f}",
        )
