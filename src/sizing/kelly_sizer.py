"""
kelly_sizer.py — Corrected Kelly position sizing.

Three fixes from panel review:

1. Confidence is a binary gate, not a continuous multiplier.
   Old: kelly_fraction = 0.5 * ((p_hat - price) / (1 - price)) * lmsr_confidence
   New: if confidence < threshold → no trade. Else full half-Kelly on raw edge.
   Reason: multiplying confidence into edge AND using it to scale Kelly output
   applies quadratic penalization. A 0.4-confidence signal got 0.16x sizing.

2. NO position Kelly formula uses correct denominator.
   YES: f* = (p_hat_yes - ask)  / (1 - ask)   denominator = max loss per share
   NO:  f* = (p_hat_no  - ask_no) / (1 - ask_no)
        where p_hat_no = 1 - p_hat_yes, ask_no = best ask for NO token
   Old code used (1 - market_price) for both sides — wrong by factor of
   p_no / (1 - p_no) on NO positions.

3. Kelly fraction scales with time-to-resolution.
   Short-dated markets (< 24h) use quarter-Kelly.
   Medium-dated (24h–72h) use third-Kelly.
   Long-dated (> 72h) use half-Kelly.
   Reason: research doc annotation "NEVER full Kelly on 5min markets" —
   posterior is stale for most of a 5-min recomputation window. Short-dated
   markets have less time for edge to converge, and model error matters more.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class SizingResult:
    """Result of Kelly sizing computation."""
    should_trade: bool
    side: str        # "BUY_YES", "BUY_NO", "HOLD"
    edge: float      # raw edge (p_hat - ask_price)
    kelly_fraction: float
    half_kelly_fraction: float  # the scaled fraction actually applied
    position_size_usd: float
    position_size_shares: float
    confidence: float
    reason: str
    ask_price: float = 0.0
    time_horizon_h: float = 48.0


class KellySizer:
    """
    Sizes positions using Kelly criterion with:
      - Confidence as a binary gate (not a continuous multiplier)
      - Correct YES and NO Kelly formulas
      - Time-horizon-scaled Kelly fraction
      - Ask price (not mid) as the fill price basis
    """

    def __init__(self, config):
        """Accept either KellyConfig or legacy TradingConfig."""
        # Kelly fraction by time horizon
        self.kelly_short = getattr(config, "kelly_short_dated", 0.25)
        self.kelly_medium = getattr(config, "kelly_medium_dated", 0.33)
        self.kelly_long = getattr(config, "kelly_long_dated",
                                   getattr(config, "kelly_fraction", 0.50))
        self.short_hours = getattr(config, "short_dated_hours", 24.0)
        self.medium_hours = getattr(config, "medium_dated_hours", 72.0)

        # Confidence gate
        self.min_confidence = getattr(config, "min_confidence", 0.40)

        # Edge thresholds
        self.min_edge = getattr(config, "min_edge_threshold", 0.05)
        self.min_edge_spread = getattr(config, "min_edge_after_spread", 0.03)

        # Position limits
        self.max_position = getattr(config, "max_position_usd", 10.0)
        self.bankroll = getattr(config, "total_bankroll_usd", 80.0)
        self.max_exposure = getattr(config, "max_portfolio_exposure", 0.50)

    def compute(
        self,
        p_hat: float,
        market_price_yes: float,
        market_price_no: float,
        current_position_usd: float,
        total_deployed_usd: float,
        lmsr_confidence: float,
        signal_count: int,
        book_yes=None,
        book_no=None,
        market=None,
    ) -> SizingResult:
        """
        Backward-compatible sizing entry point.

        If book_yes/book_no are provided, uses ask prices (panel fix).
        Otherwise falls back to market prices (legacy behavior).
        """
        # -- Gate 1: confidence (binary gate, NOT multiplier) --
        if lmsr_confidence < self.min_confidence:
            return SizingResult(
                should_trade=False,
                side="HOLD",
                edge=0.0,
                kelly_fraction=0,
                half_kelly_fraction=0,
                position_size_usd=0,
                position_size_shares=0,
                confidence=lmsr_confidence,
                reason=f"Confidence {lmsr_confidence:.2f} below gate {self.min_confidence}",
            )

        # -- Get ask prices (true fill cost) --
        if book_yes is not None and book_no is not None:
            ask_yes = self._best_ask(book_yes)
            ask_no = self._best_ask(book_no)
            # Fall back to market prices if book is empty
            if ask_yes <= 0:
                ask_yes = market_price_yes
            if ask_no <= 0:
                ask_no = market_price_no
        else:
            ask_yes = market_price_yes
            ask_no = market_price_no

        if ask_yes <= 0 or ask_no <= 0:
            return SizingResult(
                should_trade=False,
                side="HOLD",
                edge=0.0,
                kelly_fraction=0,
                half_kelly_fraction=0,
                position_size_usd=0,
                position_size_shares=0,
                confidence=lmsr_confidence,
                reason="Missing ask prices — cannot size",
            )

        # -- Raw edge (AFTER spread cost) --
        # YES: buy YES at ask_yes. Edge = p_hat - ask_yes.
        # NO:  buy NO at ask_no.  Edge = (1-p_hat) - ask_no.
        p_hat_no = 1.0 - p_hat
        edge_yes = p_hat - ask_yes
        edge_no = p_hat_no - ask_no

        # Pick the side with better edge
        if edge_yes >= edge_no and edge_yes > self.min_edge:
            side = "BUY_YES"
            best_edge = edge_yes
            ask_price = ask_yes
        elif edge_no > self.min_edge:
            side = "BUY_NO"
            best_edge = edge_no
            ask_price = ask_no
        else:
            return SizingResult(
                should_trade=False,
                side="HOLD",
                edge=max(edge_yes, edge_no),
                kelly_fraction=0,
                half_kelly_fraction=0,
                position_size_usd=0,
                position_size_shares=0,
                confidence=lmsr_confidence,
                reason=f"Edge {max(edge_yes, edge_no):.4f} below threshold {self.min_edge}",
            )

        # Secondary spread survival check
        if best_edge < self.min_edge_spread:
            return SizingResult(
                should_trade=False,
                side=side,
                edge=best_edge,
                kelly_fraction=0,
                half_kelly_fraction=0,
                position_size_usd=0,
                position_size_shares=0,
                confidence=lmsr_confidence,
                reason=f"Edge {best_edge:.4f} does not survive spread (min {self.min_edge_spread})",
            )

        if ask_price >= 0.99:
            return SizingResult(
                should_trade=False,
                side="HOLD",
                edge=best_edge,
                kelly_fraction=0,
                half_kelly_fraction=0,
                position_size_usd=0,
                position_size_shares=0,
                confidence=lmsr_confidence,
                reason="Ask price too close to 1.0",
            )

        # -- Time horizon scaling --
        hours_to_resolution = self._hours_to_resolution(market)
        kelly_frac = self._kelly_fraction_for_horizon(hours_to_resolution)

        # -- Kelly formula (correct denominator per side) --
        # f* = edge / (1 - ask_price) — max loss per share
        f_star = best_edge / (1.0 - ask_price)
        f_applied = kelly_frac * f_star

        # -- Size in USD (NO confidence multiplier — already gated above) --
        raw_size_usd = f_applied * self.bankroll

        # Apply limits
        available_bankroll = self.bankroll - total_deployed_usd
        max_from_exposure = self.bankroll * self.max_exposure - total_deployed_usd
        available = min(available_bankroll, max_from_exposure)

        position_usd = min(
            raw_size_usd,
            self.max_position - current_position_usd,
            max(0, available),
        )

        if position_usd <= 0.10:
            return SizingResult(
                should_trade=False,
                side=side,
                edge=best_edge,
                kelly_fraction=f_star,
                half_kelly_fraction=f_applied,
                position_size_usd=0,
                position_size_shares=0,
                confidence=lmsr_confidence,
                reason="Position limit or bankroll constraint",
            )

        position_shares = position_usd / ask_price

        logger.info(
            "[KELLY] %s edge=%.4f ask=%.3f kelly=%.2f (%.0fh horizon) "
            "size=$%.2f shares=%.2f conf=%.2f",
            side, best_edge, ask_price, kelly_frac,
            hours_to_resolution, position_usd, position_shares, lmsr_confidence,
        )

        return SizingResult(
            should_trade=True,
            side=side,
            edge=best_edge,
            kelly_fraction=f_star,
            half_kelly_fraction=f_applied,
            position_size_usd=round(position_usd, 2),
            position_size_shares=round(position_shares, 2),
            confidence=lmsr_confidence,
            reason=f"Edge={best_edge:.4f} Kelly={f_applied:.4f} Size=${position_usd:.2f}",
            ask_price=ask_price,
            time_horizon_h=hours_to_resolution,
        )

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _kelly_fraction_for_horizon(self, hours_to_resolution: float) -> float:
        """Scale Kelly fraction by time horizon."""
        if hours_to_resolution < self.short_hours:
            return self.kelly_short   # 0.25 for < 24h
        if hours_to_resolution < self.medium_hours:
            return self.kelly_medium  # 0.33 for 24–72h
        return self.kelly_long        # 0.50 for > 72h

    def _best_ask(self, book) -> float:
        """Return best ask price from order book, or 0.0 if empty."""
        try:
            if hasattr(book, "best_ask") and book.best_ask is not None:
                return book.best_ask
            if hasattr(book, "asks_as_tuples"):
                asks = book.asks_as_tuples()
                return asks[0][0] if asks else 0.0
            return 0.0
        except (IndexError, AttributeError):
            return 0.0

    def _hours_to_resolution(self, market) -> float:
        """Hours between now and market end_date."""
        if market is None:
            return 48.0
        try:
            end = market.end_date
            if isinstance(end, str):
                end = datetime.fromisoformat(end.replace("Z", "+00:00"))
            if end.tzinfo is None:
                end = end.replace(tzinfo=timezone.utc)
            delta = end - datetime.now(tz=timezone.utc)
            hours = delta.total_seconds() / 3600
            return max(hours, 0.0)
        except Exception:
            logger.warning("[KELLY] Could not determine time horizon — using 48h default")
            return 48.0
