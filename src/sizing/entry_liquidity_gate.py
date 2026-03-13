"""
entry_liquidity_gate.py — Ask-side liquidity check before entering a position.

Mirrors the exit liquidity gate in exit_manager.py.

Without this, the bot can enter a position in a thin market, move the book
against itself on entry, and then be blocked from exiting by the exit liquidity
gate — a one-way trap where entry was possible but exit never is.

The symmetry requirement:
  If you cannot exit a market cleanly → you should not have entered it.
  Entry gate uses the ask side (you're buying).
  Exit gate uses the bid side (you're selling).
  Both require 80% coverage of the intended order size at an acceptable price.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EntryLiquidityConfig:
    # Must be able to fill this fraction of the order at an acceptable price
    min_fill_coverage: float = 0.80

    # Won't pay more than this multiple above best ask
    max_slippage_pct: float = 0.05   # 5% above best ask is the ceiling

    # Minimum absolute depth required at any price (sanity floor)
    min_absolute_depth: float = 5.0  # at least $5 of ask liquidity

    # Check exit viability before entry
    check_exit_viability: bool = True


class EntryLiquidityGate:
    """
    Checks ask-side order book depth before approving an entry order.

    Called from the main pipeline after Kelly sizing and before order placement.
    Returns (approved: bool, available_depth: float, reason: str).
    """

    def __init__(self, config: EntryLiquidityConfig = None):
        self.cfg = config or EntryLiquidityConfig()

    def check(
        self,
        book,               # order book for the token being bought
        size_usd: float,    # intended order size in USD
        ask_price: float,   # best ask price (from Kelly sizer)
    ) -> tuple[bool, float, str]:
        """
        Evaluate whether the ask side can absorb the intended order.

        Returns:
            approved:        True if entry is safe to proceed
            available_depth: USD-equivalent depth available at acceptable prices
            reason:          Human-readable decision rationale
        """
        # Get asks as tuples: [(price, qty), ...]
        asks = self._get_asks(book)
        if not asks:
            return False, 0.0, "No ask-side liquidity — empty book"

        # Ceiling price: won't pay more than max_slippage above best ask
        max_acceptable_ask = ask_price * (1.0 + self.cfg.max_slippage_pct)

        # Sum available ask liquidity within acceptable price range
        available_shares = sum(
            qty for price, qty in asks
            if price <= max_acceptable_ask
        )
        available_depth_usd = available_shares * ask_price

        # Sanity floor
        if available_depth_usd < self.cfg.min_absolute_depth:
            reason = (
                f"Ask depth ${available_depth_usd:.2f} below "
                f"minimum ${self.cfg.min_absolute_depth:.2f}"
            )
            logger.debug("[ENTRY GATE] Rejected — %s", reason)
            return False, available_depth_usd, reason

        # Coverage check: can we fill the required fraction of the order?
        required_depth = size_usd * self.cfg.min_fill_coverage
        if available_depth_usd < required_depth:
            reason = (
                f"Insufficient depth: ${available_depth_usd:.2f} available, "
                f"${required_depth:.2f} required for {self.cfg.min_fill_coverage*100:.0f}% fill"
            )
            logger.debug("[ENTRY GATE] Rejected — %s", reason)
            return False, available_depth_usd, reason

        reason = (
            f"Approved — ${available_depth_usd:.2f} depth covers "
            f"${size_usd:.2f} order ({available_depth_usd/size_usd*100:.0f}% coverage)"
        )
        logger.debug("[ENTRY GATE] %s", reason)
        return True, available_depth_usd, reason

    def check_both_sides(
        self,
        book_entry,         # book for the token being bought (ask side checked)
        book_exit,          # book for the same token (bid side checked for exit viability)
        size_usd: float,
        ask_price: float,
        bid_price: float,
    ) -> tuple[bool, str]:
        """
        Full symmetry check: entry AND exit liquidity before committing.

        Prevents entering a position that would immediately be trapped by
        the exit liquidity gate.

        Entry check: ask side of book_entry
        Exit check:  bid side of book_exit (same token, opposite side)
        """
        entry_ok, entry_depth, entry_reason = self.check(book_entry, size_usd, ask_price)
        if not entry_ok:
            return False, f"Entry blocked: {entry_reason}"

        if not self.cfg.check_exit_viability:
            return True, entry_reason

        # Exit viability: simulate exit gate on bid side
        bids = self._get_bids(book_exit)
        if not bids:
            return False, "Exit liquidity check failed: no bids in book"

        min_bid_price = bid_price * 0.95  # mirrors exit gate's min_acceptable_bid_pct
        available_bids = sum(
            qty for price, qty in bids
            if price >= min_bid_price
        )
        available_bid_usd = available_bids * bid_price
        required_bid_usd = size_usd * self.cfg.min_fill_coverage

        if available_bid_usd < required_bid_usd:
            reason = (
                f"Exit liquidity insufficient at entry time: "
                f"${available_bid_usd:.2f} bid depth vs ${required_bid_usd:.2f} required. "
                f"Would be trapped on exit."
            )
            logger.warning("[ENTRY GATE] Symmetric check failed — %s", reason)
            return False, reason

        return True, f"Entry and exit liquidity both confirmed (${entry_depth:.2f} / ${available_bid_usd:.2f})"

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _get_asks(self, book) -> list[tuple[float, float]]:
        """Get asks as [(price, qty), ...] from any book type."""
        try:
            if hasattr(book, "asks_as_tuples"):
                return book.asks_as_tuples()
            if hasattr(book, "asks"):
                return list(book.asks)
            return []
        except Exception:
            return []

    def _get_bids(self, book) -> list[tuple[float, float]]:
        """Get bids as [(price, qty), ...] from any book type."""
        try:
            if hasattr(book, "bids_as_tuples"):
                return book.bids_as_tuples()
            if hasattr(book, "bids"):
                return list(book.bids)
            return []
        except Exception:
            return []
