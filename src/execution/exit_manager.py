"""
Position exit management — model-state-driven exits.

Exit conditions (priority order):
1. Emergency floor — catastrophic drawdown guard (model/data failure)
2. Edge floor — posterior flipped negative with confidence
3. Edge convergence — trade played out, edge exhausted
4. Time backstop — position held too long
5. Liquidity gate — defers exit if book too thin for clean execution

Philosophy: All exits are driven by MODEL STATE (edge, confidence, time),
not price movement. There is no stop-loss or take-profit based on price alone.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from src.config import ExitConfig

logger = logging.getLogger(__name__)


class ExitReason(str, Enum):
    EDGE_FLOOR = "edge_floor"
    EDGE_CONVERGENCE = "edge_convergence"
    TIME_BACKSTOP = "time_backstop"
    EMERGENCY_FLOOR = "emergency_floor"
    LIQUIDITY_DEFER = "liquidity_defer"


@dataclass
class ExitSignal:
    token_id: str
    condition_id: str
    reason: ExitReason
    current_price: float
    entry_price: float
    pnl_pct: float
    size_to_sell: float
    edge_at_exit: float = 0.0
    confidence: float = 0.0
    deferred: bool = False
    metadata: dict = field(default_factory=dict)


class ExitManager:
    """Evaluates and executes model-state-driven exits."""

    def __init__(
        self,
        config: ExitConfig,
        order_mgr,
        positions,
        state_store,
        slack,
    ):
        self.cfg = config
        self.order_mgr = order_mgr
        self.positions = positions
        self.state_store = state_store
        self.slack = slack
        self._last_exit_time: dict[str, float] = {}

    def evaluate_position(
        self,
        pos,
        book,
        p_hat: float,
        confidence: float = 0.0,
        market=None,
    ) -> Optional[ExitSignal]:
        """Evaluate a single position for exit conditions.

        Returns ExitSignal if an exit should occur, None otherwise.
        Checks conditions in priority order.
        """
        # Get best bid from book
        best_bid = None
        total_bid_qty = 0.0
        if book.bids:
            best_bid = book.bids[0][0]
            total_bid_qty = sum(qty for _, qty in book.bids)

        if best_bid is None:
            return None

        # PnL calculation
        pnl_pct = (best_bid - pos.avg_price) / pos.avg_price if pos.avg_price > 0 else 0.0

        # For YES positions: edge = p_hat - best_bid
        # For NO positions: p_hat should already be the NO probability
        if pos.side == "YES":
            edge = p_hat - best_bid
        else:
            edge = p_hat - best_bid

        # --- Priority 1: Emergency floor (model/data failure guard) ---
        drawdown_pct = (pos.avg_price - best_bid) / pos.avg_price if pos.avg_price > 0 else 0.0
        if drawdown_pct >= self.cfg.emergency_floor_pct:
            signal = ExitSignal(
                token_id=pos.token_id,
                condition_id=pos.condition_id,
                reason=ExitReason.EMERGENCY_FLOOR,
                current_price=best_bid,
                entry_price=pos.avg_price,
                pnl_pct=pnl_pct,
                size_to_sell=pos.size,
                edge_at_exit=edge,
                confidence=confidence,
            )
            return self._apply_liquidity_gate(signal, pos, total_bid_qty)

        # --- Priority 2: Edge floor (posterior flipped with conviction) ---
        _EPS = 1e-9  # tolerance for floating-point boundary comparisons
        if (
            edge <= self.cfg.edge_floor_threshold + _EPS
            and confidence >= self.cfg.edge_floor_confidence_min
        ):
            signal = ExitSignal(
                token_id=pos.token_id,
                condition_id=pos.condition_id,
                reason=ExitReason.EDGE_FLOOR,
                current_price=best_bid,
                entry_price=pos.avg_price,
                pnl_pct=pnl_pct,
                size_to_sell=pos.size,
                edge_at_exit=edge,
                confidence=confidence,
            )
            return self._apply_liquidity_gate(signal, pos, total_bid_qty)

        # --- Priority 3: Edge convergence (trade played out) ---
        hold_seconds = time.time() - pos.entry_time
        if (
            hold_seconds >= self.cfg.edge_convergence_min_hold_s
            and 0 <= edge < self.cfg.edge_convergence_threshold
        ):
            signal = ExitSignal(
                token_id=pos.token_id,
                condition_id=pos.condition_id,
                reason=ExitReason.EDGE_CONVERGENCE,
                current_price=best_bid,
                entry_price=pos.avg_price,
                pnl_pct=pnl_pct,
                size_to_sell=pos.size,
                edge_at_exit=edge,
                confidence=confidence,
            )
            return self._apply_liquidity_gate(signal, pos, total_bid_qty)

        # --- Priority 4: Time backstop ---
        max_hold_seconds = self.cfg.max_hold_hours * 3600
        if hold_seconds >= max_hold_seconds:
            signal = ExitSignal(
                token_id=pos.token_id,
                condition_id=pos.condition_id,
                reason=ExitReason.TIME_BACKSTOP,
                current_price=best_bid,
                entry_price=pos.avg_price,
                pnl_pct=pnl_pct,
                size_to_sell=pos.size,
                edge_at_exit=edge,
                confidence=confidence,
            )
            return self._apply_liquidity_gate(signal, pos, total_bid_qty)

        return None

    def _apply_liquidity_gate(
        self,
        signal: ExitSignal,
        pos,
        total_bid_qty: float,
    ) -> ExitSignal:
        """Check if there's enough liquidity to exit cleanly.

        If book is too thin, defer the exit to avoid slippage.
        """
        required_qty = pos.size * self.cfg.min_exit_liquidity_pct
        if total_bid_qty < required_qty:
            logger.warning(
                "Liquidity defer: %s needs %.1f shares, book has %.1f",
                pos.token_id[:8],
                required_qty,
                total_bid_qty,
            )
            return ExitSignal(
                token_id=signal.token_id,
                condition_id=signal.condition_id,
                reason=ExitReason.LIQUIDITY_DEFER,
                current_price=signal.current_price,
                entry_price=signal.entry_price,
                pnl_pct=signal.pnl_pct,
                size_to_sell=signal.size_to_sell,
                edge_at_exit=signal.edge_at_exit,
                confidence=signal.confidence,
                deferred=True,
                metadata={"original_reason": signal.reason.value},
            )
        return signal

    def execute_exit(
        self,
        signal: ExitSignal,
        market=None,
        dry_run: bool = False,
    ) -> bool:
        """Execute an exit order. Returns True if order was placed/logged."""
        # Don't execute deferred signals
        if signal.deferred:
            logger.info(
                "Skipping deferred exit for %s (reason: %s)",
                signal.token_id[:8],
                signal.metadata.get("original_reason", "unknown"),
            )
            return False

        # Check cooldown
        last_exit = self._last_exit_time.get(signal.token_id, 0)
        if time.time() - last_exit < self.cfg.exit_cooldown_seconds:
            return False

        if dry_run:
            logger.info(
                "DRY RUN EXIT: %s reason=%s price=%.4f pnl=%.1f%% shares=%.2f",
                signal.token_id[:8],
                signal.reason.value,
                signal.current_price,
                signal.pnl_pct * 100,
                signal.size_to_sell,
            )
            self._last_exit_time[signal.token_id] = time.time()
            # Still reduce position in dry run for tracking
            self.positions.reduce_position(
                token_id=signal.token_id,
                size_sold=signal.size_to_sell,
                exit_price=signal.current_price,
            )
            return True

        # Place real sell order
        try:
            from src.execution.order_manager import TradeRequest

            request = TradeRequest(
                condition_id=signal.condition_id,
                token_id=signal.token_id,
                side="SELL",
                price=signal.current_price,
                size=signal.size_to_sell,
                order_type="GTC",
                edge=signal.edge_at_exit,
                kelly_fraction=0.0,
                neg_risk=getattr(market, "neg_risk", False) if market else False,
                tick_size=getattr(market, "tick_size", 0.01) if market else 0.01,
            )
            order = self.order_mgr.place_order(request)

            if order.status != "failed":
                self.positions.reduce_position(
                    token_id=signal.token_id,
                    size_sold=signal.size_to_sell,
                    exit_price=signal.current_price,
                )
                self._last_exit_time[signal.token_id] = time.time()

                # Persist position state
                pos = self.positions.get_position(signal.token_id)
                if pos:
                    self.state_store.save_position(pos)
                else:
                    self.state_store.delete_position(signal.token_id)

                return True

            logger.warning("Exit order failed for %s", signal.token_id[:8])
            return False

        except Exception:
            logger.exception("Failed to execute exit for %s", signal.token_id[:8])
            return False

    def check_all_positions(self, books: dict, bayesian, scanner) -> list[ExitSignal]:
        """Sweep all open positions for exit conditions.

        Also updates high-water marks.
        """
        open_positions = self.positions.get_all_open()
        logger.info("Exit sweep: %d open positions", len(open_positions))

        signals = []
        for pos in open_positions:
            book = books.get(pos.token_id)
            if book is None:
                continue

            # Update high-water mark
            best_bid = None
            if hasattr(book, "bids") and book.bids:
                if isinstance(book.bids, list):
                    best_bid = book.bids[0][0] if book.bids else None
                elif hasattr(book, "best_bid"):
                    best_bid = book.best_bid

            if best_bid is not None:
                pos.update_high_water_mark(best_bid)

            # Get Bayesian estimate
            p_hat = bayesian.get_estimate(pos.condition_id) if bayesian else None
            if p_hat is None:
                continue

            # Get LMSR confidence if available
            confidence = 0.5  # default
            if hasattr(bayesian, "get_confidence"):
                confidence = bayesian.get_confidence(pos.condition_id) or 0.5

            market = scanner.get_market(pos.condition_id) if scanner else None

            signal = self.evaluate_position(pos, book, p_hat, confidence, market)
            if signal is not None:
                signals.append(signal)

        return signals
