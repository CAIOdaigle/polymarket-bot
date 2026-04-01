"""
tests/test_exit_manager.py

Tests for framework-aligned exit logic.

Philosophy:
  - All tests validate MODEL-STATE-driven exits, not price-movement exits.
  - The three previously-wrong conditions (stop-loss, trailing stop, take-profit)
    are replaced with edge-floor, edge-convergence, and liquidity-gate tests.
  - EMERGENCY_FLOOR is tested as a rare circuit breaker, not a routine stop-loss.
"""

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from src.execution.exit_manager import ExitConfig, ExitManager, ExitReason, ExitSignal


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

def make_config(**overrides) -> ExitConfig:
    cfg = ExitConfig()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


@dataclass
class FakePosition:
    token_id:        str   = "tok_yes_001"
    condition_id:    str   = "cond_001"
    avg_price:       float = 0.50
    size:            float = 100.0
    side:            str   = "YES"
    entry_time:      float = field(
        default_factory=lambda: time.time() - 3600  # 1 hour ago
    )
    high_water_mark: float = 0.50
    realized_pnl:    float = 0.0

    def update_high_water_mark(self, price: float):
        if price > self.high_water_mark:
            self.high_water_mark = price


@dataclass
class FakeBook:
    """Order book stub matching OrderBookState API."""
    _bids: list = field(default_factory=list)  # [(price, qty), ...]

    def __init__(self, bids=None):
        self._bids = bids or []

    @property
    def best_bid(self) -> Optional[float]:
        return self._bids[0][0] if self._bids else None

    def bids_as_tuples(self) -> list:
        return list(self._bids)


def make_manager(**cfg_overrides) -> ExitManager:
    cfg            = make_config(**cfg_overrides)
    order_mgr      = MagicMock()
    pos_tracker    = MagicMock()
    state_store    = MagicMock()
    slack          = MagicMock()
    return ExitManager(cfg, order_mgr, pos_tracker, state_store, slack)


# ---------------------------------------------------------------------------
# 1. Edge-floor exit — triggers when posterior flips negative with conviction
# ---------------------------------------------------------------------------

class TestEdgeFloor:

    def test_triggers_when_edge_below_floor_and_confidence_sufficient(self):
        mgr = make_manager(edge_floor_threshold=-0.05, edge_floor_confidence_min=0.60)
        pos = FakePosition(avg_price=0.50)
        # p_hat=0.40, best_bid=0.48 -> edge = 0.40 - 0.48 = -0.08 (below -0.05 floor)
        book = FakeBook(bids=[(0.48, 200)])
        signal = mgr.evaluate_position(pos, book, p_hat=0.40, confidence=0.75, market=None)
        assert signal is not None
        assert signal.reason == ExitReason.EDGE_FLOOR

    def test_does_not_trigger_when_confidence_below_minimum(self):
        """Low-confidence reversal should NOT exit — avoids whipsaw on noisy signals."""
        mgr = make_manager(edge_floor_threshold=-0.05, edge_floor_confidence_min=0.60)
        pos = FakePosition(avg_price=0.50)
        book = FakeBook(bids=[(0.48, 200)])
        # Same edge=-0.08 but confidence only 0.45
        signal = mgr.evaluate_position(pos, book, p_hat=0.40, confidence=0.45, market=None)
        assert signal is None or signal.reason != ExitReason.EDGE_FLOOR

    def test_does_not_trigger_when_edge_just_above_floor(self):
        mgr = make_manager(edge_floor_threshold=-0.05)
        pos = FakePosition(avg_price=0.50)
        # edge = 0.52 - 0.55 = -0.03 (above -0.05 floor — no exit)
        book = FakeBook(bids=[(0.55, 200)])
        signal = mgr.evaluate_position(pos, book, p_hat=0.52, confidence=0.80, market=None)
        assert signal is None or signal.reason != ExitReason.EDGE_FLOOR

    def test_edge_at_exact_floor_triggers(self):
        """Boundary: edge == threshold should trigger."""
        mgr = make_manager(edge_floor_threshold=-0.05)
        pos = FakePosition(avg_price=0.50)
        # edge = 0.43 - 0.48 = -0.05 exactly
        book = FakeBook(bids=[(0.48, 200)])
        signal = mgr.evaluate_position(pos, book, p_hat=0.43, confidence=0.70, market=None)
        assert signal is not None
        assert signal.reason == ExitReason.EDGE_FLOOR


# ---------------------------------------------------------------------------
# 2. Edge-convergence exit — trade has played out
# ---------------------------------------------------------------------------

class TestEdgeConvergence:

    def test_triggers_when_edge_exhausted_after_min_hold(self):
        """Market has converged to model price — edge gone, take profit naturally."""
        mgr = make_manager(
            edge_convergence_threshold=0.03,
            edge_convergence_min_hold_s=300,
        )
        # Position entered 30 min ago
        pos = FakePosition(
            avg_price=0.45,
            entry_time=time.time() - 1800,
        )
        # p_hat=0.62, best_bid=0.61 -> edge=0.01 < 0.03 threshold
        book = FakeBook(bids=[(0.61, 200)])
        signal = mgr.evaluate_position(pos, book, p_hat=0.62, confidence=0.80, market=None)
        assert signal is not None
        assert signal.reason == ExitReason.EDGE_CONVERGENCE

    def test_does_not_trigger_before_min_hold_period(self):
        """Don't exit in the first 5 minutes — let price settle."""
        mgr = make_manager(
            edge_convergence_threshold=0.03,
            edge_convergence_min_hold_s=300,
        )
        pos = FakePosition(
            avg_price=0.45,
            entry_time=time.time() - 120,  # 2 minutes ago
        )
        book = FakeBook(bids=[(0.61, 200)])
        signal = mgr.evaluate_position(pos, book, p_hat=0.62, confidence=0.80, market=None)
        assert signal is None or signal.reason != ExitReason.EDGE_CONVERGENCE

    def test_does_not_trigger_when_edge_still_substantial(self):
        mgr = make_manager(edge_convergence_threshold=0.03)
        pos = FakePosition(
            avg_price=0.45,
            entry_time=time.time() - 7200,
        )
        # edge = 0.70 - 0.55 = 0.15 — well above threshold, hold the position
        book = FakeBook(bids=[(0.55, 200)])
        signal = mgr.evaluate_position(pos, book, p_hat=0.70, confidence=0.85, market=None)
        assert signal is None

    def test_pnl_is_positive_on_convergence_exit(self):
        """Convergence exit should typically show a profit — market moved our way."""
        mgr = make_manager(edge_convergence_threshold=0.03)
        pos = FakePosition(
            avg_price=0.45,
            entry_time=time.time() - 3600,
        )
        book = FakeBook(bids=[(0.61, 200)])
        signal = mgr.evaluate_position(pos, book, p_hat=0.62, confidence=0.80, market=None)
        if signal and signal.reason == ExitReason.EDGE_CONVERGENCE:
            assert signal.pnl_pct > 0


# ---------------------------------------------------------------------------
# 3. Liquidity gate (Othman layer)
# ---------------------------------------------------------------------------

class TestLiquidityGate:

    def test_defers_exit_when_book_too_thin(self):
        """Prevent exiting into a thin book that would destroy edge through slippage."""
        mgr = make_manager(
            edge_floor_threshold=-0.05,
            edge_floor_confidence_min=0.60,
            min_exit_liquidity_pct=0.80,
            min_acceptable_bid_pct=0.95,
        )
        pos = FakePosition(avg_price=0.50, size=100)
        # Edge floor triggered, BUT only 10 shares of liquidity (need 80)
        book = FakeBook(bids=[(0.48, 10)])
        signal = mgr.evaluate_position(pos, book, p_hat=0.40, confidence=0.75, market=None)
        # Should return a deferred signal, not None
        assert signal is not None
        assert signal.deferred is True
        assert signal.reason == ExitReason.LIQUIDITY_DEFER

    def test_execute_exit_skips_deferred_signal(self):
        """A deferred signal must NOT result in an order being placed."""
        mgr = make_manager()
        signal = ExitSignal(
            token_id="tok_001", condition_id="cond_001",
            reason=ExitReason.LIQUIDITY_DEFER, current_price=0.48,
            entry_price=0.50, pnl_pct=-0.04, size_to_sell=100,
            edge_at_exit=-0.08, confidence=0.75, deferred=True,
        )
        result = mgr.execute_exit(signal, market=None)
        assert result is False
        mgr.order_mgr.place_order.assert_not_called()

    def test_proceeds_when_liquidity_adequate(self):
        """Exit should proceed when book can absorb the full position."""
        mgr = make_manager(
            edge_floor_threshold=-0.05,
            edge_floor_confidence_min=0.60,
            min_exit_liquidity_pct=0.80,
        )
        pos = FakePosition(avg_price=0.50, size=100)
        # Plenty of liquidity at or near best bid
        book = FakeBook(bids=[(0.48, 90), (0.47, 20)])
        signal = mgr.evaluate_position(pos, book, p_hat=0.40, confidence=0.75, market=None)
        assert signal is not None
        assert signal.deferred is False
        assert signal.reason == ExitReason.EDGE_FLOOR


# ---------------------------------------------------------------------------
# 4. Time backstop
# ---------------------------------------------------------------------------

class TestTimeBackstop:

    def test_triggers_after_max_hold_hours(self):
        mgr = make_manager(max_hold_hours=72)
        pos = FakePosition(
            avg_price=0.50,
            entry_time=time.time() - (73 * 3600),
        )
        # Model still has slight edge — time backstop fires anyway
        book = FakeBook(bids=[(0.55, 200)])
        signal = mgr.evaluate_position(pos, book, p_hat=0.60, confidence=0.70, market=None)
        assert signal is not None
        assert signal.reason == ExitReason.TIME_BACKSTOP

    def test_does_not_trigger_before_max_hold(self):
        mgr = make_manager(max_hold_hours=72)
        pos = FakePosition(
            avg_price=0.50,
            entry_time=time.time() - (10 * 3600),
        )
        book = FakeBook(bids=[(0.55, 200)])
        # Edge still strong — no exit
        signal = mgr.evaluate_position(pos, book, p_hat=0.70, confidence=0.80, market=None)
        assert signal is None


# ---------------------------------------------------------------------------
# 5. Emergency floor (model/data failure guard, NOT a routine stop-loss)
# ---------------------------------------------------------------------------

class TestEmergencyFloor:

    def test_triggers_at_catastrophic_drawdown(self):
        """25% floor should only fire when something has gone badly wrong."""
        mgr = make_manager(emergency_floor_pct=0.25)
        pos = FakePosition(avg_price=0.60)
        # best_bid dropped from 0.60 to 0.44 — 26.7% drop
        book = FakeBook(bids=[(0.44, 200)])
        # p_hat still "positive" — model may be broken
        signal = mgr.evaluate_position(pos, book, p_hat=0.46, confidence=0.40, market=None)
        assert signal is not None
        assert signal.reason == ExitReason.EMERGENCY_FLOOR

    def test_does_not_trigger_on_normal_adverse_move(self):
        """A 10% adverse move should NOT trigger emergency floor."""
        mgr = make_manager(emergency_floor_pct=0.25)
        pos = FakePosition(avg_price=0.50)
        book = FakeBook(bids=[(0.45, 200)])  # 10% down
        signal = mgr.evaluate_position(pos, book, p_hat=0.42, confidence=0.40, market=None)
        # Edge floor may or may not fire here, but EMERGENCY_FLOOR should not
        if signal:
            assert signal.reason != ExitReason.EMERGENCY_FLOOR


# ---------------------------------------------------------------------------
# 6. Operational tests (cooldown, dry-run, NO position, reduce_position)
# ---------------------------------------------------------------------------

class TestOperational:

    def test_exit_cooldown_prevents_rapid_re_exit(self):
        mgr = make_manager(exit_cooldown_seconds=60)
        mgr._last_exit_time["tok_yes_001"] = time.time()  # just exited
        signal = ExitSignal(
            token_id="tok_yes_001", condition_id="cond_001",
            reason=ExitReason.EDGE_FLOOR, current_price=0.48,
            entry_price=0.50, pnl_pct=-0.04, size_to_sell=100,
            edge_at_exit=-0.08, confidence=0.75,
        )
        result = mgr.execute_exit(signal, market=None)
        assert result is False
        mgr.order_mgr.place_order.assert_not_called()

    def test_dry_run_logs_without_placing_order(self):
        mgr = make_manager()
        signal = ExitSignal(
            token_id="tok_yes_001", condition_id="cond_001",
            reason=ExitReason.EDGE_CONVERGENCE, current_price=0.61,
            entry_price=0.45, pnl_pct=0.355, size_to_sell=100,
            edge_at_exit=0.01, confidence=0.80,
        )
        result = mgr.execute_exit(signal, market=None, dry_run=True)
        assert result is True
        mgr.order_mgr.place_order.assert_not_called()

    def test_no_position_pnl_calculation(self):
        """NO position: PnL uses NO token's best_bid vs avg_price."""
        mgr = make_manager(edge_floor_threshold=-0.05, edge_floor_confidence_min=0.60)
        pos = FakePosition(avg_price=0.45, size=100, side="NO", token_id="tok_no_001")
        # NO token best_bid = 0.38 -> pnl = (0.38-0.45)/0.45 = -15.6%
        book = FakeBook(bids=[(0.38, 200)])
        # p_hat for NO = 0.30 -> edge = 0.30 - 0.38 = -0.08 (below floor)
        signal = mgr.evaluate_position(pos, book, p_hat=0.30, confidence=0.75, market=None)
        assert signal is not None
        assert abs(signal.pnl_pct - (-0.1556)) < 0.01

    def test_reduce_position_called_on_dry_run_exit(self):
        """After dry_run execute_exit, position_tracker.reduce_position should be called."""
        mgr = make_manager()
        signal = ExitSignal(
            token_id="tok_yes_001", condition_id="cond_001",
            reason=ExitReason.EDGE_FLOOR, current_price=0.48,
            entry_price=0.50, pnl_pct=-0.04, size_to_sell=100,
            edge_at_exit=-0.08, confidence=0.75,
        )
        mgr.execute_exit(signal, market=None, dry_run=True)
        mgr.positions.reduce_position.assert_called_once_with(
            token_id="tok_yes_001", size_sold=100, exit_price=0.48
        )

    def test_check_all_positions_logs_sweep(self, caplog):
        """Sweep should log position count on every run."""
        mgr = make_manager()
        mgr.positions.get_all_open.return_value = []
        import logging
        with caplog.at_level(logging.INFO, logger="src.execution.exit_manager"):
            mgr.check_all_positions(books={}, bayesian=MagicMock(), scanner=MagicMock())
        assert "Exit sweep" in caplog.text or "0 open position" in caplog.text
