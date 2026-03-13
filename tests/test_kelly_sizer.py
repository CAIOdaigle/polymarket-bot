"""
Tests for corrected Kelly sizer.

Panel fixes:
  1. Confidence is a binary gate (not a multiplier)
  2. Correct NO-side Kelly formula
  3. Kelly fraction scales with time-to-resolution
"""

import pytest
from datetime import datetime, timezone, timedelta

from src.config import KellyConfig
from src.sizing.kelly_sizer import KellySizer


@pytest.fixture
def config():
    return KellyConfig(
        min_confidence=0.40,
        kelly_short_dated=0.25,
        kelly_medium_dated=0.33,
        kelly_long_dated=0.50,
        short_dated_hours=24.0,
        medium_dated_hours=72.0,
        min_edge_threshold=0.05,
        min_edge_after_spread=0.03,
        max_position_usd=100.0,
        total_bankroll_usd=1000.0,
        max_portfolio_exposure=0.50,
    )


@pytest.fixture
def sizer(config):
    return KellySizer(config)


class _FakeMarket:
    """Fake market with configurable end_date for time-horizon tests."""
    def __init__(self, hours_from_now: float):
        self.end_date = (
            datetime.now(tz=timezone.utc) + timedelta(hours=hours_from_now)
        ).isoformat()


# ---------------------------------------------------------------------------
# Edge detection
# ---------------------------------------------------------------------------

class TestEdgeDetection:
    def test_no_edge_no_trade(self, sizer):
        result = sizer.compute(
            p_hat=0.60,
            market_price_yes=0.60,
            market_price_no=0.40,
            current_position_usd=0,
            total_deployed_usd=0,
            lmsr_confidence=1.0,
            signal_count=1,
        )
        assert not result.should_trade
        assert result.side == "HOLD"

    def test_positive_edge_yes(self, sizer):
        result = sizer.compute(
            p_hat=0.70,
            market_price_yes=0.55,
            market_price_no=0.45,
            current_position_usd=0,
            total_deployed_usd=0,
            lmsr_confidence=1.0,
            signal_count=1,
        )
        assert result.should_trade
        assert result.side == "BUY_YES"
        assert result.edge > 0

    def test_positive_edge_no(self, sizer):
        # p_hat=0.30 → p_hat_no=0.70, ask_no=0.45, edge_no=0.25
        result = sizer.compute(
            p_hat=0.30,
            market_price_yes=0.55,
            market_price_no=0.45,
            current_position_usd=0,
            total_deployed_usd=0,
            lmsr_confidence=1.0,
            signal_count=1,
        )
        assert result.should_trade
        assert result.side == "BUY_NO"

    def test_edge_below_min_threshold(self, sizer):
        # p_hat=0.54, ask=0.50 → edge=0.04 < min_edge=0.05
        result = sizer.compute(
            p_hat=0.54,
            market_price_yes=0.50,
            market_price_no=0.50,
            current_position_usd=0,
            total_deployed_usd=0,
            lmsr_confidence=1.0,
            signal_count=1,
        )
        assert not result.should_trade


# ---------------------------------------------------------------------------
# Kelly math
# ---------------------------------------------------------------------------

class TestKellyMath:
    def test_kelly_formula_long_dated(self, sizer):
        """Long-dated (>72h) uses kelly_long=0.50."""
        market = _FakeMarket(hours_from_now=100)
        result = sizer.compute(
            p_hat=0.70,
            market_price_yes=0.50,
            market_price_no=0.50,
            current_position_usd=0,
            total_deployed_usd=0,
            lmsr_confidence=1.0,
            signal_count=1,
            market=market,
        )
        # Full Kelly: (0.70 - 0.50) / (1 - 0.50) = 0.40
        # Scaled: 0.50 * 0.40 = 0.20
        assert abs(result.kelly_fraction - 0.40) < 0.01
        assert abs(result.half_kelly_fraction - 0.20) < 0.01

    def test_kelly_formula_medium_dated(self, sizer):
        """Medium-dated (24-72h) uses kelly_medium=0.33."""
        market = _FakeMarket(hours_from_now=48)
        result = sizer.compute(
            p_hat=0.70,
            market_price_yes=0.50,
            market_price_no=0.50,
            current_position_usd=0,
            total_deployed_usd=0,
            lmsr_confidence=1.0,
            signal_count=1,
            market=market,
        )
        # Full Kelly: 0.40, Scaled: 0.33 * 0.40 = 0.132
        assert abs(result.kelly_fraction - 0.40) < 0.01
        assert abs(result.half_kelly_fraction - 0.132) < 0.01

    def test_kelly_formula_short_dated(self, sizer):
        """Short-dated (<24h) uses kelly_short=0.25."""
        market = _FakeMarket(hours_from_now=12)
        result = sizer.compute(
            p_hat=0.70,
            market_price_yes=0.50,
            market_price_no=0.50,
            current_position_usd=0,
            total_deployed_usd=0,
            lmsr_confidence=1.0,
            signal_count=1,
            market=market,
        )
        # Full Kelly: 0.40, Scaled: 0.25 * 0.40 = 0.10
        assert abs(result.kelly_fraction - 0.40) < 0.01
        assert abs(result.half_kelly_fraction - 0.10) < 0.01

    def test_no_side_kelly_uses_ask_no(self, sizer):
        """NO-side Kelly uses (p_hat_no - ask_no) / (1 - ask_no)."""
        market = _FakeMarket(hours_from_now=100)
        # p_hat=0.25 → p_hat_no=0.75, ask_no=0.60
        # edge_no = 0.75 - 0.60 = 0.15
        # f* = 0.15 / (1 - 0.60) = 0.375
        result = sizer.compute(
            p_hat=0.25,
            market_price_yes=0.40,
            market_price_no=0.60,
            current_position_usd=0,
            total_deployed_usd=0,
            lmsr_confidence=1.0,
            signal_count=1,
            market=market,
        )
        assert result.side == "BUY_NO"
        assert abs(result.kelly_fraction - 0.375) < 0.01
        assert abs(result.edge - 0.15) < 0.01


# ---------------------------------------------------------------------------
# Confidence gate (binary, not multiplier)
# ---------------------------------------------------------------------------

class TestConfidenceGate:
    def test_low_confidence_rejected(self, sizer):
        """Confidence below min_confidence (0.40) → no trade."""
        result = sizer.compute(
            p_hat=0.70,
            market_price_yes=0.50,
            market_price_no=0.50,
            current_position_usd=0,
            total_deployed_usd=0,
            lmsr_confidence=0.30,
            signal_count=1,
        )
        assert not result.should_trade
        assert "Confidence" in result.reason

    def test_high_confidence_approved(self, sizer):
        result = sizer.compute(
            p_hat=0.70,
            market_price_yes=0.50,
            market_price_no=0.50,
            current_position_usd=0,
            total_deployed_usd=0,
            lmsr_confidence=0.50,
            signal_count=1,
        )
        assert result.should_trade

    def test_confidence_is_gate_not_multiplier(self, sizer):
        """Two confidence values above the gate produce same sizing."""
        r1 = sizer.compute(
            p_hat=0.70,
            market_price_yes=0.50,
            market_price_no=0.50,
            current_position_usd=0,
            total_deployed_usd=0,
            lmsr_confidence=0.50,
            signal_count=1,
        )
        r2 = sizer.compute(
            p_hat=0.70,
            market_price_yes=0.50,
            market_price_no=0.50,
            current_position_usd=0,
            total_deployed_usd=0,
            lmsr_confidence=1.0,
            signal_count=1,
        )
        # Same sizing — confidence is NOT a multiplier
        assert r1.position_size_usd == r2.position_size_usd


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------

class TestConstraints:
    def test_position_limit(self, sizer):
        result = sizer.compute(
            p_hat=0.90,
            market_price_yes=0.50,
            market_price_no=0.50,
            current_position_usd=99.0,
            total_deployed_usd=99.0,
            lmsr_confidence=1.0,
            signal_count=1,
        )
        assert result.position_size_usd <= 1.0

    def test_bankroll_exposure_limit(self, sizer):
        result = sizer.compute(
            p_hat=0.90,
            market_price_yes=0.50,
            market_price_no=0.50,
            current_position_usd=0,
            total_deployed_usd=490.0,
            lmsr_confidence=1.0,
            signal_count=1,
        )
        assert result.position_size_usd <= 10.0

    def test_ask_near_one_rejected(self, sizer):
        """Ask price >= 0.99 → no trade (near-certain market)."""
        result = sizer.compute(
            p_hat=1.0,
            market_price_yes=0.995,
            market_price_no=0.005,
            current_position_usd=0,
            total_deployed_usd=0,
            lmsr_confidence=1.0,
            signal_count=1,
        )
        assert not result.should_trade
