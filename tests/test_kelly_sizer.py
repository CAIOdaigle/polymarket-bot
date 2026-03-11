import pytest

from src.config import TradingConfig
from src.sizing.kelly_sizer import KellySizer


@pytest.fixture
def sizer():
    config = TradingConfig(
        kelly_fraction=0.5,
        min_edge_threshold=0.02,
        max_position_usd=100.0,
        total_bankroll_usd=1000.0,
        max_portfolio_exposure=0.5,
    )
    return KellySizer(config)


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


class TestKellyMath:
    def test_half_kelly_is_half(self, sizer):
        result = sizer.compute(
            p_hat=0.70,
            market_price_yes=0.50,
            market_price_no=0.50,
            current_position_usd=0,
            total_deployed_usd=0,
            lmsr_confidence=1.0,
            signal_count=1,
        )
        # Full Kelly: (0.7 - 0.5) / (1 - 0.5) = 0.4
        # Half Kelly: 0.2
        assert abs(result.kelly_fraction - 0.4) < 0.01
        assert abs(result.half_kelly_fraction - 0.2) < 0.01


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

    def test_bankroll_limit(self, sizer):
        result = sizer.compute(
            p_hat=0.90,
            market_price_yes=0.50,
            market_price_no=0.50,
            current_position_usd=0,
            total_deployed_usd=490.0,  # Near 50% exposure limit
            lmsr_confidence=1.0,
            signal_count=1,
        )
        assert result.position_size_usd <= 10.0

    def test_confidence_scaling(self, sizer):
        high_conf = sizer.compute(
            p_hat=0.70,
            market_price_yes=0.50,
            market_price_no=0.50,
            current_position_usd=0,
            total_deployed_usd=0,
            lmsr_confidence=1.0,
            signal_count=1,
        )
        low_conf = sizer.compute(
            p_hat=0.70,
            market_price_yes=0.50,
            market_price_no=0.50,
            current_position_usd=0,
            total_deployed_usd=0,
            lmsr_confidence=0.1,
            signal_count=1,
        )
        assert high_conf.position_size_usd > low_conf.position_size_usd
