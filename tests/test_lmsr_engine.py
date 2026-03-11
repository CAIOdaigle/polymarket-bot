import numpy as np
import pytest

from src.analysis.lmsr_engine import LMSREngine
from src.config import LMSRConfig


@pytest.fixture
def engine():
    config = LMSRConfig(default_b=100.0, min_b=10.0, max_b=10000.0)
    return LMSREngine(config)


class TestCoreFunctions:
    def test_cost_basic(self):
        q = np.array([0.0, 0.0])
        b = 100.0
        cost = LMSREngine.cost(q, b)
        expected = b * np.log(2)  # ln(e^0 + e^0) = ln(2)
        assert abs(cost - expected) < 1e-10

    def test_price_equal_quantities(self):
        q = np.array([0.0, 0.0])
        b = 100.0
        prices = LMSREngine.price(q, b)
        assert abs(prices[0] - 0.5) < 1e-10
        assert abs(prices[1] - 0.5) < 1e-10

    def test_prices_sum_to_one(self):
        q = np.array([50.0, 30.0])
        b = 100.0
        prices = LMSREngine.price(q, b)
        assert abs(np.sum(prices) - 1.0) < 1e-10

    def test_higher_quantity_higher_price(self):
        q = np.array([100.0, 0.0])
        b = 100.0
        prices = LMSREngine.price(q, b)
        assert prices[0] > prices[1]

    def test_trade_cost_positive_for_buy(self):
        q = np.array([0.0, 0.0])
        b = 100.0
        cost = LMSREngine.trade_cost(q, b, outcome_idx=0, delta=10.0)
        assert cost > 0

    def test_max_loss_binary(self):
        b = 100_000
        loss = LMSREngine.max_loss(b, n_outcomes=2)
        expected = 100_000 * np.log(2)
        assert abs(loss - expected) < 1.0
        assert abs(loss - 69315) < 1.0  # ~$69,315 from the document


class TestBEstimation:
    def test_estimate_returns_default_with_empty_book(self, engine):
        b = engine.estimate_b_from_orderbook([], [], 0.5)
        assert b == engine.default_b

    def test_estimate_within_bounds(self, engine, sample_bids, sample_asks):
        b = engine.estimate_b_from_orderbook(sample_bids, sample_asks, 0.6)
        assert engine.min_b <= b <= engine.max_b

    def test_compute_state_basic(self, engine, sample_bids, sample_asks):
        state = engine.compute_state(sample_bids, sample_asks, 0.6)
        assert 0 < state.implied_price_yes < 1
        assert 0 < state.implied_price_no < 1
        assert abs(state.implied_price_yes + state.implied_price_no - 1.0) < 1e-10
        assert 0 <= state.confidence <= 1.0
