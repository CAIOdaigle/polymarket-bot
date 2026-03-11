import time

import numpy as np
import pytest

from src.analysis.bayesian_engine import BayesianEngine, SignalUpdate
from src.config import BayesianConfig


@pytest.fixture
def engine():
    config = BayesianConfig(
        default_prior=0.5,
        prior_strength=1.0,
        signal_decay_hours=24.0,
        min_signals_to_trade=1,
    )
    return BayesianEngine(config)


class TestInitialization:
    def test_init_with_market_price(self, engine):
        belief = engine.initialize_market("test-1", 0.7)
        assert abs(belief.posterior_yes - 0.7) < 0.01

    def test_init_at_50_50(self, engine):
        belief = engine.initialize_market("test-2", 0.5)
        assert abs(belief.posterior_yes - 0.5) < 0.01


class TestUpdating:
    def test_positive_signal_increases_yes(self, engine):
        engine.initialize_market("test", 0.5)
        before = engine.get_estimate("test")

        signal = SignalUpdate(
            signal_name="test_signal",
            timestamp=time.time(),
            log_likelihood_yes=0.5,
            log_likelihood_no=-0.5,
            confidence=1.0,
        )
        engine.update("test", signal)
        after = engine.get_estimate("test")

        assert after > before

    def test_negative_signal_decreases_yes(self, engine):
        engine.initialize_market("test", 0.5)
        before = engine.get_estimate("test")

        signal = SignalUpdate(
            signal_name="test_signal",
            timestamp=time.time(),
            log_likelihood_yes=-0.5,
            log_likelihood_no=0.5,
            confidence=1.0,
        )
        engine.update("test", signal)
        after = engine.get_estimate("test")

        assert after < before

    def test_posterior_always_valid_probability(self, engine):
        engine.initialize_market("test", 0.5)
        for _ in range(100):
            signal = SignalUpdate(
                signal_name="test",
                timestamp=time.time(),
                log_likelihood_yes=0.1,
                log_likelihood_no=-0.1,
                confidence=1.0,
            )
            engine.update("test", signal)

        p = engine.get_estimate("test")
        assert 0 < p < 1

    def test_sufficient_signals_flag(self, engine):
        engine.initialize_market("test", 0.5)
        assert not engine.has_sufficient_signals("test")

        signal = SignalUpdate(
            signal_name="test",
            timestamp=time.time(),
            log_likelihood_yes=0.1,
            log_likelihood_no=-0.1,
            confidence=1.0,
        )
        engine.update("test", signal)
        assert engine.has_sufficient_signals("test")


class TestDecay:
    def test_decay_moves_toward_prior(self, engine):
        engine.initialize_market("test", 0.5)

        # Add an old signal
        old_signal = SignalUpdate(
            signal_name="test",
            timestamp=time.time() - 100 * 3600,  # 100 hours ago
            log_likelihood_yes=1.0,
            log_likelihood_no=-1.0,
            confidence=1.0,
        )
        engine.update("test", old_signal)

        # Recompute with decay — old signal should have minimal effect
        engine.recompute_with_decay("test")
        p = engine.get_estimate("test")

        # Should be close to the prior of 0.5
        assert abs(p - 0.5) < 0.1
