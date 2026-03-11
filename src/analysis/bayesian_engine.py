"""
Real-Time Bayesian Signal Processing Engine.

Implements sequential Bayesian updating from the agent decision architecture doc:
  1. Bayes' theorem:  P(H|D) = P(D|H) * P(H) / P(D)
  2. Sequential update:  P(H|D1,...,Dt) proportional to P(H) * prod(P(Dk|H))
  3. Log-space:  log P(H|D) = log P(H) + sum(log P(Dk|H)) - log Z
  4. Expected value:  EV = p_hat - p

All computation in log-space for numerical stability.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.config import BayesianConfig
from src.utils.math_helpers import clamp

logger = logging.getLogger(__name__)


@dataclass
class SignalUpdate:
    """A single signal observation with log-likelihoods."""

    signal_name: str
    timestamp: float
    log_likelihood_yes: float  # log P(data | outcome=YES)
    log_likelihood_no: float  # log P(data | outcome=NO)
    confidence: float  # 0-1 self-reported signal confidence
    metadata: dict = field(default_factory=dict)


@dataclass
class MarketBelief:
    """Bayesian belief state for a single market."""

    condition_id: str
    log_prior_yes: float
    log_prior_no: float
    log_posterior_yes: float
    log_posterior_no: float
    signal_history: list[SignalUpdate] = field(default_factory=list)
    last_updated: float = 0.0

    @property
    def posterior_yes(self) -> float:
        """Normalized P(YES | all data), clamped to [0.05, 0.95]."""
        max_log = max(self.log_posterior_yes, self.log_posterior_no)
        log_z = max_log + np.log(
            np.exp(self.log_posterior_yes - max_log)
            + np.exp(self.log_posterior_no - max_log)
        )
        raw = float(np.exp(self.log_posterior_yes - log_z))
        return clamp(raw, 0.05, 0.95)

    @property
    def posterior_no(self) -> float:
        return 1.0 - self.posterior_yes

    @property
    def signal_count(self) -> int:
        return len(self.signal_history)


class BayesianEngine:
    # Dampen signal updates to prevent overconfidence from single observations.
    # Each signal's log-likelihood is multiplied by this factor before updating.
    SIGNAL_DAMPENING = 0.3

    def __init__(self, config: BayesianConfig):
        self.default_prior = config.default_prior
        self.prior_strength = config.prior_strength
        self.signal_decay_hours = config.signal_decay_hours
        self.min_signals = config.min_signals_to_trade
        self._beliefs: dict[str, MarketBelief] = {}

    def initialize_market(
        self,
        condition_id: str,
        market_price_yes: float,
        prior: Optional[float] = None,
    ) -> MarketBelief:
        """
        Initialize belief for a market.

        Uses market price as a weakly informative prior (efficient market assumption).
        """
        p = prior if prior is not None else market_price_yes
        p = clamp(p, 0.001, 0.999)

        log_prior_yes = np.log(p) * self.prior_strength
        log_prior_no = np.log(1 - p) * self.prior_strength

        belief = MarketBelief(
            condition_id=condition_id,
            log_prior_yes=float(log_prior_yes),
            log_prior_no=float(log_prior_no),
            log_posterior_yes=float(log_prior_yes),
            log_posterior_no=float(log_prior_no),
            last_updated=time.time(),
        )
        self._beliefs[condition_id] = belief
        return belief

    def update(self, condition_id: str, signal: SignalUpdate) -> Optional[MarketBelief]:
        """
        Apply a single Bayesian update:
          log P(YES|D) += log P(D|YES) * confidence
          log P(NO|D)  += log P(D|NO)  * confidence
        """
        belief = self._beliefs.get(condition_id)
        if belief is None:
            return None

        weight = signal.confidence * self.SIGNAL_DAMPENING
        belief.log_posterior_yes += signal.log_likelihood_yes * weight
        belief.log_posterior_no += signal.log_likelihood_no * weight
        belief.signal_history.append(signal)
        belief.last_updated = time.time()
        return belief

    def recompute_with_decay(self, condition_id: str) -> Optional[MarketBelief]:
        """
        Recompute posterior from scratch with time-decayed signals.
        decay = exp(-dt / (decay_hours * 3600))
        """
        belief = self._beliefs.get(condition_id)
        if belief is None:
            return None

        now = time.time()
        log_post_yes = belief.log_prior_yes
        log_post_no = belief.log_prior_no

        for sig in belief.signal_history:
            dt = now - sig.timestamp
            decay = np.exp(-dt / (self.signal_decay_hours * 3600))
            weight = sig.confidence * self.SIGNAL_DAMPENING * decay
            log_post_yes += sig.log_likelihood_yes * weight
            log_post_no += sig.log_likelihood_no * weight

        belief.log_posterior_yes = float(log_post_yes)
        belief.log_posterior_no = float(log_post_no)
        belief.last_updated = now
        return belief

    def get_estimate(self, condition_id: str) -> Optional[float]:
        """Current estimated true probability P(YES)."""
        belief = self._beliefs.get(condition_id)
        if belief is None:
            return None
        return belief.posterior_yes

    def has_sufficient_signals(self, condition_id: str) -> bool:
        belief = self._beliefs.get(condition_id)
        if belief is None:
            return False
        return belief.signal_count >= self.min_signals

    def get_belief(self, condition_id: str) -> Optional[MarketBelief]:
        return self._beliefs.get(condition_id)

    @property
    def beliefs(self) -> dict[str, MarketBelief]:
        return self._beliefs
