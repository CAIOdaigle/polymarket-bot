"""
bayesian_engine.py — Corrected Bayesian signal aggregation.

Key fix: signals are grouped by the information source they observe.
Signals within the same group (order_flow, price_discovery) are correlated —
they often fire from the same underlying event. Summing their log-likelihoods
as if independent inflates posterior conviction beyond what the information
content warrants (likelihood inflation).

Fix: within each group, take the strongest signal at full weight.
Additional signals in the same group contribute at a heavily dampened rate (0.15).
Cross-group signals sum normally — they observe genuinely different phenomena.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Signal dependency groups
# ---------------------------------------------------------------------------
# Signals in the same group share an underlying information source.
# Whale trade → triggers whale_tracker, orderbook_imbalance, AND volume.
# That's one event, not three independent confirmations.
#
# price_discovery signals all observe price discrepancies from different angles
# but are driven by the same latent mispricing — also correlated.

SIGNAL_GROUPS: dict[str, list[str]] = {
    "order_flow":       ["whale_tracker", "orderbook_imbalance", "volume"],
    "price_discovery":  ["lmsr_deviation", "cross_market", "related_market"],
}

# Weight for 2nd, 3rd signals within a group (heavy dampening — not independent)
INTRA_GROUP_DECAY = 0.15

# Minimum confidence for a signal to contribute at all
MIN_SIGNAL_CONFIDENCE = 0.20


@dataclass
class SignalUpdate:
    """A single signal observation with log-likelihoods.

    Kept as SignalUpdate (not Signal) for backward compatibility with
    signal_registry.py and all signal implementations.
    """
    signal_name: str
    timestamp: float
    log_likelihood_yes: float  # log P(data | outcome=YES)
    log_likelihood_no: float   # log P(data | outcome=NO)
    confidence: float          # 0-1 self-reported signal confidence
    metadata: dict = field(default_factory=dict)


@dataclass
class MarketBelief:
    """Bayesian belief state for a single market.

    Maintained for backward compatibility with exit_manager and main.py.
    """
    condition_id: str
    signal_history: list[SignalUpdate] = field(default_factory=list)
    last_updated: float = 0.0
    _cached_posterior: float = 0.5
    _cached_confidence: float = 0.0

    @property
    def posterior_yes(self) -> float:
        return self._cached_posterior

    @property
    def posterior_no(self) -> float:
        return 1.0 - self._cached_posterior

    @property
    def signal_count(self) -> int:
        return len(self.signal_history)


class BayesianEngine:
    """
    Sequential Bayesian updater with group-aware signal aggregation.

    The market price at evaluation time is used as the prior — this anchors
    the posterior to observed market information rather than a fixed flat prior,
    which is more appropriate when the market itself is informative.

    Maintains backward-compatible API:
      - initialize_market(), update(), get_estimate(), has_sufficient_signals()
      - Internal state stored in MarketBelief objects
      - Group-aware aggregation applied when computing posteriors
    """

    def __init__(self, config):
        # Accept both old BayesianConfig and new-style config
        self.min_signals = getattr(config, "min_signals_to_trade", 2)
        self.posterior_clamp_low = getattr(config, "posterior_clamp_low", 0.05)
        self.posterior_clamp_high = getattr(config, "posterior_clamp_high", 0.95)
        self.decay_interval_seconds = getattr(config, "decay_interval_seconds", 300.0)
        self.time_decay_alpha = getattr(config, "time_decay_alpha", 0.85)
        self.min_signal_confidence = getattr(config, "min_signal_confidence", MIN_SIGNAL_CONFIDENCE)
        self.intra_group_decay = getattr(config, "intra_group_decay", INTRA_GROUP_DECAY)

        # Signal group configuration (can be overridden from config)
        self.signal_groups = dict(SIGNAL_GROUPS)

        self._beliefs: dict[str, MarketBelief] = {}
        self._market_prices: dict[str, float] = {}  # condition_id -> latest market price

    def initialize_market(
        self,
        condition_id: str,
        market_price_yes: float,
        prior: Optional[float] = None,
    ) -> MarketBelief:
        """Initialize belief for a market."""
        p = prior if prior is not None else market_price_yes
        p = max(0.001, min(0.999, p))

        belief = MarketBelief(
            condition_id=condition_id,
            last_updated=time.time(),
            _cached_posterior=p,
            _cached_confidence=0.0,
        )
        self._beliefs[condition_id] = belief
        self._market_prices[condition_id] = p
        return belief

    def update_market_price(self, condition_id: str, market_price: float) -> None:
        """Update the market price used as prior for this market."""
        self._market_prices[condition_id] = max(0.001, min(0.999, market_price))

    def update(self, condition_id: str, signal: SignalUpdate) -> Optional[MarketBelief]:
        """
        Add a signal and recompute posterior with group-aware aggregation.
        """
        belief = self._beliefs.get(condition_id)
        if belief is None:
            return None

        belief.signal_history.append(signal)
        belief.last_updated = time.time()

        # Recompute posterior using all signals with group deduplication
        market_price = self._market_prices.get(condition_id, 0.5)
        p_hat, confidence = self._compute_posterior(belief.signal_history, market_price)
        belief._cached_posterior = p_hat
        belief._cached_confidence = confidence

        return belief

    def recompute_with_decay(self, condition_id: str) -> Optional[MarketBelief]:
        """
        Recompute posterior with time-decayed signals.
        Old signals contribute less — decay_alpha^(intervals_elapsed).
        """
        belief = self._beliefs.get(condition_id)
        if belief is None:
            return None

        now = time.time()

        # Apply time decay to signal confidences (temporary for recompute)
        decayed_signals = []
        for sig in belief.signal_history:
            elapsed = now - sig.timestamp
            n_intervals = elapsed / self.decay_interval_seconds
            if n_intervals >= 1.0:
                decay_factor = self.time_decay_alpha ** int(n_intervals)
                decayed = SignalUpdate(
                    signal_name=sig.signal_name,
                    timestamp=sig.timestamp,
                    log_likelihood_yes=sig.log_likelihood_yes,
                    log_likelihood_no=sig.log_likelihood_no,
                    confidence=sig.confidence * decay_factor,
                    metadata=sig.metadata,
                )
                decayed_signals.append(decayed)
            else:
                decayed_signals.append(sig)

        market_price = self._market_prices.get(condition_id, 0.5)
        p_hat, confidence = self._compute_posterior(decayed_signals, market_price)
        belief._cached_posterior = p_hat
        belief._cached_confidence = confidence
        belief.last_updated = now

        return belief

    def get_estimate(self, condition_id: str) -> Optional[float]:
        """Current estimated true probability P(YES)."""
        belief = self._beliefs.get(condition_id)
        if belief is None:
            return None
        return belief.posterior_yes

    def get_confidence(self, condition_id: str) -> Optional[float]:
        """Current model confidence in the posterior."""
        belief = self._beliefs.get(condition_id)
        if belief is None:
            return None
        return belief._cached_confidence

    def has_sufficient_signals(self, condition_id: str) -> bool:
        belief = self._beliefs.get(condition_id)
        if belief is None:
            return False
        # Count only signals above minimum confidence threshold
        valid_count = sum(
            1 for s in belief.signal_history
            if s.confidence >= self.min_signal_confidence
        )
        return valid_count >= self.min_signals

    def get_belief(self, condition_id: str) -> Optional[MarketBelief]:
        return self._beliefs.get(condition_id)

    @property
    def beliefs(self) -> dict[str, MarketBelief]:
        return self._beliefs

    # -----------------------------------------------------------------------
    # Group-aware posterior computation
    # -----------------------------------------------------------------------

    def _compute_posterior(
        self,
        signals: list[SignalUpdate],
        market_price: float,
    ) -> tuple[float, float]:
        """
        Aggregate signals using group-aware log-likelihood combination,
        then compute Bayesian posterior in log-space for numerical stability.

        Returns (p_hat, confidence).
        """
        valid = [s for s in signals if s.confidence >= self.min_signal_confidence]

        if len(valid) < self.min_signals:
            return market_price, 0.0

        # Group-aware aggregation
        log_lr_yes, log_lr_no, effective_signal_count = self._aggregate(valid)

        # Log-space Bayesian update
        # Prior: market_price (anchored to current observed price)
        clamped_price = max(0.001, min(0.999, market_price))
        log_prior_yes = math.log(clamped_price)
        log_prior_no = math.log(1.0 - clamped_price)

        log_post_yes = log_prior_yes + log_lr_yes
        log_post_no = log_prior_no + log_lr_no

        # Normalize (log-sum-exp for stability)
        log_z = _log_sum_exp(log_post_yes, log_post_no)
        p_hat = math.exp(log_post_yes - log_z)

        # Clamp posterior
        p_hat = max(self.posterior_clamp_low, min(self.posterior_clamp_high, p_hat))

        # Confidence: based on effective (deduplicated) signal count and
        # mean signal confidence — NOT on posterior magnitude
        mean_conf = sum(s.confidence for s in valid) / len(valid)
        eff_count = min(effective_signal_count, 4.0)  # saturates at 4
        confidence = mean_conf * (eff_count / 4.0)
        confidence = max(0.0, min(1.0, confidence))

        logger.debug(
            "[BAYES] p_hat=%.3f conf=%.2f eff_signals=%.1f (raw=%d)",
            p_hat, confidence, effective_signal_count, len(valid),
        )
        return p_hat, confidence

    def _aggregate(
        self, signals: list[SignalUpdate],
    ) -> tuple[float, float, float]:
        """
        Combine log-likelihoods with group deduplication.

        Within each group:
          - Sort by |log_likelihood| descending
          - Strongest signal: full confidence-weighted contribution
          - Each additional signal: INTRA_GROUP_DECAY weight

        Cross-group: sum contributions normally.

        Returns (log_lr_yes, log_lr_no, effective_signal_count).
        """
        signal_map = {s.signal_name: s for s in signals}
        accounted: set[str] = set()
        log_lr_yes = 0.0
        log_lr_no = 0.0
        eff_count = 0.0

        for group_name, members in self.signal_groups.items():
            group = [signal_map[m] for m in members if m in signal_map]
            if not group:
                continue

            # Sort by absolute log-likelihood magnitude (strongest first)
            group.sort(key=lambda s: abs(s.log_likelihood_yes), reverse=True)

            for rank, sig in enumerate(group):
                weight = 1.0 if rank == 0 else self.intra_group_decay
                adj_weight = weight * sig.confidence

                log_lr_yes += adj_weight * sig.log_likelihood_yes
                log_lr_no += adj_weight * sig.log_likelihood_no
                eff_count += adj_weight * (1.0 if rank == 0 else self.intra_group_decay)
                accounted.add(sig.signal_name)

        # Any signals not in a defined group: treat as independent
        for sig in signals:
            if sig.signal_name not in accounted:
                adj_weight = sig.confidence
                log_lr_yes += adj_weight * sig.log_likelihood_yes
                log_lr_no += adj_weight * sig.log_likelihood_no
                eff_count += adj_weight

        return log_lr_yes, log_lr_no, eff_count


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _log_sum_exp(a: float, b: float) -> float:
    """Numerically stable log(exp(a) + exp(b))."""
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))
