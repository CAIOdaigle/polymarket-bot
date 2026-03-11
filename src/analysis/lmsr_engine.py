"""
LMSR (Logarithmic Market Scoring Rule) Engine.

Implements the core formulas from the LMSR pricing document:
  1. Cost function:  C(q) = b * ln( sum( exp(q_i / b) ) )
  2. Price function:  p_i(q) = exp(q_i/b) / sum(exp(q_j/b))   (softmax)
  3. Trade cost:  C(q + delta*e_i) - C(q)
  4. Max loss:  L_max = b * ln(n)

Since Polymarket uses a CLOB (not LMSR), this engine serves as an
analytical overlay: it estimates an equivalent 'b' parameter from the
CLOB order book and detects mispricings.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

from src.config import LMSRConfig
from src.utils.math_helpers import clamp, log_sum_exp, softmax

logger = logging.getLogger(__name__)


@dataclass
class LMSRState:
    b: float
    q_yes: float
    q_no: float
    implied_price_yes: float
    implied_price_no: float
    confidence: float  # 0-1, based on order book depth


class LMSREngine:
    def __init__(self, config: LMSRConfig):
        self.default_b = config.default_b
        self.min_b = config.min_b
        self.max_b = config.max_b

    # ---- Core LMSR functions (numerically stable) ----

    @staticmethod
    def cost(q: np.ndarray, b: float) -> float:
        """C(q) = b * ln(sum(exp(q_i / b)))"""
        return b * log_sum_exp(q / b)

    @staticmethod
    def price(q: np.ndarray, b: float) -> np.ndarray:
        """p_i(q) = exp(q_i/b) / sum(exp(q_j/b))  — the softmax / price function."""
        return softmax(q / b)

    @staticmethod
    def trade_cost(q: np.ndarray, b: float, outcome_idx: int, delta: float) -> float:
        """Cost to buy delta shares of outcome_idx: C(q + delta*e_i) - C(q)."""
        q_new = q.copy()
        q_new[outcome_idx] += delta
        return LMSREngine.cost(q_new, b) - LMSREngine.cost(q, b)

    @staticmethod
    def max_loss(b: float, n_outcomes: int = 2) -> float:
        """Maximum market maker loss: L_max = b * ln(n)."""
        return b * np.log(n_outcomes)

    # ---- b estimation from CLOB order book ----

    def estimate_b_from_orderbook(
        self,
        bids: list[tuple[float, float]],
        asks: list[tuple[float, float]],
        current_price_yes: float,
    ) -> float:
        """
        Estimate LMSR liquidity parameter b from order book depth.

        Walks the ask side to build an empirical price-impact curve, then
        finds the b that minimizes squared error vs. the LMSR theoretical curve.
        """
        if not asks or len(asks) < 2:
            return self.default_b

        p = clamp(current_price_yes, 0.01, 0.99)

        # Build empirical price impact curve from asks (sorted by price asc)
        sorted_asks = sorted(asks, key=lambda x: x[0])
        empirical: list[tuple[float, float]] = []
        cumulative = 0.0
        for price, size in sorted_asks:
            cumulative += size
            empirical.append((cumulative, price))

        if len(empirical) < 2:
            return self.default_b

        def objective(b_cand: float) -> float:
            if b_cand <= 0:
                return float("inf")
            q_yes_init = b_cand * np.log(p / (1 - p))
            error = 0.0
            for delta, emp_price in empirical:
                q_after = np.array([q_yes_init + delta, 0.0])
                theo_price = float(softmax(q_after / b_cand)[0])
                error += (theo_price - emp_price) ** 2
            return error

        result = minimize_scalar(objective, bounds=(self.min_b, self.max_b), method="bounded")
        estimated = result.x if result.success else self.default_b
        return float(clamp(estimated, self.min_b, self.max_b))

    def compute_state(
        self,
        bids: list[tuple[float, float]],
        asks: list[tuple[float, float]],
        mid_price_yes: float,
    ) -> LMSRState:
        """Compute full LMSR state for a binary market."""
        b = self.estimate_b_from_orderbook(bids, asks, mid_price_yes)

        p = clamp(mid_price_yes, 0.01, 0.99)
        q_yes = b * np.log(p / (1 - p))
        q_no = 0.0

        prices = self.price(np.array([q_yes, q_no]), b)

        # Confidence scales with order book depth
        total_depth = sum(s for _, s in bids) + sum(s for _, s in asks)
        confidence = clamp(total_depth / 1000.0, 0.0, 1.0)

        # If b hit the max bound, the fit is unreliable — reduce confidence
        if b >= self.max_b * 0.99:
            confidence = min(confidence, 0.1)

        return LMSRState(
            b=b,
            q_yes=float(q_yes),
            q_no=float(q_no),
            implied_price_yes=float(prices[0]),
            implied_price_no=float(prices[1]),
            confidence=confidence,
        )
