"""
Related Market Signal — compares pricing across semantically similar markets.

Uses keyword overlap (Jaccard similarity) on market question text to find
related markets across different events. When related markets disagree on
pricing, this signal pushes toward the consensus.
"""

from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING, Optional

import numpy as np

from src.analysis.bayesian_engine import SignalUpdate
from src.signals.base import BaseSignal

if TYPE_CHECKING:
    from src.analysis.lmsr_engine import LMSRState
    from src.feed.order_book import OrderBookState
    from src.market.models import Market

STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "will", "would", "could", "should", "may", "might", "can", "do", "does",
    "did", "has", "have", "had", "to", "of", "in", "for", "on", "at", "by",
    "with", "from", "or", "and", "not", "no", "yes", "if", "it", "its",
    "this", "that", "than", "what", "which", "who", "whom", "how", "when",
    "where", "before", "after", "between", "during", "about", "into",
    "through", "over", "under", "up", "down", "out", "more", "most",
    "other", "any", "each", "every", "all", "both", "few", "some",
})


def _tokenize(text: str) -> set[str]:
    """Extract lowercase keyword tokens, removing stop words and short tokens."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in STOP_WORDS and len(w) > 2}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class RelatedMarketSignal(BaseSignal):
    """
    Finds semantically related markets and flags pricing divergence.

    If related markets average 70% YES but the target is at 40%,
    this signal pushes toward YES (price convergence).
    """

    def __init__(
        self,
        min_similarity: float = 0.3,
        max_related: int = 5,
        min_price_divergence: float = 0.10,
    ):
        self._min_sim = min_similarity
        self._max_related = max_related
        self._min_divergence = min_price_divergence

    @property
    def name(self) -> str:
        return "related_market"

    @property
    def description(self) -> str:
        return "Related market price convergence"

    def is_applicable(self, market: Market) -> bool:
        tokens = _tokenize(market.question)
        return len(tokens) >= 3

    async def compute(
        self,
        condition_id: str,
        market: Market,
        order_book: OrderBookState,
        lmsr_state: LMSRState,
        **context,
    ) -> Optional[SignalUpdate]:
        scanner = context.get("scanner")
        if scanner is None:
            return None

        target_tokens = _tokenize(market.question)
        if len(target_tokens) < 3:
            return None

        current_price = self._get_yes_price(market)
        if current_price is None:
            return None

        # Find related markets by keyword similarity (exclude same event)
        candidates: list[tuple[float, float, str]] = []  # (similarity, price, cid)

        for cid, m in scanner.markets.items():
            if cid == condition_id:
                continue
            # Exclude same-event markets (handled by CrossMarketSignal)
            if market.event_slug and m.event_slug == market.event_slug:
                continue

            m_tokens = _tokenize(m.question)
            sim = _jaccard(target_tokens, m_tokens)
            if sim < self._min_sim:
                continue

            p = self._get_yes_price(m)
            if p is not None:
                candidates.append((sim, p, cid))

        if not candidates:
            return None

        # Take top N by similarity
        candidates.sort(key=lambda x: x[0], reverse=True)
        top = candidates[: self._max_related]

        # Weighted mean of related market prices (weighted by similarity)
        total_weight = sum(sim for sim, _, _ in top)
        if total_weight == 0:
            return None

        mean_related_price = sum(sim * p for sim, p, _ in top) / total_weight
        divergence = mean_related_price - current_price

        if abs(divergence) < self._min_divergence:
            return None

        # Signal pushes toward consensus of related markets
        signal_strength = np.log(1.0 + abs(divergence) * 2.0)
        avg_sim = total_weight / len(top)

        if divergence > 0:
            # Related markets price higher — push YES
            log_ll_yes = float(signal_strength)
            log_ll_no = float(-signal_strength)
        else:
            # Related markets price lower — push NO
            log_ll_yes = float(-signal_strength)
            log_ll_no = float(signal_strength)

        # Confidence from similarity quality and count
        confidence = min(1.0, len(top) / 3.0) * min(1.0, avg_sim / 0.5)

        return SignalUpdate(
            signal_name=self.name,
            timestamp=time.time(),
            log_likelihood_yes=log_ll_yes,
            log_likelihood_no=log_ll_no,
            confidence=max(0.01, confidence),
            metadata={
                "related_count": len(top),
                "avg_similarity": round(avg_sim, 3),
                "mean_related_price": round(mean_related_price, 4),
                "price_divergence": round(divergence, 4),
                "top_related": [
                    {"condition_id": cid[:16], "similarity": round(s, 3), "price": round(p, 4)}
                    for s, p, cid in top[:3]
                ],
            },
        )

    @staticmethod
    def _get_yes_price(market: Market) -> Optional[float]:
        for t in market.tokens:
            if t.outcome.lower() == "yes" and t.price is not None:
                return t.price
        return None
