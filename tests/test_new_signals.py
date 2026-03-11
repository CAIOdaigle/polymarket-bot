"""Tests for Phase B signals: cross-market, whale tracker, related market."""

import time

import numpy as np
import pytest

from src.analysis.lmsr_engine import LMSRState
from src.feed.order_book import OrderBookState
from src.market.models import Market, MarketToken
from src.signals.cross_market_signal import CrossMarketSignal
from src.signals.related_market_signal import RelatedMarketSignal, _tokenize, _jaccard
from src.signals.whale_tracker_signal import WhaleTrackerSignal


# ---- Helpers ----


def make_market(
    condition_id: str = "cid-1",
    question: str = "Will X happen?",
    event_slug: str = "event-1",
    yes_price: float = 0.5,
    end_date: str = "2026-12-31T00:00:00Z",
    liquidity: float = 1000.0,
) -> Market:
    return Market(
        condition_id=condition_id,
        question=question,
        slug=condition_id,
        tokens=[
            MarketToken(token_id=f"{condition_id}-yes", outcome="Yes", price=yes_price),
            MarketToken(token_id=f"{condition_id}-no", outcome="No", price=1 - yes_price),
        ],
        end_date=end_date,
        volume_24h=1000.0,
        liquidity=liquidity,
        neg_risk=False,
        tick_size=0.01,
        event_slug=event_slug,
    )


def make_order_book(trades=None) -> OrderBookState:
    ob = OrderBookState(token_id="test-token")
    ob._bids = {0.49: 100.0, 0.48: 200.0}
    ob._asks = {0.51: 100.0, 0.52: 200.0}
    if trades:
        ob.recent_trades = trades
    return ob


def make_lmsr_state() -> LMSRState:
    return LMSRState(b=100.0, q_yes=0.0, q_no=0.0, implied_price_yes=0.5, implied_price_no=0.5, confidence=0.8)


class MockScanner:
    def __init__(self, markets: dict[str, Market]):
        self.markets = markets


def trade(price=0.5, size=10.0, side="BUY", ts=None):
    return {"price": price, "size": size, "side": side, "timestamp": ts or time.time()}


# ---- Cross-Market Signal Tests ----


class TestCrossMarketSignal:
    @pytest.fixture
    def signal(self):
        return CrossMarketSignal(min_inconsistency=0.05, max_event_markets=20)

    @pytest.mark.asyncio
    async def test_no_event_slug(self, signal):
        m = make_market(event_slug="")
        assert signal.is_applicable(m) is False

    @pytest.mark.asyncio
    async def test_no_siblings_returns_none(self, signal):
        m = make_market(condition_id="solo", event_slug="lonely-event")
        scanner = MockScanner({m.condition_id: m})
        result = await signal.compute("solo", m, make_order_book(), make_lmsr_state(), scanner=scanner)
        assert result is None

    @pytest.mark.asyncio
    async def test_consistent_prices_returns_none(self, signal):
        """Two markets summing to 1.0 should produce no signal."""
        m1 = make_market(condition_id="m1", yes_price=0.6, event_slug="ev1")
        m2 = make_market(condition_id="m2", yes_price=0.4, event_slug="ev1")
        scanner = MockScanner({"m1": m1, "m2": m2})
        result = await signal.compute("m1", m1, make_order_book(), make_lmsr_state(), scanner=scanner)
        assert result is None

    @pytest.mark.asyncio
    async def test_overpriced_sum_signals_no(self, signal):
        """Three markets at 0.5 each (sum=1.5) should push NO."""
        m1 = make_market(condition_id="m1", yes_price=0.5, event_slug="ev1")
        m2 = make_market(condition_id="m2", yes_price=0.5, event_slug="ev1")
        m3 = make_market(condition_id="m3", yes_price=0.5, event_slug="ev1")
        scanner = MockScanner({"m1": m1, "m2": m2, "m3": m3})
        result = await signal.compute("m1", m1, make_order_book(), make_lmsr_state(), scanner=scanner)
        assert result is not None
        assert result.log_likelihood_no > 0  # push NO
        assert result.log_likelihood_yes < 0
        assert result.metadata["deviation"] > 0

    @pytest.mark.asyncio
    async def test_underpriced_sum_signals_yes(self, signal):
        """Two markets at 0.3 each (sum=0.6) should push YES."""
        m1 = make_market(condition_id="m1", yes_price=0.3, event_slug="ev1")
        m2 = make_market(condition_id="m2", yes_price=0.3, event_slug="ev1")
        scanner = MockScanner({"m1": m1, "m2": m2})
        result = await signal.compute("m1", m1, make_order_book(), make_lmsr_state(), scanner=scanner)
        assert result is not None
        assert result.log_likelihood_yes > 0  # push YES
        assert result.metadata["deviation"] < 0

    @pytest.mark.asyncio
    async def test_max_event_markets_filter(self, signal):
        """Events with too many markets should be skipped."""
        signal._max_event_markets = 3
        markets = {}
        for i in range(5):
            m = make_market(condition_id=f"m{i}", yes_price=0.5, event_slug="big-event")
            markets[f"m{i}"] = m
        scanner = MockScanner(markets)
        result = await signal.compute("m0", markets["m0"], make_order_book(), make_lmsr_state(), scanner=scanner)
        assert result is None

    @pytest.mark.asyncio
    async def test_no_scanner_returns_none(self, signal):
        m = make_market()
        result = await signal.compute("m1", m, make_order_book(), make_lmsr_state())
        assert result is None


# ---- Whale Tracker Signal Tests ----


class TestWhaleTrackerSignal:
    @pytest.fixture
    def signal(self):
        return WhaleTrackerSignal(
            whale_threshold_usd=500.0, volume_spike_multiplier=3.0, lookback_trades=100, min_whale_trades=1
        )

    @pytest.mark.asyncio
    async def test_no_trades_returns_none(self, signal):
        ob = make_order_book(trades=[])
        result = await signal.compute("c1", make_market(), ob, make_lmsr_state())
        assert result is None

    @pytest.mark.asyncio
    async def test_no_whale_trades_returns_none(self, signal):
        """All small trades should produce no signal."""
        trades = [trade(price=0.5, size=10.0, side="BUY") for _ in range(10)]
        ob = make_order_book(trades=trades)
        result = await signal.compute("c1", make_market(), ob, make_lmsr_state())
        assert result is None

    @pytest.mark.asyncio
    async def test_large_buy_signals_yes(self, signal):
        """A single large buy should push YES."""
        trades = [trade(price=0.5, size=10.0, side="BUY") for _ in range(10)]
        trades.append(trade(price=0.5, size=1200.0, side="BUY"))  # $600 whale
        ob = make_order_book(trades=trades)
        result = await signal.compute("c1", make_market(), ob, make_lmsr_state())
        assert result is not None
        assert result.log_likelihood_yes > 0
        assert result.metadata["whale_count"] >= 1

    @pytest.mark.asyncio
    async def test_large_sell_signals_no(self, signal):
        """A single large sell should push NO."""
        trades = [trade(price=0.5, size=10.0, side="BUY") for _ in range(10)]
        trades.append(trade(price=0.5, size=1200.0, side="SELL"))  # $600 whale sell
        ob = make_order_book(trades=trades)
        result = await signal.compute("c1", make_market(), ob, make_lmsr_state())
        assert result is not None
        assert result.log_likelihood_no > 0

    @pytest.mark.asyncio
    async def test_mixed_whales_net_direction(self, signal):
        """Net whale direction should determine signal."""
        trades = [trade(price=0.5, size=10.0, side="BUY") for _ in range(5)]
        trades.append(trade(price=0.5, size=2000.0, side="BUY"))   # $1000 buy
        trades.append(trade(price=0.5, size=1200.0, side="SELL"))   # $600 sell
        ob = make_order_book(trades=trades)
        result = await signal.compute("c1", make_market(), ob, make_lmsr_state())
        assert result is not None
        assert result.log_likelihood_yes > 0  # net buy

    @pytest.mark.asyncio
    async def test_consecutive_trades_increase_strength(self, signal):
        """Many consecutive same-direction trades should boost signal."""
        trades = [trade(price=0.5, size=1200.0, side="BUY") for _ in range(10)]
        ob = make_order_book(trades=trades)
        result = await signal.compute("c1", make_market(), ob, make_lmsr_state())
        assert result is not None
        assert result.metadata["consecutive_count"] == 10


# ---- Related Market Signal Tests ----


class TestRelatedMarketSignal:
    @pytest.fixture
    def signal(self):
        return RelatedMarketSignal(min_similarity=0.3, max_related=5, min_price_divergence=0.10)

    def test_tokenize(self):
        tokens = _tokenize("Will Bitcoin hit $100k by December 2026?")
        assert "bitcoin" in tokens
        assert "100k" in tokens
        assert "december" in tokens
        assert "will" not in tokens  # stop word
        assert "by" not in tokens  # stop word

    def test_jaccard_identical(self):
        assert _jaccard({"a", "b", "c"}, {"a", "b", "c"}) == 1.0

    def test_jaccard_disjoint(self):
        assert _jaccard({"a", "b"}, {"c", "d"}) == 0.0

    def test_jaccard_partial(self):
        result = _jaccard({"a", "b", "c"}, {"b", "c", "d"})
        assert abs(result - 0.5) < 0.01  # 2/4

    @pytest.mark.asyncio
    async def test_short_question_not_applicable(self, signal):
        m = make_market(question="Yes or no?")
        assert signal.is_applicable(m) is False

    @pytest.mark.asyncio
    async def test_no_related_returns_none(self, signal):
        m1 = make_market(condition_id="m1", question="Will Bitcoin reach 100k by December?", event_slug="ev1")
        m2 = make_market(condition_id="m2", question="Will Argentina win the World Cup?", event_slug="ev2")
        scanner = MockScanner({"m1": m1, "m2": m2})
        result = await signal.compute("m1", m1, make_order_book(), make_lmsr_state(), scanner=scanner)
        assert result is None

    @pytest.mark.asyncio
    async def test_related_markets_push_convergence(self, signal):
        """Related markets averaging higher should push YES."""
        m1 = make_market(condition_id="m1", question="Will Bitcoin reach 100k by June 2026?", yes_price=0.3, event_slug="ev1")
        m2 = make_market(condition_id="m2", question="Will Bitcoin reach 100k by December 2026?", yes_price=0.7, event_slug="ev2")
        m3 = make_market(condition_id="m3", question="Will Bitcoin hit 100k in 2026?", yes_price=0.6, event_slug="ev3")
        scanner = MockScanner({"m1": m1, "m2": m2, "m3": m3})
        result = await signal.compute("m1", m1, make_order_book(), make_lmsr_state(), scanner=scanner)
        assert result is not None
        assert result.log_likelihood_yes > 0  # related markets higher, push YES
        assert result.metadata["price_divergence"] > 0

    @pytest.mark.asyncio
    async def test_excludes_same_event(self, signal):
        """Same-event markets should be excluded (handled by CrossMarketSignal)."""
        m1 = make_market(condition_id="m1", question="Will Bitcoin reach 100k?", yes_price=0.3, event_slug="same")
        m2 = make_market(condition_id="m2", question="Will Bitcoin reach 100k by Dec?", yes_price=0.7, event_slug="same")
        scanner = MockScanner({"m1": m1, "m2": m2})
        result = await signal.compute("m1", m1, make_order_book(), make_lmsr_state(), scanner=scanner)
        assert result is None

    @pytest.mark.asyncio
    async def test_small_divergence_returns_none(self, signal):
        """Below min_price_divergence should return None."""
        m1 = make_market(condition_id="m1", question="Will Ethereum reach 5000 dollars?", yes_price=0.5, event_slug="ev1")
        m2 = make_market(condition_id="m2", question="Will Ethereum reach 5000 price target?", yes_price=0.55, event_slug="ev2")
        scanner = MockScanner({"m1": m1, "m2": m2})
        result = await signal.compute("m1", m1, make_order_book(), make_lmsr_state(), scanner=scanner)
        assert result is None  # 0.05 divergence < 0.10 threshold

    @pytest.mark.asyncio
    async def test_no_scanner_returns_none(self, signal):
        m = make_market(question="Will Bitcoin reach one hundred thousand dollars?")
        result = await signal.compute("m1", m, make_order_book(), make_lmsr_state())
        assert result is None
