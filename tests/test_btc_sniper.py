"""Tests for BTC 5-minute sniper strategy."""

import pytest

from src.btc_sniper.binance_feed import Candle
from src.btc_sniper.ta_strategy import TAResult, analyze, _ema, _rsi
from src.btc_sniper.sniper import _estimate_token_price, _next_window_ts


class TestEMA:
    def test_basic_ema(self):
        prices = [10.0, 11.0, 12.0, 11.5, 12.5]
        result = _ema(prices, 3)
        assert len(result) == 5
        assert result[0] == 10.0

    def test_empty(self):
        assert _ema([], 3) == []


class TestRSI:
    def test_all_gains(self):
        closes = list(range(1, 20))
        rsi = _rsi(closes, 14)
        assert rsi == 100.0

    def test_insufficient_data(self):
        assert _rsi([1.0, 2.0], 14) is None


class TestTokenPricing:
    def test_coin_flip(self):
        assert _estimate_token_price(0.002) == 0.50

    def test_slight_lean(self):
        assert _estimate_token_price(0.01) == 0.55

    def test_moderate(self):
        assert _estimate_token_price(0.03) == 0.65

    def test_strong(self):
        assert _estimate_token_price(0.08) == 0.80

    def test_decisive(self):
        assert _estimate_token_price(0.12) == 0.92

    def test_extreme(self):
        price = _estimate_token_price(0.20)
        assert 0.92 < price <= 0.97


class TestWindowTimestamp:
    def test_aligned_to_300(self):
        ts = _next_window_ts()
        assert ts % 300 == 0


def _make_candles(closes: list[float], volumes: list[float] | None = None) -> list[Candle]:
    """Helper to create candles from close prices."""
    if volumes is None:
        volumes = [100.0] * len(closes)
    return [
        Candle(
            open_time=i * 60000,
            open=c - 0.5,
            high=c + 1.0,
            low=c - 1.0,
            close=c,
            volume=v,
            close_time=(i + 1) * 60000 - 1,
        )
        for i, (c, v) in enumerate(zip(closes, volumes))
    ]


class TestAnalyze:
    def test_strong_up_signal(self):
        """Clear upward move should produce positive score."""
        candles = _make_candles([100 + i * 0.1 for i in range(25)])
        result = analyze(candles, window_open_price=100.0, current_price=102.5)
        assert result.direction == "UP"
        assert result.score > 0
        assert result.confidence > 0

    def test_strong_down_signal(self):
        """Clear downward move should produce negative score."""
        candles = _make_candles([100 - i * 0.1 for i in range(25)])
        result = analyze(candles, window_open_price=100.0, current_price=97.5)
        assert result.direction == "DOWN"
        assert result.score < 0
        assert result.confidence > 0

    def test_window_delta_dominates(self):
        """Window delta with 0.15% move should give high weight."""
        candles = _make_candles([100.0] * 25)  # flat candles
        # But current price is clearly up
        result = analyze(candles, window_open_price=100.0, current_price=100.15)
        assert result.components["window_delta"] == 7.0  # weight 7 for >0.10%

    def test_minimal_data(self):
        """Should not crash with very few candles."""
        candles = _make_candles([100.0])
        result = analyze(candles, window_open_price=100.0, current_price=100.0)
        assert isinstance(result, TAResult)

    def test_tick_trend_bullish(self):
        """Tick prices trending up should contribute positively."""
        candles = _make_candles([100.0] * 25)
        ticks = [(i, 100.0 + i * 0.01) for i in range(10)]  # steady up
        result = analyze(candles, 100.0, 100.09, tick_prices=ticks)
        assert result.components["tick_trend"] == 2.0

    def test_volume_surge(self):
        """Volume spike should confirm price direction."""
        volumes = [100] * 3 + [500, 500, 500]  # 5x surge
        closes = [100, 100, 100, 101, 102, 103]  # going up
        candles = _make_candles(closes, volumes)
        result = analyze(candles, 100.0, 103.0)
        assert result.components["volume_surge"] == 1.0

    def test_confidence_capped_at_1(self):
        """Even with extreme score, confidence shouldn't exceed 1.0."""
        candles = _make_candles([100 + i for i in range(25)])
        result = analyze(candles, window_open_price=100.0, current_price=125.0)
        assert result.confidence <= 1.0
