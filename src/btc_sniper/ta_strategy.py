"""Technical analysis strategy for BTC 5-minute binary markets.

Produces a composite score from 7 weighted indicators.
Positive = Up, Negative = Down. Magnitude = confidence.

Based on: https://gist.github.com/Archetapp/7680adabc48f812a561ca79d73cbac69
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from src.btc_sniper.binance_feed import Candle

logger = logging.getLogger(__name__)


@dataclass
class TAResult:
    score: float  # positive = UP, negative = DOWN
    confidence: float  # 0.0 - 1.0
    direction: str  # "UP" or "DOWN"
    window_delta_pct: float
    components: dict  # breakdown of each indicator's contribution


def _ema(prices: list[float], period: int) -> list[float]:
    """Exponential moving average."""
    if not prices or period <= 0:
        return []
    k = 2.0 / (period + 1)
    ema_vals = [prices[0]]
    for p in prices[1:]:
        ema_vals.append(p * k + ema_vals[-1] * (1 - k))
    return ema_vals


def _rsi(closes: list[float], period: int = 14) -> Optional[float]:
    """Relative Strength Index."""
    if len(closes) < period + 1:
        return None
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [d if d > 0 else 0 for d in deltas[-period:]]
    losses = [-d if d < 0 else 0 for d in deltas[-period:]]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def analyze(
    candles: list[Candle],
    window_open_price: float,
    current_price: float,
    tick_prices: list[tuple[float, float]] | None = None,
) -> TAResult:
    """Run composite TA analysis.

    Args:
        candles: Recent 1-minute candles (at least 21 needed for EMA).
        window_open_price: BTC price at start of the 5-min window.
        current_price: Latest BTC price.
        tick_prices: (timestamp, price) tuples from 2s polling.

    Returns:
        TAResult with composite score and confidence.
    """
    score = 0.0
    components = {}

    # --- 1. Window Delta (weight 5-7) — THE dominant signal ---
    if window_open_price > 0:
        window_pct = (current_price - window_open_price) / window_open_price * 100
    else:
        window_pct = 0.0

    abs_pct = abs(window_pct)
    if abs_pct > 0.10:
        w = 7.0
    elif abs_pct > 0.02:
        w = 5.0
    elif abs_pct > 0.005:
        w = 3.0
    elif abs_pct > 0.001:
        w = 1.0
    else:
        w = 0.0

    sign = 1.0 if window_pct >= 0 else -1.0
    score += sign * w
    components["window_delta"] = round(sign * w, 2)

    # Need enough candles for remaining indicators
    closes = [c.close for c in candles]
    volumes = [c.volume for c in candles]

    # --- 2. Micro Momentum (weight 2) — last 2 candles ---
    if len(closes) >= 2:
        micro = closes[-1] - closes[-2]
        micro_sign = 1.0 if micro > 0 else -1.0
        score += micro_sign * 2.0
        components["micro_momentum"] = round(micro_sign * 2.0, 2)
    else:
        components["micro_momentum"] = 0.0

    # --- 3. Acceleration (weight 1.5) — momentum building or fading ---
    if len(closes) >= 3:
        move_latest = closes[-1] - closes[-2]
        move_prior = closes[-2] - closes[-3]
        if abs(move_latest) > abs(move_prior):
            # Accelerating
            acc_sign = 1.0 if move_latest > 0 else -1.0
            score += acc_sign * 1.5
            components["acceleration"] = round(acc_sign * 1.5, 2)
        else:
            components["acceleration"] = 0.0
    else:
        components["acceleration"] = 0.0

    # --- 4. EMA Crossover 9/21 (weight 1) ---
    if len(closes) >= 21:
        ema9 = _ema(closes, 9)
        ema21 = _ema(closes, 21)
        if ema9[-1] > ema21[-1]:
            score += 1.0
            components["ema_crossover"] = 1.0
        else:
            score -= 1.0
            components["ema_crossover"] = -1.0
    else:
        components["ema_crossover"] = 0.0

    # --- 5. RSI 14-period (weight 1-2) ---
    rsi_val = _rsi(closes)
    if rsi_val is not None:
        if rsi_val > 75:
            score += 2.0  # overbought = bullish momentum still
            components["rsi"] = 2.0
        elif rsi_val < 25:
            score -= 2.0  # oversold = bearish momentum still
            components["rsi"] = -2.0
        elif rsi_val > 60:
            score += 1.0
            components["rsi"] = 1.0
        elif rsi_val < 40:
            score -= 1.0
            components["rsi"] = -1.0
        else:
            components["rsi"] = 0.0
    else:
        components["rsi"] = 0.0

    # --- 6. Volume Surge (weight 1) — confirms direction ---
    if len(volumes) >= 6:
        recent_avg = sum(volumes[-3:]) / 3
        prior_avg = sum(volumes[-6:-3]) / 3
        if prior_avg > 0 and recent_avg > prior_avg * 1.5:
            # Volume surge — confirms whatever direction price is going
            vol_sign = 1.0 if closes[-1] > closes[-3] else -1.0
            score += vol_sign * 1.0
            components["volume_surge"] = round(vol_sign * 1.0, 2)
        else:
            components["volume_surge"] = 0.0
    else:
        components["volume_surge"] = 0.0

    # --- 7. Real-Time Tick Trend (weight 2) ---
    if tick_prices and len(tick_prices) >= 5:
        ups = 0
        downs = 0
        for i in range(1, len(tick_prices)):
            if tick_prices[i][1] > tick_prices[i - 1][1]:
                ups += 1
            elif tick_prices[i][1] < tick_prices[i - 1][1]:
                downs += 1
        total_moves = ups + downs
        if total_moves > 0:
            up_pct = ups / total_moves
            tick_move_pct = (tick_prices[-1][1] - tick_prices[0][1]) / tick_prices[0][1] * 100
            if up_pct >= 0.60 and abs(tick_move_pct) > 0.005:
                score += 2.0
                components["tick_trend"] = 2.0
            elif up_pct <= 0.40 and abs(tick_move_pct) > 0.005:
                score -= 2.0
                components["tick_trend"] = -2.0
            else:
                components["tick_trend"] = 0.0
        else:
            components["tick_trend"] = 0.0
    else:
        components["tick_trend"] = 0.0

    # --- Confidence: divide by 7 (not 10) since TA is noisy at 5-min scale ---
    confidence = min(abs(score) / 7.0, 1.0)
    direction = "UP" if score >= 0 else "DOWN"

    return TAResult(
        score=round(score, 2),
        confidence=round(confidence, 4),
        direction=direction,
        window_delta_pct=round(window_pct, 6),
        components=components,
    )
