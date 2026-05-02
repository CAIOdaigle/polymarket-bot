"""Anomaly detectors — small pure functions that scan recent trades and
report whether something looks structurally wrong.

Each detector takes a list of trade rows from `bot.db` and returns a
DetectorResult with severity and evidence. The hypothesis ranker combines
detector outputs into a small set of plausible explanations.

Severity levels:
  - info:  worth noting but no action
  - warn:  surface on dashboard, log incident
  - halt:  kill switch — runner stops placing trades until cleared
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Callable

# A trade row from bot.db, keyed for our use:
#   {'order_id', 'price', 'size', 'pnl_usd', 'outcome', 'confidence',
#    'spot_open', 'spot_close', 'placed_at', 'asset'}
Trade = dict[str, Any]


@dataclass
class DetectorResult:
    name: str
    fired: bool
    severity: str  # 'info' | 'warn' | 'halt'
    message: str
    evidence: dict = field(default_factory=dict)


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def _asset_of(order_id: str) -> str:
    parts = (order_id or "").split("-")
    if len(parts) >= 4 and not parts[1].isdigit():
        return parts[1].upper()
    return "BTC"


def _bet_size(t: Trade) -> float:
    return (t.get("price") or 0) * (t.get("size") or 0)


# ────────────────────────────────────────────────────────────────────────────
# Detectors
# ────────────────────────────────────────────────────────────────────────────

def detect_price_quantization(trades: list[Trade]) -> DetectorResult:
    """Distinct-entry-price ratio. Real CLOB asks vary; if 70%+ of entries
    cluster at the same handful of prices, the price source is suspect."""
    name = "price_quantization"
    if len(trades) < 30:
        return DetectorResult(name, False, "info", "not enough trades", {"n": len(trades)})
    prices = [round(t["price"], 4) for t in trades if t.get("price")]
    n = len(prices)
    distinct = len(set(prices))
    ratio = distinct / n if n else 0
    fired = ratio < 0.30
    severity = "halt" if ratio < 0.15 else ("warn" if fired else "info")
    return DetectorResult(
        name, fired, severity,
        f"{distinct} distinct prices in last {n} trades ({ratio*100:.0f}%)",
        {"distinct_ratio": ratio, "distinct_count": distinct, "sample": n,
         "top_prices": _top_values(prices, 5)},
    )


def detect_win_rate_impossible(trades: list[Trade]) -> DetectorResult:
    """Real BTC/ETH 5-min binary sniper edge tops out near 55-62%.
    A sustained win rate beyond 70% over 30+ trades implies measurement error."""
    name = "win_rate_impossible"
    resolved = [t for t in trades if t.get("outcome") in ("WIN", "LOSS")]
    if len(resolved) < 30:
        return DetectorResult(name, False, "info", "not enough resolved trades", {"n": len(resolved)})
    wins = sum(1 for t in resolved if t["outcome"] == "WIN")
    rate = wins / len(resolved)
    fired = rate > 0.70
    severity = "halt" if rate > 0.85 else ("warn" if fired else "info")
    return DetectorResult(
        name, fired, severity,
        f"win rate {rate*100:.1f}% over last {len(resolved)} trades (research ceiling ~62%)",
        {"win_rate": rate, "wins": wins, "losses": len(resolved) - wins, "n": len(resolved)},
    )


def detect_bankroll_growth(trades: list[Trade]) -> DetectorResult:
    """If bet sizes are doubling faster than 24h, exponential compounding is
    happening on top of an unverified edge — a classic blow-up risk."""
    name = "bankroll_growth"
    if len(trades) < 20:
        return DetectorResult(name, False, "info", "not enough trades", {"n": len(trades)})
    # Compare median bet size in oldest 10 vs newest 10 of the supplied window
    ordered = sorted(trades, key=lambda t: t.get("placed_at") or 0)
    old_med = statistics.median(_bet_size(t) for t in ordered[:10])
    new_med = statistics.median(_bet_size(t) for t in ordered[-10:])
    growth = new_med / old_med if old_med > 0 else float("inf")
    span_h = ((ordered[-1].get("placed_at") or 0) - (ordered[0].get("placed_at") or 0)) / 3600
    # Doubling time in hours, assuming compound growth
    doubling_h = span_h / max(0.001, _log2(growth)) if growth > 1 else float("inf")
    fired = doubling_h < 24 and growth > 4
    severity = "halt" if doubling_h < 8 else ("warn" if fired else "info")
    return DetectorResult(
        name, fired, severity,
        f"median bet ${old_med:.2f} -> ${new_med:.2f} over {span_h:.1f}h "
        f"(doubling every {doubling_h:.1f}h)" if span_h > 0
        else "insufficient time span",
        {"old_median_bet": old_med, "new_median_bet": new_med,
         "growth_factor": growth, "doubling_hours": doubling_h, "span_hours": span_h},
    )


def detect_outlier_concentration(trades: list[Trade]) -> DetectorResult:
    """If the top-3 trades by PnL drive >50% of total PnL, the strategy is
    fragile — strip those 3 and the edge disappears. Note this is not always
    a problem (could be a real heavy-tailed distribution), but combined with
    other detectors it points to phantom liquidity windfalls."""
    name = "outlier_concentration"
    pnls = sorted([t.get("pnl_usd") or 0 for t in trades], reverse=True)
    if len(pnls) < 30:
        return DetectorResult(name, False, "info", "not enough trades", {"n": len(pnls)})
    total = sum(pnls)
    if total <= 0:
        return DetectorResult(name, False, "info", "non-positive total", {"total_pnl": total})
    top3 = sum(pnls[:3])
    ratio = top3 / total
    fired = ratio > 0.50
    severity = "warn" if ratio > 0.70 else ("info" if fired else "info")
    # outlier concentration alone is not a halt-level issue
    return DetectorResult(
        name, fired, severity,
        f"top 3 trades = {ratio*100:.0f}% of total PnL (${top3:.2f} of ${total:.2f})",
        {"top3_share": ratio, "top3_pnl": top3, "total_pnl": total},
    )


def detect_wins_on_noise(trades: list[Trade]) -> DetectorResult:
    """If winning trades systematically resolve on tiny spot moves (<0.05%),
    the signal isn't predicting — it's relying on asymmetric pricing.
    That's only profitable if the prices are real."""
    name = "wins_on_noise"
    wins = []
    for t in trades:
        if t.get("outcome") != "WIN":
            continue
        so, sc = t.get("spot_open"), t.get("spot_close")
        if not (so and sc):
            continue
        wins.append(abs((sc - so) / so) * 100)
    if len(wins) < 20:
        return DetectorResult(name, False, "info", "not enough wins", {"n": len(wins)})
    median_move = statistics.median(wins)
    fired = median_move < 0.05  # 5 basis points
    severity = "warn" if fired else "info"
    return DetectorResult(
        name, fired, severity,
        f"median |spot move| on {len(wins)} wins is {median_move:.4f}% (signal predicts noise)",
        {"median_move_pct": median_move, "n_wins": len(wins)},
    )


def detect_calibration_gap(trades: list[Trade], min_per_bucket: int = 20) -> DetectorResult:
    """For each calibration bucket with enough samples, compare the bucket's
    midpoint stated confidence to the empirical win rate. A gap > 20pp says
    the TA confidence isn't tracking the real probability — Kelly is wrong."""
    name = "calibration_gap"
    buckets = [(0.00, 0.30), (0.30, 0.50), (0.50, 0.70), (0.70, 0.90), (0.90, 1.01)]
    biggest_gap = 0.0
    biggest_bucket = None
    details = []
    for lo, hi in buckets:
        sub = [t for t in trades
               if t.get("confidence") is not None
               and lo <= t["confidence"] < hi
               and t.get("outcome") in ("WIN", "LOSS")]
        if len(sub) < min_per_bucket:
            continue
        wins = sum(1 for t in sub if t["outcome"] == "WIN")
        empirical = wins / len(sub)
        stated_mid = (lo + hi) / 2
        gap = abs(empirical - stated_mid)
        details.append({"bucket": [lo, hi], "n": len(sub), "empirical": empirical,
                        "stated_mid": stated_mid, "gap": gap})
        if gap > biggest_gap:
            biggest_gap, biggest_bucket = gap, (lo, hi)
    fired = biggest_gap > 0.20
    severity = "warn" if fired else "info"
    msg = (f"largest calibration gap: bucket {biggest_bucket} "
           f"({biggest_gap*100:.1f}pp)" if biggest_bucket else "no eligible buckets")
    return DetectorResult(name, fired, severity, msg,
                          {"max_gap": biggest_gap, "buckets": details})


# All detectors registered as a list so callers can iterate.
ALL_DETECTORS: list[Callable[[list[Trade]], DetectorResult]] = [
    detect_price_quantization,
    detect_win_rate_impossible,
    detect_bankroll_growth,
    detect_outlier_concentration,
    detect_wins_on_noise,
    detect_calibration_gap,
]


def _top_values(values: list, k: int) -> list[dict]:
    from collections import Counter
    counts = Counter(values).most_common(k)
    total = len(values)
    return [{"value": v, "count": c, "pct": round(c / total * 100, 1)} for v, c in counts]


def _log2(x: float) -> float:
    import math
    return math.log2(x) if x > 0 else float("-inf")
