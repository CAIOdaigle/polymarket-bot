"""Empirical confidence calibration — maps stated TA confidence to observed win rate.

THE PROBLEM: TA confidence is `abs(score) / 7.0`, a heuristic normalization.
It's not a calibrated probability. In our dry-run data, stated confidence of
0.98 corresponded to an ACTUAL 43% win rate. Kelly was sizing as if the bot
had 98% edge when reality was a coin flip.

THIS MODULE: provides a calibrator that looks up a stated-confidence bucket
in a JSON file populated from historical wins/losses. The runner calls
`calibrate()` before Kelly to get the true probability to size against.

Falls back to a hard ceiling (`default_cap`) when no calibration data exists
or the bucket is too sparse — ensures we never over-size on fake confidence.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_BUCKETS: list[tuple[float, float]] = [
    (0.00, 0.30),
    (0.30, 0.50),
    (0.50, 0.70),
    (0.70, 0.90),
    (0.90, 1.01),  # upper bound exclusive of 1.00 exactly would miss conf=1.0
]

# If we have fewer than this many observations in a bucket, we refuse to
# return a calibrated probability — too noisy to trust. The runner treats
# None as "no edge / skip trade."
MIN_SAMPLES_PER_BUCKET = 20


@dataclass
class Bucket:
    asset: str
    stated_min: float
    stated_max: float
    empirical: float  # observed win rate
    n: int


class ConfidenceCalibrator:
    """Loads a calibration.json file and maps stated confidence -> empirical probability."""

    def __init__(
        self,
        calibration_path: Path,
        default_cap: float = 0.55,
        min_samples: int = MIN_SAMPLES_PER_BUCKET,
    ):
        self._path = calibration_path
        self._default_cap = default_cap
        self._min_samples = min_samples
        # keyed by asset -> list[Bucket]
        self._buckets: dict[str, list[Bucket]] = {}
        self._loaded_at: float = 0.0
        self.reload()

    def reload(self) -> None:
        """Re-read calibration.json from disk. Safe to call frequently."""
        self._buckets = {}
        if not self._path.exists():
            logger.info(
                "No calibration file at %s — using default_cap=%.2f",
                self._path, self._default_cap,
            )
            return
        try:
            data = json.loads(self._path.read_text())
        except Exception as e:
            logger.warning("Could not parse %s: %s — using default_cap", self._path, e)
            return
        for asset_key, raw_buckets in (data.get("by_asset") or {}).items():
            self._buckets[asset_key.upper()] = [
                Bucket(
                    asset=asset_key.upper(),
                    stated_min=float(b["stated_min"]),
                    stated_max=float(b["stated_max"]),
                    empirical=float(b["empirical"]),
                    n=int(b["n"]),
                )
                for b in raw_buckets
            ]
        self._loaded_at = time.time()
        logger.info(
            "Calibration loaded: %s (min_samples=%d)",
            {k: len(v) for k, v in self._buckets.items()}, self._min_samples,
        )

    def calibrate(self, asset: str, stated_conf: float) -> Optional[float]:
        """Map stated confidence -> empirical probability for this asset.

        Returns:
          - Empirical win-rate of the matching bucket, IF:
              * bucket has >= min_samples observations
          - `default_cap` otherwise (conservative — no calibration data yet)

        Never returns more than `default_cap` until a bucket's empirical
        value is proven above it with enough samples.
        """
        asset_u = asset.upper()
        buckets = self._buckets.get(asset_u, [])
        for b in buckets:
            if b.stated_min <= stated_conf < b.stated_max:
                if b.n >= self._min_samples:
                    return b.empirical
                # Not enough samples yet — fall through to cap
                break
        # No bucket found or too sparse — return the conservative default.
        return min(stated_conf, self._default_cap)


def build_calibration_from_db(
    db_path: Path,
    output_path: Path,
    buckets: list[tuple[float, float]] = DEFAULT_BUCKETS,
) -> dict:
    """Scan historical trades in bot.db and write calibration.json.

    Groups trades by (asset, confidence bucket) and computes the empirical
    win rate in each. Only considers trades from the real-price pipeline
    (estimated_price IS NOT NULL). Returns the written dict.
    """
    if not db_path.exists():
        logger.warning("Cannot build calibration: DB missing at %s", db_path)
        return {}

    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            """
            SELECT order_id, confidence, outcome
            FROM orders
            WHERE order_id LIKE 'sniper-%'
              AND estimated_price IS NOT NULL
              AND outcome IN ('WIN', 'LOSS')
              AND confidence IS NOT NULL
            """
        ).fetchall()
    finally:
        conn.close()

    # Parse asset from order_id: "sniper-<asset>-<ts>-<strategy>"
    by_asset: dict[str, list[tuple[float, str]]] = {}
    for order_id, conf, outcome in rows:
        parts = (order_id or "").split("-")
        if len(parts) < 4 or parts[1].isdigit():
            # Legacy row without asset segment — treat as BTC
            asset = "BTC"
        else:
            asset = parts[1].upper()
        by_asset.setdefault(asset, []).append((float(conf), outcome))

    result: dict = {
        "generated_at": time.time(),
        "total_trades": len(rows),
        "by_asset": {},
    }
    for asset, trades in by_asset.items():
        buckets_out = []
        for lo, hi in buckets:
            sample = [o for c, o in trades if lo <= c < hi]
            n = len(sample)
            wins = sum(1 for o in sample if o == "WIN")
            empirical = wins / n if n else 0.0
            buckets_out.append({
                "stated_min": round(lo, 4),
                "stated_max": round(hi, 4),
                "empirical": round(empirical, 4),
                "n": n,
            })
        result["by_asset"][asset] = buckets_out

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2))
    tmp.replace(output_path)  # atomic
    logger.info(
        "Wrote calibration to %s: %d trades across %d assets",
        output_path, len(rows), len(by_asset),
    )
    return result
