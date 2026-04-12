"""Append-only trade log for analysis."""

from __future__ import annotations

import csv
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

LOG_PATH = Path("data/trades.csv")

HEADERS = [
    "timestamp",
    "order_id",
    "condition_id",
    "market_question",
    "side",
    "price",
    "size_shares",
    "size_usd",
    "edge",
    "kelly_fraction",
    "p_hat",
    "market_price",
    "b_estimate",
    "confidence",
    "status",
    "dry_run",
]


class TradeLog:
    def __init__(self, path: Optional[Path] = None):
        self._path = path or LOG_PATH
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            with open(self._path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(HEADERS)

    def log(
        self,
        order_id: str,
        condition_id: str,
        market_question: str,
        side: str,
        price: float,
        size_shares: float,
        size_usd: float,
        edge: float,
        kelly_fraction: float,
        p_hat: float,
        market_price: float,
        b_estimate: float,
        confidence: float,
        status: str,
        dry_run: bool,
    ) -> None:
        try:
            with open(self._path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        datetime.now(timezone.utc).isoformat(),
                        order_id,
                        condition_id,
                        market_question,
                        side,
                        f"{price:.6f}",
                        f"{size_shares:.4f}",
                        f"{size_usd:.2f}",
                        f"{edge:.6f}",
                        f"{kelly_fraction:.6f}",
                        f"{p_hat:.6f}",
                        f"{market_price:.6f}",
                        f"{b_estimate:.2f}",
                        f"{confidence:.4f}",
                        status,
                        str(dry_run),
                    ]
                )
        except Exception:
            logger.exception("Failed to write trade log")
