from __future__ import annotations

import asyncio
import time


class TokenBucket:
    """Simple token-bucket rate limiter."""

    def __init__(self, rate: float, capacity: float):
        self._rate = rate  # tokens per second
        self._capacity = capacity
        self._tokens = capacity
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
        self._last_refill = now

    async def acquire(self, tokens: float = 1.0) -> None:
        async with self._lock:
            while True:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                wait = (tokens - self._tokens) / self._rate
                await asyncio.sleep(wait)


class RateLimiter:
    """Dual-bucket rate limiter for Polymarket API."""

    def __init__(
        self,
        public_rate: float = 1.0,
        public_capacity: float = 60.0,
        orders_rate: float = 5.0,
        orders_capacity: float = 300.0,
    ) -> None:
        self.public = TokenBucket(rate=public_rate, capacity=public_capacity)
        self.orders = TokenBucket(rate=orders_rate, capacity=orders_capacity)
