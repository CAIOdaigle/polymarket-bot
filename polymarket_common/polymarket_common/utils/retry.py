from __future__ import annotations

import asyncio
import functools
import logging
from typing import Callable, Type

logger = logging.getLogger(__name__)


def async_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: tuple[Type[Exception], ...] = (Exception,),
) -> Callable:
    """Decorator: retry an async function with exponential backoff."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt == max_retries:
                        break
                    delay = min(base_delay * (2**attempt), max_delay)
                    logger.warning(
                        "Retry %d/%d for %s after %.1fs: %s",
                        attempt + 1,
                        max_retries,
                        func.__name__,
                        delay,
                        exc,
                    )
                    await asyncio.sleep(delay)
            raise last_exc  # type: ignore[misc]

        return wrapper

    return decorator
