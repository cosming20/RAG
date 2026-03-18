"""LLM rate limiter — moving window per-minute rate limiting (Cognee pattern)."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class RateLimitConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LLM_RATE_LIMIT_")
    enabled: bool = False
    requests: int = 60  # requests per interval
    interval: float = 60.0  # seconds


@dataclass
class MovingWindowRateLimiter:
    max_requests: int = 60
    window_seconds: float = 60.0
    _timestamps: list[float] = field(default_factory=list)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            cutoff = now - self.window_seconds
            self._timestamps = [t for t in self._timestamps if t > cutoff]
            if len(self._timestamps) >= self.max_requests:
                wait_time = self._timestamps[0] - cutoff
                logger.warning("Rate limit hit, waiting %.1fs", wait_time)
                await asyncio.sleep(wait_time)
                # Re-check after sleep
                now = time.monotonic()
                cutoff = now - self.window_seconds
                self._timestamps = [t for t in self._timestamps if t > cutoff]
            self._timestamps.append(now)
