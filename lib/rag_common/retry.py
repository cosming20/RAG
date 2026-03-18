"""Retry policies and circuit breaker for inter-agent communication."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import grpc
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from rag_common.errors import CircuitOpenError


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


# gRPC status codes that are safe to retry
RETRYABLE_GRPC_CODES = {
    grpc.StatusCode.UNAVAILABLE,
    grpc.StatusCode.DEADLINE_EXCEEDED,
    grpc.StatusCode.RESOURCE_EXHAUSTED,
    grpc.StatusCode.ABORTED,
}


@dataclass
class CircuitBreaker:
    """Circuit breaker for protecting inter-agent gRPC calls.

    Opens after `failure_threshold` consecutive failures.
    Transitions to half-open after `recovery_timeout` seconds.
    Closes again on first success in half-open state.
    Only one probe is allowed through in half-open state.
    """

    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    state: CircuitState = field(default=CircuitState.CLOSED)
    failure_count: int = field(default=0)
    last_failure_time: float = field(default=0.0)
    _half_open_permit: bool = field(default=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def call(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        async with self._lock:
            self._check_state_transition()
            if self.state == CircuitState.OPEN:
                raise CircuitOpenError(target=str(func))
            if self.state == CircuitState.HALF_OPEN:
                if self._half_open_permit:
                    raise CircuitOpenError(target=str(func))
                self._half_open_permit = True

        try:
            result = await func(*args, **kwargs)
        except Exception:
            await self._record_failure()
            raise
        else:
            await self._record_success()
            return result

    def _check_state_transition(self) -> None:
        if self.state == CircuitState.OPEN:
            elapsed = time.monotonic() - self.last_failure_time
            if elapsed >= self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                self._half_open_permit = False

    async def _record_failure(self) -> None:
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.monotonic()
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                self._half_open_permit = False
            elif self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN

    async def _record_success(self) -> None:
        async with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.CLOSED
                self._half_open_permit = False
            self.failure_count = 0


def grpc_retry(max_attempts: int = 3) -> Any:
    """Retry decorator for gRPC calls. Only retries transient gRPC errors."""
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential_jitter(initial=0.5, max=10, jitter=2),
        retry=retry_if_exception_type(grpc.aio.AioRpcError),
        reraise=True,
    )
