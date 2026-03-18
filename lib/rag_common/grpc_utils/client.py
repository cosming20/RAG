"""gRPC client factory with retry and circuit breaker."""

from __future__ import annotations

import logging
from typing import TypeVar

import grpc

from rag_common.retry import CircuitBreaker, grpc_retry

logger = logging.getLogger(__name__)

T = TypeVar("T")


class GrpcClientFactory:
    """Creates gRPC channel and stub with retry and circuit breaker."""

    def __init__(self) -> None:
        self._channels: dict[str, grpc.aio.Channel] = {}
        self._circuit_breakers: dict[str, CircuitBreaker] = {}

    async def get_channel(self, address: str) -> grpc.aio.Channel:
        if address not in self._channels:
            self._channels[address] = grpc.aio.insecure_channel(
                address,
                options=[
                    ("grpc.keepalive_time_ms", 10000),
                    ("grpc.keepalive_timeout_ms", 5000),
                    ("grpc.keepalive_permit_without_calls", True),
                    ("grpc.max_receive_message_length", 100 * 1024 * 1024),  # 100MB
                ],
            )
        return self._channels[address]

    def get_circuit_breaker(self, target: str) -> CircuitBreaker:
        if target not in self._circuit_breakers:
            self._circuit_breakers[target] = CircuitBreaker()
        return self._circuit_breakers[target]

    async def close_all(self) -> None:
        for channel in self._channels.values():
            await channel.close()
        self._channels.clear()
