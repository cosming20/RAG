"""Async Redis client wrapper with connection management."""

from __future__ import annotations

import logging

import redis.asyncio as aioredis

from rag_common.config import RedisConfig

logger = logging.getLogger(__name__)


class RedisClient:
    """Async Redis client with connection pooling."""

    def __init__(self, config: RedisConfig | None = None) -> None:
        self._config = config or RedisConfig()
        self._pool: aioredis.ConnectionPool | None = None
        self._client: aioredis.Redis | None = None

    async def connect(self) -> None:
        self._pool = aioredis.ConnectionPool.from_url(
            self._config.url,
            max_connections=self._config.max_connections,
            socket_timeout=self._config.socket_timeout,
            decode_responses=self._config.decode_responses,
        )
        self._client = aioredis.Redis(connection_pool=self._pool)
        await self._client.ping()
        logger.info("Connected to Redis at %s:%s", self._config.host, self._config.port)

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
        if self._pool:
            await self._pool.aclose()
        logger.info("Redis connection closed")

    @property
    def client(self) -> aioredis.Redis:
        if self._client is None:
            msg = "Redis client not connected. Call connect() first."
            raise RuntimeError(msg)
        return self._client

    async def __aenter__(self) -> RedisClient:
        await self.connect()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()
