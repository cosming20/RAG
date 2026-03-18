"""Shared state store for intermediate results between agents."""

from __future__ import annotations

import json
import logging
from typing import Any

from rag_common.redis.client import RedisClient

logger = logging.getLogger(__name__)

RESULTS_KEY_PREFIX = "rag:results"
SESSION_KEY_PREFIX = "rag:sessions"
DEFAULT_TTL = 3600  # 1 hour


class StateStore:
    """Read/write intermediate results and session state in Redis."""

    def __init__(self, redis_client: RedisClient) -> None:
        self._redis = redis_client

    # ── Intermediate Results ──────────────────────────────

    async def store_result(
        self,
        task_id: str,
        stage: str,
        data: Any,
        ttl: int = DEFAULT_TTL,
    ) -> str:
        key = f"{RESULTS_KEY_PREFIX}:{task_id}:{stage}"
        payload = json.dumps(data, default=str)
        await self._redis.client.set(key, payload, ex=ttl)
        logger.debug("Stored result: %s", key)
        return key

    async def get_result(self, task_id: str, stage: str) -> Any | None:
        key = f"{RESULTS_KEY_PREFIX}:{task_id}:{stage}"
        data = await self._redis.client.get(key)
        if data is None:
            return None
        return json.loads(data)

    async def store_raw(
        self,
        task_id: str,
        data: bytes,
        ttl: int = DEFAULT_TTL,
    ) -> str:
        key = f"{RESULTS_KEY_PREFIX}:{task_id}:raw"
        # Use a separate Redis client for binary data (decode_responses=False)
        # For now, store as base64 or reference to object store
        import base64
        encoded = base64.b64encode(data).decode("ascii")
        await self._redis.client.set(key, encoded, ex=ttl)
        return key

    async def get_raw(self, task_id: str) -> bytes | None:
        key = f"{RESULTS_KEY_PREFIX}:{task_id}:raw"
        data = await self._redis.client.get(key)
        if data is None:
            return None
        import base64
        return base64.b64decode(data)

    # ── Session State ─────────────────────────────────────

    async def get_session(self, session_id: str) -> dict[str, str] | None:
        key = f"{SESSION_KEY_PREFIX}:{session_id}"
        data = await self._redis.client.hgetall(key)
        return data if data else None

    async def update_session(
        self,
        session_id: str,
        data: dict[str, str],
        ttl: int = 3600,
    ) -> None:
        key = f"{SESSION_KEY_PREFIX}:{session_id}"
        await self._redis.client.hset(key, mapping=data)
        await self._redis.client.expire(key, ttl)

    async def append_session_history(
        self,
        session_id: str,
        entry: dict[str, Any],
        ttl: int = 3600,
    ) -> None:
        key = f"{SESSION_KEY_PREFIX}:{session_id}:history"
        await self._redis.client.rpush(key, json.dumps(entry, default=str))
        await self._redis.client.expire(key, ttl)
