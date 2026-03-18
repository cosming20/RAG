"""Task claim/complete/fail protocol via Redis."""

from __future__ import annotations

import json
import logging
import time

from rag_common.models.task import Task, TaskStatus
from rag_common.redis.client import RedisClient

logger = logging.getLogger(__name__)

TASK_KEY_PREFIX = "rag:tasks"


class TaskQueue:
    """Manages task state transitions in Redis."""

    def __init__(self, redis_client: RedisClient) -> None:
        self._redis = redis_client

    async def create_task(self, task: Task) -> None:
        key = f"{TASK_KEY_PREFIX}:{task.task_id}"
        await self._redis.client.hset(key, mapping={
            "status": task.status,
            "agent_id": task.agent_id,
            "attempt": str(task.attempt),
            "max_attempts": str(task.max_attempts),
            "pipeline_depth": str(task.pipeline_depth),
            "created_at": task.created_at.isoformat(),
            "updated_at": task.updated_at.isoformat(),
            "deadline": task.deadline.isoformat() if task.deadline else "",
            "trace_id": task.trace_id,
            "payload_ref": task.payload_ref,
            "error": task.error or "",
        })
        # TTL: 24h after creation
        await self._redis.client.expire(key, 86400)

    async def claim_task(self, task_id: str, agent_id: str) -> bool:
        """Atomically claim a task. Returns True if claimed, False if already taken."""
        key = f"{TASK_KEY_PREFIX}:{task_id}"
        # Use Lua script for atomic check-and-set
        lua_script = """
        local status = redis.call('HGET', KEYS[1], 'status')
        if status == 'pending' then
            redis.call('HSET', KEYS[1], 'status', ARGV[1], 'agent_id', ARGV[2], 'updated_at', ARGV[3])
            return 1
        end
        return 0
        """
        result = await self._redis.client.eval(
            lua_script, 1, key,
            TaskStatus.CLAIMED, agent_id, str(time.time()),
        )
        if result == 1:
            logger.debug("Task %s claimed by %s", task_id, agent_id)
            return True
        logger.debug("Task %s already claimed, skipping", task_id)
        return False

    async def complete_task(self, task_id: str, result_ref: str = "") -> None:
        key = f"{TASK_KEY_PREFIX}:{task_id}"
        await self._redis.client.hset(key, mapping={
            "status": TaskStatus.COMPLETED,
            "updated_at": str(time.time()),
            "result_ref": result_ref,
        })
        logger.debug("Task %s completed", task_id)

    async def fail_task(self, task_id: str, error: str) -> None:
        key = f"{TASK_KEY_PREFIX}:{task_id}"
        await self._redis.client.hset(key, mapping={
            "status": TaskStatus.FAILED,
            "updated_at": str(time.time()),
            "error": error,
        })
        # Longer TTL for failed tasks to aid debugging (Cognee pattern: 7 days)
        await self._redis.client.expire(key, 86400)
        logger.debug("Task %s failed: %s", task_id, error)

    async def poison_task(self, task_id: str) -> None:
        key = f"{TASK_KEY_PREFIX}:{task_id}"
        await self._redis.client.hset(key, mapping={
            "status": TaskStatus.POISON,
            "updated_at": str(time.time()),
        })
        # Longer TTL for failed tasks to aid debugging (Cognee pattern: 7 days)
        await self._redis.client.expire(key, 86400)
        logger.warning("Task %s marked as poison", task_id)

    async def get_task(self, task_id: str) -> dict[str, str] | None:
        key = f"{TASK_KEY_PREFIX}:{task_id}"
        data = await self._redis.client.hgetall(key)
        return data if data else None

    async def increment_attempt(self, task_id: str) -> int:
        key = f"{TASK_KEY_PREFIX}:{task_id}"
        new_attempt = await self._redis.client.hincrby(key, "attempt", 1)
        return new_attempt
