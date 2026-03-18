"""Agent registry — register, heartbeat, discover agents via Redis."""

from __future__ import annotations

import asyncio
import logging
import time

from rag_common.models.agent import AgentInfo, AgentRole, AgentStatus
from rag_common.redis.client import RedisClient

logger = logging.getLogger(__name__)

AGENT_KEY_PREFIX = "rag:agents"
ROLE_SET_PREFIX = "rag:agents:by_role"


class AgentRegistry:
    """Manages agent registration, heartbeat, and discovery in Redis."""

    def __init__(self, redis_client: RedisClient) -> None:
        self._redis = redis_client
        self._heartbeat_task: asyncio.Task[None] | None = None

    async def register(self, agent: AgentInfo) -> None:
        key = f"{AGENT_KEY_PREFIX}:{agent.agent_id}"
        role_key = f"{ROLE_SET_PREFIX}:{agent.role}"
        pipe = self._redis.client.pipeline()
        pipe.hset(key, mapping={
            "role": agent.role,
            "grpc_address": agent.grpc_address,
            "status": agent.status,
            "registered_at": agent.registered_at.isoformat(),
            "last_heartbeat": str(int(time.time() * 1000)),
            "capabilities": str(agent.capabilities),
        })
        pipe.expire(key, 30)
        pipe.sadd(role_key, agent.agent_id)
        await pipe.execute()
        logger.info("Registered agent %s (role=%s) at %s", agent.agent_id, agent.role, agent.grpc_address)

    async def deregister(self, agent_id: str, role: AgentRole) -> None:
        key = f"{AGENT_KEY_PREFIX}:{agent_id}"
        role_key = f"{ROLE_SET_PREFIX}:{role}"
        pipe = self._redis.client.pipeline()
        pipe.delete(key)
        pipe.srem(role_key, agent_id)
        await pipe.execute()
        logger.info("Deregistered agent %s", agent_id)

    async def heartbeat(self, agent_id: str, ttl_seconds: int = 30) -> None:
        key = f"{AGENT_KEY_PREFIX}:{agent_id}"
        pipe = self._redis.client.pipeline()
        pipe.hset(key, "last_heartbeat", str(int(time.time() * 1000)))
        pipe.hset(key, "status", AgentStatus.SERVING)
        pipe.expire(key, ttl_seconds)
        await pipe.execute()

    async def start_heartbeat_loop(
        self,
        agent_id: str,
        interval_seconds: float = 5.0,
        ttl_seconds: int = 30,
    ) -> None:
        async def _loop() -> None:
            while True:
                try:
                    await self.heartbeat(agent_id, ttl_seconds)
                except Exception:
                    logger.exception("Heartbeat failed for %s", agent_id)
                await asyncio.sleep(interval_seconds)

        self._heartbeat_task = asyncio.create_task(_loop())
        logger.info("Heartbeat loop started for %s (every %.1fs)", agent_id, interval_seconds)

    async def stop_heartbeat_loop(self) -> None:
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None

    async def discover(self, role: AgentRole) -> list[str]:
        role_key = f"{ROLE_SET_PREFIX}:{role}"
        members = await self._redis.client.smembers(role_key)
        return list(members)

    async def get_agent_info(self, agent_id: str) -> dict[str, str] | None:
        key = f"{AGENT_KEY_PREFIX}:{agent_id}"
        data = await self._redis.client.hgetall(key)
        return data if data else None
