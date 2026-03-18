"""Redis Streams producer and consumer for inter-agent communication."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import redis.exceptions

from rag_common.redis.client import RedisClient

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StreamMessage:
    """A message from a Redis Stream."""

    message_id: str
    stream: str
    task_id: str
    trace_id: str
    span_id: str
    attempt: int
    max_attempts: int
    deadline_unix_ms: int
    pipeline_depth: int
    payload_ref: str
    metadata: dict[str, str]
    version: int = 1


class StreamProducer:
    """Publishes task envelopes to Redis Streams."""

    def __init__(self, redis_client: RedisClient) -> None:
        self._redis = redis_client

    async def publish(
        self,
        stream: str,
        task_id: str,
        *,
        trace_id: str = "",
        span_id: str = "",
        attempt: int = 0,
        max_attempts: int = 3,
        deadline_unix_ms: int = 0,
        pipeline_depth: int = 0,
        payload_ref: str = "",
        metadata: dict[str, str] | None = None,
        version: int = 1,
    ) -> str:
        envelope = {
            "task_id": task_id,
            "trace_id": trace_id,
            "span_id": span_id,
            "attempt": str(attempt),
            "max_attempts": str(max_attempts),
            "deadline_unix_ms": str(deadline_unix_ms),
            "pipeline_depth": str(pipeline_depth),
            "payload_ref": payload_ref,
            "metadata": json.dumps(metadata or {}),
            "published_at": str(int(time.time() * 1000)),
            "version": str(version),
        }
        message_id = await self._redis.client.xadd(stream, envelope)
        logger.debug("Published to %s: task_id=%s", stream, task_id)
        return message_id


class StreamConsumer:
    """Consumes messages from Redis Streams using consumer groups."""

    def __init__(
        self,
        redis_client: RedisClient,
        stream: str,
        group: str,
        consumer_id: str,
    ) -> None:
        self._redis = redis_client
        self._stream = stream
        self._group = group
        self._consumer_id = consumer_id

    async def ensure_group(self) -> None:
        try:
            await self._redis.client.xgroup_create(
                self._stream, self._group, id="0", mkstream=True,
            )
            logger.info("Created consumer group %s on %s", self._group, self._stream)
        except redis.exceptions.ResponseError as e:
            if "BUSYGROUP" in str(e):
                logger.debug("Consumer group %s already exists on %s", self._group, self._stream)
            else:
                raise

    async def read(
        self,
        count: int = 1,
        block_ms: int = 5000,
    ) -> list[StreamMessage]:
        results = await self._redis.client.xreadgroup(
            groupname=self._group,
            consumername=self._consumer_id,
            streams={self._stream: ">"},
            count=count,
            block=block_ms,
        )
        messages = []
        if results:
            for stream_name, stream_messages in results:
                for message_id, data in stream_messages:
                    msg = StreamMessage(
                        message_id=message_id,
                        stream=stream_name if isinstance(stream_name, str) else stream_name.decode(),
                        task_id=data.get("task_id", ""),
                        trace_id=data.get("trace_id", ""),
                        span_id=data.get("span_id", ""),
                        attempt=int(data.get("attempt", "0")),
                        max_attempts=int(data.get("max_attempts", "3")),
                        deadline_unix_ms=int(data.get("deadline_unix_ms", "0")),
                        pipeline_depth=int(data.get("pipeline_depth", "0")),
                        payload_ref=data.get("payload_ref", ""),
                        metadata=json.loads(data.get("metadata", "{}")),
                        version=int(data.get("version", "1")),
                    )
                    messages.append(msg)
        return messages

    async def ack(self, message_id: str) -> None:
        await self._redis.client.xack(self._stream, self._group, message_id)

    async def claim_pending(
        self,
        min_idle_ms: int = 60000,
        count: int = 10,
    ) -> list[StreamMessage]:
        """Claim messages that have been pending longer than min_idle_ms (orphan recovery)."""
        pending = await self._redis.client.xpending_range(
            self._stream, self._group, min="-", max="+", count=count,
        )
        messages = []
        for entry in pending:
            msg_id = entry["message_id"]
            idle = entry.get("time_since_delivered", 0)
            if idle >= min_idle_ms:
                claimed = await self._redis.client.xclaim(
                    self._stream,
                    self._group,
                    self._consumer_id,
                    min_idle_time=min_idle_ms,
                    message_ids=[msg_id],
                )
                for message_id, data in claimed:
                    if data:
                        msg = StreamMessage(
                            message_id=message_id,
                            stream=self._stream,
                            task_id=data.get("task_id", ""),
                            trace_id=data.get("trace_id", ""),
                            span_id=data.get("span_id", ""),
                            attempt=int(data.get("attempt", "0")),
                            max_attempts=int(data.get("max_attempts", "3")),
                            deadline_unix_ms=int(data.get("deadline_unix_ms", "0")),
                            pipeline_depth=int(data.get("pipeline_depth", "0")),
                            payload_ref=data.get("payload_ref", ""),
                            metadata=json.loads(data.get("metadata", "{}")),
                            version=int(data.get("version", "1")),
                        )
                        messages.append(msg)
        return messages
