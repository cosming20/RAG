"""Base stream worker — shared consumer loop with deadline, retry, and DLQ logic.

Eliminates ~200 lines of duplicated boilerplate across all 5+ agent workers.
Subclasses only need to implement `_process_message()`.
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Any

from rag_common.config import AgentConfig
from rag_common.redis.client import RedisClient
from rag_common.redis.state import StateStore
from rag_common.redis.streams import StreamConsumer, StreamMessage, StreamProducer
from rag_common.redis.task_queue import TaskQueue

logger = logging.getLogger(__name__)


class BaseStreamWorker(ABC):
    """Base class for all agent stream workers.

    Provides the common consumer loop with:
    - Deadline checking (skip expired tasks)
    - Max attempts / DLQ routing (poison message protection)
    - Pipeline depth enforcement (loop prevention)
    - Retry logic (increment attempt, re-publish on transient failure)
    - Graceful shutdown

    Subclasses implement `_process_message()` with their domain logic
    and `_publish_success()` to route output to the next stream.
    """

    def __init__(
        self,
        redis_client: RedisClient,
        *,
        input_stream: str,
        output_stream: str,
        dlq_stream: str,
        consumer_group: str,
        consumer_id: str,
        agent_config: AgentConfig | None = None,
        output_stage: str = "",
    ) -> None:
        self._redis = redis_client
        self._state = StateStore(redis_client)
        self._task_queue = TaskQueue(redis_client)
        self._producer = StreamProducer(redis_client)
        self._consumer = StreamConsumer(
            redis_client, input_stream, consumer_group, consumer_id,
        )
        self._input_stream = input_stream
        self._output_stream = output_stream
        self._dlq_stream = dlq_stream
        self._config = agent_config or AgentConfig()
        self._output_stage = output_stage
        self._running = False

    async def start(self) -> None:
        """Start the consumer loop."""
        await self._consumer.ensure_group()
        self._running = True
        logger.info(
            "Worker started: group=%s stream=%s",
            self._consumer._group,
            self._input_stream,
        )
        while self._running:
            try:
                messages = await self._consumer.read(count=1, block_ms=5000)
                for msg in messages:
                    await self._handle_message(msg)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in worker loop")
                await asyncio.sleep(1)

    def stop(self) -> None:
        """Signal the worker to stop."""
        self._running = False

    async def _handle_message(self, msg: StreamMessage) -> None:
        """Full message handling: deadline, depth, retry, DLQ, process, ack."""
        task_id = msg.task_id

        # 1. Deadline check
        if msg.deadline_unix_ms > 0:
            now_ms = int(time.time() * 1000)
            if now_ms > msg.deadline_unix_ms:
                logger.warning("Task %s expired (deadline passed), skipping", task_id)
                await self._task_queue.fail_task(task_id, "deadline_exceeded")
                await self._consumer.ack(msg.message_id)
                return

        # 2. Pipeline depth check
        max_depth = self._config.max_pipeline_depth
        if msg.pipeline_depth > max_depth:
            logger.error("Task %s exceeded max pipeline depth %d", task_id, max_depth)
            await self._move_to_dlq(msg, reason="pipeline_depth_exceeded")
            return

        # 3. Max attempts check
        if msg.attempt >= msg.max_attempts:
            logger.warning("Task %s exceeded max attempts (%d), moving to DLQ", task_id, msg.max_attempts)
            await self._task_queue.poison_task(task_id)
            await self._move_to_dlq(msg, reason="max_attempts_exceeded")
            return

        # 4. Idempotency check — skip re-processing if output already exists
        if self._output_stage:
            result_key = f"rag:results:{task_id}:{self._output_stage}"
            already_done = await self._redis.client.exists(result_key)
            if already_done:
                logger.info("Task %s already processed for stage %s, skipping", task_id, self._output_stage)
                await self._publish_success(msg, {"output_ref": f"{result_key}"})
                await self._consumer.ack(msg.message_id)
                return

        # 5. Claim + process
        claimed = await self._task_queue.claim_task(task_id, self._consumer._consumer_id)
        if not claimed:
            logger.debug("Task %s already claimed, skipping", task_id)
            await self._consumer.ack(msg.message_id)
            return

        try:
            result = await self._process_message(msg)
            await self._publish_success(msg, result)
            await self._task_queue.complete_task(task_id)
            await self._consumer.ack(msg.message_id)
        except Exception as e:
            logger.exception("Failed to process task %s", task_id)
            new_attempt = await self._task_queue.increment_attempt(task_id)
            if new_attempt >= msg.max_attempts:
                await self._task_queue.poison_task(task_id)
                await self._move_to_dlq(msg, reason=str(e))
            else:
                await self._task_queue.fail_task(task_id, str(e))
                # Leave unacked for re-delivery via XCLAIM
            await self._consumer.ack(msg.message_id)

    @abstractmethod
    async def _process_message(self, msg: StreamMessage) -> Any:
        """Process a single message. Return result data for downstream publish."""

    async def _publish_success(self, msg: StreamMessage, result: Any) -> None:
        """Publish to output stream, forwarding all envelope fields."""
        output_ref = result.get("output_ref", "") if isinstance(result, dict) else ""
        await self._producer.publish(
            self._output_stream,
            msg.task_id,
            trace_id=msg.trace_id,
            span_id=msg.span_id,
            attempt=msg.attempt,
            max_attempts=msg.max_attempts,
            deadline_unix_ms=msg.deadline_unix_ms,
            pipeline_depth=msg.pipeline_depth + 1,
            payload_ref=output_ref,
            metadata=msg.metadata,
        )

    async def _move_to_dlq(self, msg: StreamMessage, *, reason: str = "") -> None:
        """Move a message to the dead letter queue."""
        await self._producer.publish(
            self._dlq_stream,
            msg.task_id,
            trace_id=msg.trace_id,
            span_id=msg.span_id,
            attempt=msg.attempt,
            max_attempts=msg.max_attempts,
            pipeline_depth=msg.pipeline_depth,
            payload_ref=msg.payload_ref,
            metadata={**msg.metadata, "dlq_reason": reason},
        )
        await self._consumer.ack(msg.message_id)
