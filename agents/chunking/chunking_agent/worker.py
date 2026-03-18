"""Chunking stream worker -- consumes parsed documents and produces chunks."""

from __future__ import annotations

import logging
import time
from typing import Any

from rag_common.redis.client import RedisClient
from rag_common.redis.state import StateStore
from rag_common.redis.streams import StreamConsumer, StreamMessage, StreamProducer
from rag_common.redis.task_queue import TaskQueue

from .config import ChunkingConfig
from .service import ChunkingService

logger = logging.getLogger(__name__)


class ChunkingWorker:
    """Reads tasks from the parsed stream and chunks them.

    Consumer group: ``chunking-agents``
    Input stream:   ``rag:stream:ingest:parsed``
    Output stream:  ``rag:stream:ingest:chunked``
    """

    def __init__(
        self,
        redis_client: RedisClient,
        service: ChunkingService,
        config: ChunkingConfig,
        consumer_id: str,
    ) -> None:
        self._redis = redis_client
        self._service = service
        self._config = config
        self._consumer = StreamConsumer(
            redis_client=redis_client,
            stream=config.input_stream,
            group=config.consumer_group,
            consumer_id=consumer_id,
        )
        self._producer = StreamProducer(redis_client)
        self._state = StateStore(redis_client)
        self._task_queue = TaskQueue(redis_client)
        self._running = False

    async def run(self) -> None:
        """Main consumer loop. Blocks until stopped."""
        await self._consumer.ensure_group()
        self._running = True
        logger.info(
            "ChunkingWorker started: group=%s stream=%s",
            self._config.consumer_group,
            self._config.input_stream,
        )

        while self._running:
            try:
                messages = await self._consumer.read(
                    count=self._config.batch_size,
                    block_ms=self._config.block_ms,
                )
                for msg in messages:
                    await self._handle_message(msg)
            except Exception:
                logger.error("ChunkingWorker read loop error", exc_info=True)

    async def stop(self) -> None:
        """Signal the worker to stop after the current iteration."""
        self._running = False
        logger.info("ChunkingWorker stop requested")

    async def _handle_message(self, msg: StreamMessage) -> None:
        """Process a single stream message with deadline and retry logic."""
        task_id = msg.task_id
        logger.info("Processing task_id=%s attempt=%d", task_id, msg.attempt)

        # -- Deadline check --------------------------------------------------
        if self._is_expired(msg):
            logger.warning("Task %s expired (deadline passed), skipping", task_id)
            await self._consumer.ack(msg.message_id)
            await self._task_queue.fail_task(task_id, "Deadline exceeded")
            return

        # -- Max-attempts check (DLQ) ----------------------------------------
        if msg.attempt >= msg.max_attempts:
            logger.warning(
                "Task %s exceeded max attempts (%d), moving to DLQ",
                task_id,
                msg.max_attempts,
            )
            await self._move_to_dlq(msg)
            await self._consumer.ack(msg.message_id)
            return

        # -- Claim task -------------------------------------------------------
        claimed = await self._task_queue.claim_task(
            task_id, f"chunking-{self._consumer._consumer_id}"
        )
        if not claimed:
            logger.debug("Task %s already claimed, skipping", task_id)
            await self._consumer.ack(msg.message_id)
            return

        # -- Load parsed content from StateStore -----------------------------
        parsed = await self._load_parsed_content(msg)
        if parsed is None:
            error_msg = "Parsed content not found in StateStore"
            logger.error("Task %s: %s", task_id, error_msg)
            await self._task_queue.fail_task(task_id, error_msg)
            await self._consumer.ack(msg.message_id)
            return

        # -- Extract fields ---------------------------------------------------
        parsed_text = parsed.get("text", "")
        document_id = msg.metadata.get("document_id", parsed.get("document_id", ""))
        fmt = parsed.get("format", "txt")
        doc_metadata = parsed.get("parser_metadata", {})

        # -- Process ----------------------------------------------------------
        try:
            result = await self._service.chunk_document(
                task_id=task_id,
                document_id=document_id,
                parsed_content=parsed_text,
                fmt=fmt,
                metadata=doc_metadata,
            )
        except Exception as exc:
            logger.error(
                "Task %s chunking failed: %s", task_id, exc, exc_info=True
            )
            await self._retry_or_dlq(msg, str(exc))
            return

        # -- Outcome ----------------------------------------------------------
        if result.get("success"):
            await self._on_success(msg, result)
        else:
            errors = "; ".join(result.get("errors", ["Unknown error"]))
            await self._retry_or_dlq(msg, errors)

    # ------------------------------------------------------------------ #
    #  Helpers                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _is_expired(msg: StreamMessage) -> bool:
        if msg.deadline_unix_ms <= 0:
            return False
        now_ms = int(time.time() * 1000)
        return now_ms > msg.deadline_unix_ms

    async def _load_parsed_content(
        self, msg: StreamMessage
    ) -> dict[str, Any] | None:
        """Load parsed document data from StateStore."""
        parsed = await self._state.get_result(msg.task_id, "parsed")
        if parsed is not None:
            return parsed

        # Try the payload_ref key directly (it may point to the parsed stage)
        if msg.payload_ref:
            data = await self._redis.client.get(msg.payload_ref)
            if data is not None:
                import json

                try:
                    return json.loads(data)
                except (json.JSONDecodeError, TypeError):
                    logger.warning(
                        "Could not parse payload_ref data for task %s",
                        msg.task_id,
                    )
        return None

    async def _on_success(
        self, msg: StreamMessage, result: dict[str, Any]
    ) -> None:
        """Publish downstream and acknowledge."""
        await self._producer.publish(
            stream=self._config.output_stream,
            task_id=msg.task_id,
            trace_id=msg.trace_id,
            span_id=msg.span_id,
            attempt=0,
            max_attempts=msg.max_attempts,
            deadline_unix_ms=msg.deadline_unix_ms,
            pipeline_depth=msg.pipeline_depth + 1,
            payload_ref=result.get("chunks_key", ""),
            metadata={
                "document_id": result.get("document_id", ""),
                "chunk_count": str(result.get("chunk_count", 0)),
            },
        )
        await self._task_queue.complete_task(
            msg.task_id, result_ref=result.get("chunks_key", "")
        )
        await self._consumer.ack(msg.message_id)
        logger.info(
            "Task %s chunked (%d chunks) and forwarded to chunked stream",
            msg.task_id,
            result.get("chunk_count", 0),
        )

    async def _retry_or_dlq(self, msg: StreamMessage, error: str) -> None:
        """Re-publish with incremented attempt or move to DLQ."""
        next_attempt = msg.attempt + 1
        if next_attempt >= msg.max_attempts:
            logger.warning(
                "Task %s moving to DLQ after %d attempts",
                msg.task_id,
                next_attempt,
            )
            await self._move_to_dlq(msg)
        else:
            logger.info(
                "Task %s retrying (attempt %d/%d): %s",
                msg.task_id,
                next_attempt,
                msg.max_attempts,
                error,
            )
            await self._producer.publish(
                stream=self._config.input_stream,
                task_id=msg.task_id,
                trace_id=msg.trace_id,
                span_id=msg.span_id,
                attempt=next_attempt,
                max_attempts=msg.max_attempts,
                deadline_unix_ms=msg.deadline_unix_ms,
                pipeline_depth=msg.pipeline_depth,
                payload_ref=msg.payload_ref,
                metadata=msg.metadata,
            )
        await self._consumer.ack(msg.message_id)

    async def _move_to_dlq(self, msg: StreamMessage) -> None:
        """Move a message to the dead-letter queue stream."""
        await self._producer.publish(
            stream=self._config.dlq_stream,
            task_id=msg.task_id,
            trace_id=msg.trace_id,
            span_id=msg.span_id,
            attempt=msg.attempt,
            max_attempts=msg.max_attempts,
            deadline_unix_ms=msg.deadline_unix_ms,
            pipeline_depth=msg.pipeline_depth,
            payload_ref=msg.payload_ref,
            metadata={**msg.metadata, "dlq_reason": "max_attempts_exceeded"},
        )
        await self._task_queue.fail_task(
            msg.task_id, "Moved to DLQ after max attempts"
        )
