"""Ingestion stream worker -- consumes from Redis Streams and processes documents."""

from __future__ import annotations

import base64
import logging
import time

from rag_common.redis.client import RedisClient
from rag_common.redis.state import StateStore
from rag_common.redis.streams import StreamConsumer, StreamMessage, StreamProducer
from rag_common.redis.task_queue import TaskQueue

from .config import IngestionConfig
from .service import IngestionService

logger = logging.getLogger(__name__)


class IngestionWorker:
    """Reads tasks from the ingestion pending stream and processes them.

    Consumer group: ``ingestion-agents``
    Input stream:   ``rag:stream:ingest:pending``
    Output stream:  ``rag:stream:ingest:parsed``
    """

    def __init__(
        self,
        redis_client: RedisClient,
        service: IngestionService,
        config: IngestionConfig,
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
            "IngestionWorker started: group=%s stream=%s",
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
                logger.error("IngestionWorker read loop error", exc_info=True)

    async def stop(self) -> None:
        """Signal the worker to stop after the current iteration."""
        self._running = False
        logger.info("IngestionWorker stop requested")

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
            task_id, f"ingestion-{self._consumer._consumer_id}"
        )
        if not claimed:
            logger.debug("Task %s already claimed, skipping", task_id)
            await self._consumer.ack(msg.message_id)
            return

        # -- Retrieve raw content from StateStore ----------------------------
        raw_content = await self._load_raw_content(msg)
        if raw_content is None:
            error_msg = "Raw content not found in StateStore"
            logger.error("Task %s: %s", task_id, error_msg)
            await self._task_queue.fail_task(task_id, error_msg)
            await self._consumer.ack(msg.message_id)
            return

        # -- Retrieve document metadata --------------------------------------
        meta = await self._state.get_result(task_id, "meta")
        document_id = msg.metadata.get("document_id", "")
        filename = msg.metadata.get("filename", "unknown")
        fmt = "txt"  # default
        if meta:
            document_id = document_id or meta.get("document_id", "")
            filename = meta.get("filename", filename)
            fmt = meta.get("format", fmt)

        # -- Process ----------------------------------------------------------
        try:
            result = await self._service.process_document(
                task_id=task_id,
                document_id=document_id,
                content=raw_content,
                filename=filename,
                fmt=fmt,
            )
        except Exception as exc:
            logger.error(
                "Task %s processing failed: %s", task_id, exc, exc_info=True
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

    async def _load_raw_content(self, msg: StreamMessage) -> bytes | None:
        """Load raw document bytes from the StateStore."""
        raw_b64 = await self._state.get_result(msg.task_id, "raw")
        if raw_b64 is None:
            # Try the payload_ref key directly
            if msg.payload_ref:
                data = await self._redis.client.get(msg.payload_ref)
                if data is not None:
                    try:
                        return base64.b64decode(data)
                    except Exception:
                        if isinstance(data, bytes):
                            return data
                        return data.encode("utf-8") if isinstance(data, str) else None
            return None
        # StateStore.get_raw stores base64-encoded content
        if isinstance(raw_b64, str):
            try:
                return base64.b64decode(raw_b64)
            except Exception:
                return raw_b64.encode("utf-8")
        return raw_b64  # type: ignore[return-value]

    async def _on_success(
        self, msg: StreamMessage, result: dict
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
            payload_ref=result.get("parsed_key", ""),
            metadata={
                "document_id": result.get("document_id", ""),
                "filename": msg.metadata.get("filename", ""),
            },
        )
        await self._task_queue.complete_task(
            msg.task_id, result_ref=result.get("parsed_key", "")
        )
        await self._consumer.ack(msg.message_id)
        logger.info("Task %s completed and forwarded to parsed stream", msg.task_id)

    async def _retry_or_dlq(self, msg: StreamMessage, error: str) -> None:
        """Re-publish with incremented attempt or move to DLQ."""
        next_attempt = msg.attempt + 1
        if next_attempt >= msg.max_attempts:
            logger.warning("Task %s moving to DLQ after %d attempts", msg.task_id, next_attempt)
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
