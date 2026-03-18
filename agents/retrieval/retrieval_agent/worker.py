"""Retrieval storage worker — consumes from the embedded stream and upserts to Qdrant."""

from __future__ import annotations

import asyncio
import json
import logging
from uuid import uuid4

from rag_common.redis.client import RedisClient
from rag_common.redis.state import StateStore
from rag_common.redis.streams import StreamConsumer, StreamMessage, StreamProducer

from agents.retrieval.retrieval_agent.service import RetrievalService

logger = logging.getLogger(__name__)

STREAM_IN = "rag:stream:ingest:embedded"
STREAM_OUT = "rag:stream:ingest:stored"
CONSUMER_GROUP = "retrieval-storage-agents"


class RetrievalStorageWorker:
    """Stream consumer that reads embedded chunks and persists them in Qdrant.

    Joins consumer group ``retrieval-storage-agents`` on
    ``rag:stream:ingest:embedded``.  After successful storage, publishes a
    completion event to ``rag:stream:ingest:stored`` with metadata indicating
    ``qdrant_stored``.
    """

    def __init__(
        self,
        redis_client: RedisClient,
        service: RetrievalService,
        *,
        consumer_id: str | None = None,
    ) -> None:
        self._redis = redis_client
        self._service = service
        self._state = StateStore(redis_client)
        self._consumer_id = consumer_id or f"retrieval-{uuid4().hex[:8]}"
        self._consumer = StreamConsumer(
            redis_client,
            stream=STREAM_IN,
            group=CONSUMER_GROUP,
            consumer_id=self._consumer_id,
        )
        self._producer = StreamProducer(redis_client)
        self._running = False

    async def start(self) -> None:
        """Create the consumer group (idempotent) and begin the read loop."""
        await self._consumer.ensure_group()
        self._running = True
        logger.info(
            "RetrievalStorageWorker started (consumer=%s, group=%s, stream=%s)",
            self._consumer_id,
            CONSUMER_GROUP,
            STREAM_IN,
        )
        await self._run_loop()

    async def stop(self) -> None:
        """Signal the worker to stop after the current iteration."""
        self._running = False
        logger.info("RetrievalStorageWorker stopping (consumer=%s)", self._consumer_id)

    async def _run_loop(self) -> None:
        """Main consumer loop: read, process, ack."""
        while self._running:
            try:
                messages = await self._consumer.read(count=5, block_ms=5000)
                for msg in messages:
                    await self._process_message(msg)
            except asyncio.CancelledError:
                logger.info("Worker loop cancelled")
                break
            except Exception:
                logger.exception("Error in retrieval storage worker loop")
                await asyncio.sleep(1)

    async def _process_message(self, msg: StreamMessage) -> None:
        """Handle a single stream message.

        Steps:
            1. Read embeddings from StateStore (``rag:results:{task_id}:embeddings``).
            2. Call ``RetrievalService.store_chunks()``.
            3. Publish to ``rag:stream:ingest:stored`` with ``qdrant_stored`` metadata.
            4. XACK the message.
        """
        task_id = msg.task_id
        logger.info("Processing task_id=%s (message_id=%s)", task_id, msg.message_id)

        try:
            # 1. Read embedding results from state store
            embeddings_data = await self._state.get_result(task_id, "embeddings")
            if embeddings_data is None:
                logger.error(
                    "task_id=%s: no embeddings found in state store, skipping",
                    task_id,
                )
                await self._consumer.ack(msg.message_id)
                return

            # The embeddings_data is expected to be a list of chunk dicts with vectors
            chunks_with_vectors: list[dict] = (
                embeddings_data
                if isinstance(embeddings_data, list)
                else embeddings_data.get("chunks", [])
            )

            # 2. Store in Qdrant
            result = await self._service.store_chunks(task_id, chunks_with_vectors)

            # 3. Publish downstream
            await self._producer.publish(
                stream=STREAM_OUT,
                task_id=task_id,
                trace_id=msg.trace_id,
                span_id=msg.span_id,
                attempt=msg.attempt,
                max_attempts=msg.max_attempts,
                deadline_unix_ms=msg.deadline_unix_ms,
                pipeline_depth=msg.pipeline_depth + 1,
                payload_ref=msg.payload_ref,
                metadata={
                    **msg.metadata,
                    "qdrant_stored": "true",
                    "stored_count": str(result.get("stored_count", 0)),
                },
            )

            # 4. ACK
            await self._consumer.ack(msg.message_id)
            logger.info(
                "task_id=%s: stored %d chunks in Qdrant, published to %s",
                task_id,
                result.get("stored_count", 0),
                STREAM_OUT,
            )

        except Exception:
            logger.exception("task_id=%s: failed to process message", task_id)
            # Message will be re-delivered after idle timeout via claim_pending
