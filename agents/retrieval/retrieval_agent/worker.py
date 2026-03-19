"""Retrieval storage worker — consumes from the embedded stream and upserts to Qdrant.

Extends :class:`BaseStreamWorker` which provides the common consumer loop
with deadline checks, retry/DLQ logic, pipeline depth enforcement, and
idempotency checks.  This worker only implements ``_process_message()``
and ``_publish_success()`` with Qdrant storage-specific domain logic.
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

from rag_common.redis.client import RedisClient
from rag_common.redis.streams import StreamMessage
from rag_common.redis.worker import BaseStreamWorker

from agents.retrieval.retrieval_agent.service import RetrievalService

logger = logging.getLogger(__name__)

STREAM_IN = "rag:stream:ingest:embedded"
STREAM_OUT = "rag:stream:ingest:stored"
DLQ_STREAM = "rag:stream:retrieval-storage:dlq"
CONSUMER_GROUP = "retrieval-storage-agents"


class RetrievalStorageWorker(BaseStreamWorker):
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
        cid = consumer_id or f"retrieval-{uuid4().hex[:8]}"
        super().__init__(
            redis_client,
            input_stream=STREAM_IN,
            output_stream=STREAM_OUT,
            dlq_stream=DLQ_STREAM,
            consumer_group=CONSUMER_GROUP,
            consumer_id=cid,
            output_stage="qdrant_stored",
        )
        self._service = service

    async def _process_message(self, msg: StreamMessage) -> Any:
        """Read embedding results from StateStore and upsert to Qdrant.

        Steps:
            1. Read embeddings from StateStore (``rag:results:{task_id}:embeddings``).
            2. Call ``RetrievalService.store_chunks()``.
            3. Return result dict for downstream publish.
        """
        task_id = msg.task_id
        logger.info("Processing task_id=%s (message_id=%s)", task_id, msg.message_id)

        # 1. Read embedding results from state store
        embeddings_data = await self._state.get_result(task_id, "embeddings")
        if embeddings_data is None:
            raise ValueError(f"No embeddings found in state store for task {task_id}")

        # The embeddings_data is expected to be a list of chunk dicts with vectors
        chunks_with_vectors: list[dict] = (
            embeddings_data
            if isinstance(embeddings_data, list)
            else embeddings_data.get("chunks", [])
        )

        # 2. Store in Qdrant
        result = await self._service.store_chunks(task_id, chunks_with_vectors)

        logger.info(
            "task_id=%s: stored %d chunks in Qdrant",
            task_id,
            result.get("stored_count", 0),
        )

        return result

    async def _publish_success(self, msg: StreamMessage, result: Any) -> None:
        """Publish to the stored stream with Qdrant-specific metadata."""
        await self._producer.publish(
            stream=STREAM_OUT,
            task_id=msg.task_id,
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
        logger.info(
            "task_id=%s: published to %s",
            msg.task_id,
            STREAM_OUT,
        )
