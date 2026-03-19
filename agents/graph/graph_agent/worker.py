"""Graph storage worker — consumes from the embedded stream and stores entities in Neo4j.

Extends :class:`BaseStreamWorker` which provides the common consumer loop
with deadline checks, retry/DLQ logic, pipeline depth enforcement, and
idempotency checks.  This worker only implements ``_process_message()``
and ``_publish_success()`` with Neo4j storage-specific domain logic.
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

from rag_common.redis.client import RedisClient
from rag_common.redis.streams import StreamMessage
from rag_common.redis.worker import BaseStreamWorker

from agents.graph.graph_agent.service import GraphService

logger = logging.getLogger(__name__)

STREAM_IN = "rag:stream:ingest:embedded"
STREAM_OUT = "rag:stream:ingest:stored"
DLQ_STREAM = "rag:stream:graph-storage:dlq"
CONSUMER_GROUP = "graph-storage-agents"


class GraphStorageWorker(BaseStreamWorker):
    """Stream consumer that reads embedded chunks and stores entities in Neo4j.

    Joins consumer group ``graph-storage-agents`` on
    ``rag:stream:ingest:embedded``.  This is the same stream consumed by the
    Retrieval agent -- both agents receive every message via different consumer
    groups (fan-out pattern).

    After successful entity extraction and storage, publishes a completion
    event to ``rag:stream:ingest:stored`` with metadata indicating
    ``neo4j_stored``.
    """

    def __init__(
        self,
        redis_client: RedisClient,
        service: GraphService,
        *,
        consumer_id: str | None = None,
    ) -> None:
        cid = consumer_id or f"graph-{uuid4().hex[:8]}"
        super().__init__(
            redis_client,
            input_stream=STREAM_IN,
            output_stream=STREAM_OUT,
            dlq_stream=DLQ_STREAM,
            consumer_group=CONSUMER_GROUP,
            consumer_id=cid,
            output_stage="neo4j_stored",
        )
        self._service = service

    async def _process_message(self, msg: StreamMessage) -> Any:
        """Extract entities from chunks and store in Neo4j.

        Steps:
            1. Read chunks from StateStore (``rag:results:{task_id}:chunks``).
            2. Read document metadata from StateStore.
            3. Call ``GraphService.store_entities()``.
            4. Return result dict for downstream publish.
        """
        task_id = msg.task_id
        logger.info("Processing task_id=%s (message_id=%s)", task_id, msg.message_id)

        # 1. Read original chunks from state store
        chunks_data = await self._state.get_result(task_id, "chunks")
        if chunks_data is None:
            raise ValueError(f"No chunks found in state store for task {task_id}")

        chunks: list[dict] = (
            chunks_data
            if isinstance(chunks_data, list)
            else chunks_data.get("chunks", [])
        )

        # 2. Read document metadata
        doc_meta = await self._state.get_result(task_id, "document_meta")
        document_id = ""
        filename = ""
        doc_format = ""
        if doc_meta and isinstance(doc_meta, dict):
            document_id = str(doc_meta.get("document_id", ""))
            filename = doc_meta.get("filename", "")
            doc_format = doc_meta.get("format", "")

        # Fallback: extract document_id from message metadata or chunks
        if not document_id and chunks:
            document_id = str(chunks[0].get("document_id", ""))

        # 3. Store in Neo4j
        result = await self._service.store_entities(
            task_id=task_id,
            document_id=document_id,
            chunks=chunks,
            filename=filename,
            doc_format=doc_format,
        )

        logger.info(
            "task_id=%s: created %d nodes, %d rels in Neo4j",
            task_id,
            result.get("nodes_created", 0),
            result.get("relationships_created", 0),
        )

        return result

    async def _publish_success(self, msg: StreamMessage, result: Any) -> None:
        """Publish to the stored stream with Neo4j-specific metadata."""
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
                "neo4j_stored": "true",
                "nodes_created": str(result.get("nodes_created", 0)),
                "relationships_created": str(result.get("relationships_created", 0)),
            },
        )
        logger.info(
            "task_id=%s: published to %s",
            msg.task_id,
            STREAM_OUT,
        )
