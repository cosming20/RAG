"""Graph storage worker — consumes from the embedded stream and stores entities in Neo4j."""

from __future__ import annotations

import asyncio
import logging
from uuid import uuid4

from rag_common.redis.client import RedisClient
from rag_common.redis.state import StateStore
from rag_common.redis.streams import StreamConsumer, StreamMessage, StreamProducer

from agents.graph.graph_agent.service import GraphService

logger = logging.getLogger(__name__)

STREAM_IN = "rag:stream:ingest:embedded"
STREAM_OUT = "rag:stream:ingest:stored"
CONSUMER_GROUP = "graph-storage-agents"


class GraphStorageWorker:
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
        self._redis = redis_client
        self._service = service
        self._state = StateStore(redis_client)
        self._consumer_id = consumer_id or f"graph-{uuid4().hex[:8]}"
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
            "GraphStorageWorker started (consumer=%s, group=%s, stream=%s)",
            self._consumer_id,
            CONSUMER_GROUP,
            STREAM_IN,
        )
        await self._run_loop()

    async def stop(self) -> None:
        """Signal the worker to stop after the current iteration."""
        self._running = False
        logger.info("GraphStorageWorker stopping (consumer=%s)", self._consumer_id)

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
                logger.exception("Error in graph storage worker loop")
                await asyncio.sleep(1)

    async def _process_message(self, msg: StreamMessage) -> None:
        """Handle a single stream message.

        Steps:
            1. Read chunks from StateStore (``rag:results:{task_id}:chunks``).
            2. Read document metadata from StateStore.
            3. Call ``GraphService.store_entities()``.
            4. Publish to ``rag:stream:ingest:stored`` with ``neo4j_stored`` metadata.
            5. XACK the message.
        """
        task_id = msg.task_id
        logger.info("Processing task_id=%s (message_id=%s)", task_id, msg.message_id)

        try:
            # 1. Read original chunks from state store
            chunks_data = await self._state.get_result(task_id, "chunks")
            if chunks_data is None:
                logger.error(
                    "task_id=%s: no chunks found in state store, skipping",
                    task_id,
                )
                await self._consumer.ack(msg.message_id)
                return

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

            # 4. Publish downstream
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
                    "neo4j_stored": "true",
                    "nodes_created": str(result.get("nodes_created", 0)),
                    "relationships_created": str(result.get("relationships_created", 0)),
                },
            )

            # 5. ACK
            await self._consumer.ack(msg.message_id)
            logger.info(
                "task_id=%s: created %d nodes, %d rels in Neo4j, published to %s",
                task_id,
                result.get("nodes_created", 0),
                result.get("relationships_created", 0),
                STREAM_OUT,
            )

        except Exception:
            logger.exception("task_id=%s: failed to process message", task_id)
