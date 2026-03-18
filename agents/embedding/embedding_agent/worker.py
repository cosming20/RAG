"""Stream consumer worker for the Embedding agent."""

from __future__ import annotations

import asyncio
import logging
from uuid import uuid4

from rag_common.llm.client import LLMClient
from rag_common.redis.client import RedisClient
from rag_common.redis.state import StateStore
from rag_common.redis.streams import StreamConsumer, StreamProducer

from agents.embedding.embedding_agent.config import EmbeddingConfig
from agents.embedding.embedding_agent.service import EmbeddingService

logger = logging.getLogger(__name__)

STREAM_IN = "rag:stream:ingest:chunked"
STREAM_OUT = "rag:stream:ingest:embedded"
CONSUMER_GROUP = "embedding-agents"


class EmbeddingWorker:
    """Consumes chunked messages, generates embeddings, publishes results.

    Joins the ``embedding-agents`` consumer group on the
    ``rag:stream:ingest:chunked`` stream.  For each incoming message it:

    1. Reads chunks from :class:`StateStore` at
       ``rag:results:{task_id}:chunks``.
    2. Calls :meth:`EmbeddingService.embed_chunks` to produce dense + sparse
       vectors.
    3. Publishes a downstream message on ``rag:stream:ingest:embedded``.
    4. Acknowledges the original message.
    """

    def __init__(
        self,
        redis_client: RedisClient,
        llm_client: LLMClient,
        config: EmbeddingConfig,
        *,
        consumer_id: str | None = None,
    ) -> None:
        self._redis = redis_client
        self._config = config
        self._consumer_id = consumer_id or f"emb-{uuid4().hex[:8]}"
        self._consumer = StreamConsumer(redis_client, STREAM_IN, CONSUMER_GROUP, self._consumer_id)
        self._producer = StreamProducer(redis_client)
        self._state = StateStore(redis_client)
        self._service = EmbeddingService(redis_client, llm_client, config)
        self._running = False

    async def start(self) -> None:
        """Initialize the consumer group and begin the processing loop."""
        await self._consumer.ensure_group()
        self._running = True
        logger.info(
            "EmbeddingWorker %s started on %s (group=%s)",
            self._consumer_id, STREAM_IN, CONSUMER_GROUP,
        )
        await self._run_loop()

    async def stop(self) -> None:
        """Signal the worker loop to stop."""
        self._running = False

    async def _run_loop(self) -> None:
        while self._running:
            messages = await self._consumer.read(count=1, block_ms=5000)

            for msg in messages:
                task_id = msg.task_id
                logger.info("Processing task %s (message=%s)", task_id, msg.message_id)
                try:
                    await self._process_message(task_id, msg.message_id, msg.trace_id)
                except Exception:
                    logger.exception("Failed to process task %s", task_id)
                    # Message is NOT acknowledged so it can be retried / claimed.

    async def _process_message(
        self,
        task_id: str,
        message_id: str,
        trace_id: str,
    ) -> None:
        # 1. Read chunks from state store
        chunks_data = await self._state.get_result(task_id, "chunks")
        if chunks_data is None:
            logger.error("No chunks found in state store for task %s", task_id)
            await self._consumer.ack(message_id)
            return

        chunks = chunks_data.get("chunks", []) if isinstance(chunks_data, dict) else chunks_data

        # 2. Generate embeddings
        result = await self._service.embed_chunks(task_id, chunks)
        logger.info(
            "Task %s: embedded %d chunks",
            task_id, result["embedding_count"],
        )

        # 3. Publish downstream
        await self._producer.publish(
            STREAM_OUT,
            task_id,
            trace_id=trace_id,
            payload_ref=f"rag:results:{task_id}:embeddings",
        )

        # 4. Acknowledge
        await self._consumer.ack(message_id)
        logger.info("Task %s acknowledged", task_id)
