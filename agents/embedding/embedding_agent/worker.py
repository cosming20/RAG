"""Stream consumer worker for the Embedding agent.

Extends :class:`BaseStreamWorker` which provides the common consumer loop
with deadline checks, retry/DLQ logic, pipeline depth enforcement, and
idempotency checks.  This worker only implements ``_process_message()``
with embedding-specific domain logic.
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

from rag_common.llm.client import LLMClient
from rag_common.redis.client import RedisClient
from rag_common.redis.streams import StreamMessage
from rag_common.redis.worker import BaseStreamWorker

from agents.embedding.embedding_agent.config import EmbeddingConfig
from agents.embedding.embedding_agent.service import EmbeddingService

logger = logging.getLogger(__name__)

STREAM_IN = "rag:stream:ingest:chunked"
STREAM_OUT = "rag:stream:ingest:embedded"
DLQ_STREAM = "rag:stream:embedding:dlq"
CONSUMER_GROUP = "embedding-agents"


class EmbeddingWorker(BaseStreamWorker):
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
        cid = consumer_id or f"emb-{uuid4().hex[:8]}"
        super().__init__(
            redis_client,
            input_stream=STREAM_IN,
            output_stream=STREAM_OUT,
            dlq_stream=DLQ_STREAM,
            consumer_group=CONSUMER_GROUP,
            consumer_id=cid,
            output_stage="embeddings",
        )
        self._embedding_config = config
        self._service = EmbeddingService(redis_client, llm_client, config)

    async def _process_message(self, msg: StreamMessage) -> Any:
        """Generate embeddings for chunked document data.

        Steps:
            1. Read chunks from StateStore (``rag:results:{task_id}:chunks``).
            2. Call :meth:`EmbeddingService.embed_chunks`.
            3. Return result dict with ``output_ref``.
        """
        task_id = msg.task_id
        logger.info("Processing task %s (message=%s)", task_id, msg.message_id)

        # 1. Read chunks from state store
        chunks_data = await self._state.get_result(task_id, "chunks")
        if chunks_data is None:
            raise ValueError(f"No chunks found in state store for task {task_id}")

        chunks = chunks_data.get("chunks", []) if isinstance(chunks_data, dict) else chunks_data

        # 2. Generate embeddings
        result = await self._service.embed_chunks(task_id, chunks)
        logger.info(
            "Task %s: embedded %d chunks",
            task_id, result["embedding_count"],
        )

        return {"output_ref": f"rag:results:{task_id}:embeddings"}
