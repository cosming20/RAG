"""Reranking worker -- extends BaseStreamWorker for the reranking pipeline stage.

Consumes from ``rag:stream:query:retrieved``, waits briefly for graph context,
then runs the full reranking pipeline and publishes to ``rag:stream:query:reranked``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from rag_common.redis.client import RedisClient
from rag_common.redis.streams import StreamMessage
from rag_common.redis.worker import BaseStreamWorker

from agents.reranking.reranking_agent.config import RerankingConfig
from agents.reranking.reranking_agent.service import RerankingService

logger = logging.getLogger(__name__)

STREAM_IN = "rag:stream:query:retrieved"
STREAM_OUT = "rag:stream:query:reranked"
DLQ_STREAM = "rag:stream:query:reranked:dlq"
CONSUMER_GROUP = "reranking-agents"


class RerankingWorker(BaseStreamWorker):
    """Stream worker that merges retrieval + graph results and applies reranking.

    Waits for both retrieval results (required) and graph context (optional,
    with a brief timeout) before running the Jina reranker, LLM trimmer, and
    reference resolver.
    """

    def __init__(
        self,
        redis_client: RedisClient,
        service: RerankingService,
        config: RerankingConfig,
        *,
        consumer_id: str,
    ) -> None:
        super().__init__(
            redis_client,
            input_stream=STREAM_IN,
            output_stream=STREAM_OUT,
            dlq_stream=DLQ_STREAM,
            consumer_group=CONSUMER_GROUP,
            consumer_id=consumer_id,
        )
        self._service = service
        self._reranking_config = config

    async def _process_message(self, msg: StreamMessage) -> Any:
        """Process a single reranking task.

        Steps:
            1. Read retrieval results from state store (must exist).
            2. Attempt to read graph context (wait briefly, proceed without).
            3. Read classification data for the original query text.
            4. Delegate to RerankingService.rerank_and_process().
        """
        task_id = msg.task_id
        logger.info("Reranking task_id=%s (message_id=%s)", task_id, msg.message_id)

        # 1. Read retrieval results (required)
        retrieved_data = await self._state.get_result(task_id, "retrieved")
        if retrieved_data is None:
            msg_text = f"task_id={task_id}: no retrieval results found in state store"
            logger.error(msg_text)
            raise ValueError(msg_text)

        retrieved_chunks: list[dict] = (
            retrieved_data
            if isinstance(retrieved_data, list)
            else retrieved_data.get("results", [])
        )

        # 2. Read graph context (optional -- wait briefly then proceed without)
        graph_context = await self._wait_for_graph_context(task_id)

        # 3. Read classification for query text
        query_text = await self._get_query_text(task_id)

        # 4. Run reranking pipeline
        result = await self._service.rerank_and_process(
            task_id=task_id,
            query_text=query_text,
            retrieved_chunks=retrieved_chunks,
            graph_context=graph_context,
        )

        logger.info(
            "task_id=%s: reranking complete (primary=%d, referenced=%d, trimmed=%d)",
            task_id,
            result.get("primary_count", 0),
            result.get("referenced_count", 0),
            result.get("trimmed_count", 0),
        )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _wait_for_graph_context(self, task_id: str) -> dict | None:
        """Try to read graph context, polling briefly if not yet available."""
        wait_seconds = self._reranking_config.graph_context_wait_seconds
        poll_interval = 0.5
        elapsed = 0.0

        while elapsed < wait_seconds:
            graph_data = await self._state.get_result(task_id, "graph_context")
            if graph_data is not None:
                logger.info("task_id=%s: graph context found", task_id)
                return graph_data if isinstance(graph_data, dict) else {"summary": str(graph_data)}
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        logger.info("task_id=%s: proceeding without graph context (timeout after %.1fs)", task_id, wait_seconds)
        return None

    async def _get_query_text(self, task_id: str) -> str:
        """Read the original query text from classification results."""
        classification = await self._state.get_result(task_id, "classification")
        if classification is not None and isinstance(classification, dict):
            return classification.get("query_text", classification.get("query", ""))
        # Fallback: try to get query from metadata
        query_data = await self._state.get_result(task_id, "query")
        if query_data is not None and isinstance(query_data, dict):
            return query_data.get("text", query_data.get("query", ""))
        logger.warning("task_id=%s: no query text found, using empty string", task_id)
        return ""
