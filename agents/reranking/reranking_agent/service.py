"""Reranking service -- orchestrates Jina reranking, LLM trimming, and reference resolution."""

from __future__ import annotations

import logging
from typing import Any

from rag_common.config import JinaConfig, LLMConfig
from rag_common.llm.client import LLMClient
from rag_common.redis.client import RedisClient
from rag_common.redis.state import StateStore

from agents.reranking.reranking_agent.config import RerankingConfig
from agents.reranking.reranking_agent.jina_reranker import JinaReranker
from agents.reranking.reranking_agent.reference_resolver import ReferenceResolver
from agents.reranking.reranking_agent.trimmer import RelevanceTrimmer

logger = logging.getLogger(__name__)


class RerankingService:
    """Orchestrates the full reranking pipeline.

    Steps:
        1. Merge retrieval chunks + graph context chunks.
        2. Deduplicate by ``chunk_id``.
        3. Jina rerank against the query.
        4. LLM relevance trimming.
        5. Reference resolution for cross-document context.
        6. Store final result in the state store.
    """

    def __init__(
        self,
        redis_client: RedisClient,
        llm_client: LLMClient,
        config: RerankingConfig,
        jina_config: JinaConfig,
    ) -> None:
        self._state = StateStore(redis_client)
        self._reranker = JinaReranker(jina_config)
        self._trimmer = RelevanceTrimmer(llm_client, model=config.trimmer_model)
        self._resolver = ReferenceResolver(
            self._state,
            llm_client,
            model=config.trimmer_model,
        )
        self._config = config

    async def rerank_and_process(
        self,
        task_id: str,
        query_text: str,
        retrieved_chunks: list[dict[str, Any]],
        graph_context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Run the full reranking pipeline.

        Args:
            task_id: Pipeline task identifier.
            query_text: The original user query.
            retrieved_chunks: Chunks from the Retrieval agent.
            graph_context: Optional context from the Graph agent.

        Returns:
            A dict with ``output_ref``, ``primary_count``, and ``referenced_count``.
        """
        # 1. Merge retrieved chunks + graph context chunks
        merged = self._merge_chunks(retrieved_chunks, graph_context)
        logger.info("task_id=%s: merged %d chunks (retrieved=%d)", task_id, len(merged), len(retrieved_chunks))

        # 2. Deduplicate by chunk_id
        deduped = self._deduplicate(merged)
        logger.info("task_id=%s: %d unique chunks after dedup", task_id, len(deduped))

        # 3. Jina rerank
        reranked = await self._reranker.rerank(
            query_text,
            deduped,
            top_n=self._config.top_n,
        )
        logger.info("task_id=%s: Jina returned %d reranked chunks", task_id, len(reranked))

        # 4. Relevance trimming
        relevant, trimmed = await self._trimmer.trim(
            query_text,
            reranked,
            threshold=self._config.relevance_threshold,
            batch_size=self._config.trimmer_batch_size,
        )

        # 5. Reference resolution
        referenced = await self._resolver.resolve(
            relevant,
            max_depth=self._config.max_reference_depth,
            max_references=self._config.max_references,
        )

        # 6. Store final result
        graph_summary = ""
        if graph_context is not None:
            graph_summary = graph_context.get("context_text", str(graph_context))

        result_data = {
            "primary_chunks": relevant,
            "referenced_chunks": referenced,
            "graph_context": graph_summary,
            "trimmed_count": len(trimmed),
        }

        output_key = await self._state.store_result(task_id, "reranked", result_data)
        logger.info(
            "task_id=%s: stored reranked result (%d primary, %d referenced, %d trimmed) at %s",
            task_id,
            len(relevant),
            len(referenced),
            len(trimmed),
            output_key,
        )

        return {
            "output_ref": output_key,
            "primary_count": len(relevant),
            "referenced_count": len(referenced),
            "trimmed_count": len(trimmed),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_chunks(
        retrieved_chunks: list[dict[str, Any]],
        graph_context: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """Combine retrieval chunks with any chunks from graph context."""
        merged = list(retrieved_chunks)
        if graph_context is not None:
            graph_chunks = graph_context.get("related_chunks", [])
            if isinstance(graph_chunks, list):
                merged.extend(graph_chunks)
        return merged

    @staticmethod
    def _deduplicate(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Remove duplicate chunks by chunk_id, preserving first occurrence order."""
        seen: set[str] = set()
        unique: list[dict[str, Any]] = []
        for chunk in chunks:
            chunk_id = chunk.get("chunk_id", "")
            if not chunk_id or chunk_id not in seen:
                if chunk_id:
                    seen.add(chunk_id)
                unique.append(chunk)
        return unique
