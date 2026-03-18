"""Synthesis service -- orchestrates answer generation and state persistence."""

from __future__ import annotations

import logging
from typing import Any

from rag_common.llm.client import LLMClient
from rag_common.redis.client import RedisClient
from rag_common.redis.state import StateStore

from agents.synthesis.synthesis_agent.config import SynthesisConfig
from agents.synthesis.synthesis_agent.generator import AnswerGenerator

logger = logging.getLogger(__name__)


class SynthesisService:
    """Coordinates context assembly, LLM generation, and result storage.

    Reads reranked chunks from the state store, delegates to
    :class:`AnswerGenerator`, and persists the answer back into Redis.
    """

    def __init__(
        self,
        redis_client: RedisClient,
        llm_client: LLMClient,
        config: SynthesisConfig,
    ) -> None:
        self._state = StateStore(redis_client)
        self._config = config
        self._generator = AnswerGenerator(llm_client, config)

    async def synthesize(
        self,
        task_id: str,
        query_text: str,
        reranked_data: dict[str, Any],
        intent: str,
        feedback: str | None = None,
    ) -> dict[str, Any]:
        """Generate an answer from reranked context and store the result.

        Args:
            task_id: Unique task identifier.
            query_text: The user's original question.
            reranked_data: Output from the reranking stage, containing
                ``primary_chunks``, ``referenced_chunks``, ``graph_context``,
                and ``summary``.
            intent: Detected query intent string.
            feedback: Optional revision feedback from a failed validation
                (triggers adjusted system prompt on retry).

        Returns:
            Dict with ``output_ref`` (Redis key) and ``citation_count``.
        """
        # 1. Extract chunks from reranked data
        primary_chunks: list[dict] = reranked_data.get("primary_chunks", [])
        referenced_chunks: list[dict] = reranked_data.get("referenced_chunks", [])
        combined_chunks = primary_chunks + referenced_chunks

        # Fallback: if structured keys missing, try flat "chunks" key
        if not combined_chunks:
            combined_chunks = reranked_data.get("chunks", [])

        # 2. Extract supplementary context
        graph_context: str | None = reranked_data.get("graph_context")
        summary: str | None = reranked_data.get("summary")

        # 3. Build system prompt override if revision feedback provided
        system_prompt_override: str | None = None
        if feedback:
            system_prompt_override = (
                f"{self._config.system_prompt}\n\n"
                f"Previous answer had issues: {feedback}. Please address them."
            )

        # 4. Generate answer
        result = await self._generator.generate(
            query=query_text,
            context_chunks=combined_chunks,
            graph_context=graph_context,
            summary=summary,
            intent=intent,
            system_prompt_override=system_prompt_override,
        )

        # 5. Store answer in state store
        answer_data = {
            "text": result["text"],
            "citations": result["citations"],
            "model": result["model"],
            "query_text": query_text,
            "intent": intent,
            "chunk_count": len(combined_chunks),
        }
        output_ref = await self._state.store_result(task_id, "answer", answer_data)

        logger.info(
            "task_id=%s: generated answer with %d citations, stored at %s",
            task_id,
            len(result["citations"]),
            output_ref,
        )

        return {
            "output_ref": output_ref,
            "citation_count": len(result["citations"]),
        }
