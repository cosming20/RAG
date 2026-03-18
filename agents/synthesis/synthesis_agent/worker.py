"""Synthesis stream worker -- extends BaseStreamWorker.

Consumes from ``rag:stream:query:reranked``, generates LLM answers with
citations, and publishes to ``rag:stream:query:synthesized``.
"""

from __future__ import annotations

import logging
from typing import Any

from rag_common.config import AgentConfig
from rag_common.redis.client import RedisClient
from rag_common.redis.streams import StreamMessage
from rag_common.redis.worker import BaseStreamWorker

from agents.synthesis.synthesis_agent.service import SynthesisService

logger = logging.getLogger(__name__)

INPUT_STREAM = "rag:stream:query:reranked"
OUTPUT_STREAM = "rag:stream:query:synthesized"
DLQ_STREAM = "rag:stream:synthesis:dlq"
CONSUMER_GROUP = "synthesis-agents"


class SynthesisWorker(BaseStreamWorker):
    """Consumes reranked context and generates cited answers.

    Reads reranked data and classification results from
    :class:`StateStore`, delegates to :class:`SynthesisService`,
    and publishes the answer reference downstream.
    """

    def __init__(
        self,
        redis_client: RedisClient,
        service: SynthesisService,
        *,
        consumer_id: str,
        agent_config: AgentConfig | None = None,
    ) -> None:
        super().__init__(
            redis_client,
            input_stream=INPUT_STREAM,
            output_stream=OUTPUT_STREAM,
            dlq_stream=DLQ_STREAM,
            consumer_group=CONSUMER_GROUP,
            consumer_id=consumer_id,
            agent_config=agent_config,
        )
        self._service = service

    async def _process_message(self, msg: StreamMessage) -> Any:
        """Process a single reranked message and generate an answer.

        Steps:
            1. Read reranked data from StateStore.
            2. Read classification for query text and intent.
            3. Check for revision_feedback in metadata (from validation retry).
            4. Call SynthesisService.synthesize().

        Returns:
            Result dict with ``output_ref`` for downstream publish.
        """
        task_id = msg.task_id
        logger.info("Synthesizing answer for task_id=%s", task_id)

        # 1. Read reranked data
        reranked_data = await self._state.get_result(task_id, "reranked")
        if reranked_data is None:
            msg_text = f"task_id={task_id}: no reranked data found in state store"
            logger.error(msg_text)
            raise ValueError(msg_text)

        # 2. Read classification for query text and intent
        classification = await self._state.get_result(task_id, "classification")
        query_text = ""
        intent = "factual"
        if classification and isinstance(classification, dict):
            query_text = classification.get("query_text", "")
            intent = classification.get("intent", "factual")

        # Fallback: try metadata for query text
        if not query_text:
            query_text = msg.metadata.get("query_text", "")

        # 3. Check for revision feedback from validation retry
        revision_feedback = msg.metadata.get("revision_feedback")

        # 4. Generate answer
        result = await self._service.synthesize(
            task_id=task_id,
            query_text=query_text,
            reranked_data=reranked_data,
            intent=intent,
            feedback=revision_feedback,
        )

        logger.info(
            "task_id=%s: synthesis complete, %d citations",
            task_id,
            result.get("citation_count", 0),
        )

        return result
