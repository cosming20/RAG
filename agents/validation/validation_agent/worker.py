"""Validation stream worker -- extends BaseStreamWorker.

Consumes from ``rag:stream:query:synthesized``, validates answer quality,
and either publishes to ``rag:stream:query:validated`` (pass) or retries
via the synthesis stream (fail).
"""

from __future__ import annotations

import logging
from typing import Any

from rag_common.config import AgentConfig
from rag_common.redis.client import RedisClient
from rag_common.redis.streams import StreamMessage
from rag_common.redis.worker import BaseStreamWorker

from agents.validation.validation_agent.config import ValidationConfig
from agents.validation.validation_agent.service import ValidationService

logger = logging.getLogger(__name__)

INPUT_STREAM = "rag:stream:query:synthesized"
OUTPUT_STREAM = "rag:stream:query:validated"
DLQ_STREAM = "rag:stream:validation:dlq"
CONSUMER_GROUP = "validation-agents"

# Stream to publish retries back to synthesis
RETRY_STREAM = "rag:stream:query:reranked"


class ValidationWorker(BaseStreamWorker):
    """Consumes synthesized answers and validates quality.

    If quality is below threshold and retries remain, publishes the
    task back to the reranked stream with revision feedback so the
    synthesis agent can regenerate. Otherwise, publishes to the
    validated stream.
    """

    def __init__(
        self,
        redis_client: RedisClient,
        service: ValidationService,
        validation_config: ValidationConfig,
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
        self._validation_config = validation_config

    async def _process_message(self, msg: StreamMessage) -> Any:
        """Validate a synthesized answer.

        Steps:
            1. Read answer from StateStore.
            2. Read reranked chunks for source verification.
            3. Read classification for query text.
            4. Call ValidationService.validate().
            5. If not acceptable and retries remain, publish back to
               synthesis via the reranked stream.

        Returns:
            Result dict with ``output_ref`` for downstream publish.
        """
        task_id = msg.task_id
        logger.info("Validating answer for task_id=%s", task_id)

        # 1. Read answer data
        answer_data = await self._state.get_result(task_id, "answer")
        if answer_data is None:
            msg_text = f"task_id={task_id}: no answer data found in state store"
            logger.error(msg_text)
            raise ValueError(msg_text)

        # 2. Read reranked chunks as source material
        reranked_data = await self._state.get_result(task_id, "reranked")
        source_chunks: list[dict] = []
        if reranked_data and isinstance(reranked_data, dict):
            primary = reranked_data.get("primary_chunks", [])
            referenced = reranked_data.get("referenced_chunks", [])
            source_chunks = primary + referenced
            if not source_chunks:
                source_chunks = reranked_data.get("chunks", [])

        # 3. Read classification for query text
        classification = await self._state.get_result(task_id, "classification")
        query_text = ""
        if classification and isinstance(classification, dict):
            query_text = classification.get("query_text", "")
        if not query_text:
            query_text = msg.metadata.get("query_text", "")

        # 4. Validate
        result = await self._service.validate(
            task_id=task_id,
            answer_data=answer_data,
            source_chunks=source_chunks,
            query_text=query_text,
        )

        # 5. Handle retry if not acceptable
        if not result["is_acceptable"]:
            attempt = int(msg.metadata.get("validation_attempt", "0"))
            if attempt < self._validation_config.max_retries:
                logger.info(
                    "task_id=%s: quality %.2f below threshold %.2f, "
                    "retrying synthesis (attempt %d/%d)",
                    task_id,
                    result["quality_score"],
                    self._validation_config.quality_threshold,
                    attempt + 1,
                    self._validation_config.max_retries,
                )
                await self._publish_retry(msg, result, attempt)
                return result

            logger.warning(
                "task_id=%s: quality %.2f below threshold after max retries, "
                "accepting as-is",
                task_id,
                result["quality_score"],
            )

        return result

    async def _publish_retry(
        self,
        msg: StreamMessage,
        result: dict[str, Any],
        current_attempt: int,
    ) -> None:
        """Publish the task back to the reranked stream for synthesis retry.

        Includes revision feedback in metadata so the synthesis agent
        can adjust its generation.
        """
        metadata = {
            **msg.metadata,
            "revision_feedback": result.get("revision_feedback", ""),
            "validation_attempt": str(current_attempt + 1),
            "quality_score": str(result.get("quality_score", 0.0)),
        }

        await self._producer.publish(
            RETRY_STREAM,
            msg.task_id,
            trace_id=msg.trace_id,
            span_id=msg.span_id,
            attempt=msg.attempt,
            max_attempts=msg.max_attempts,
            deadline_unix_ms=msg.deadline_unix_ms,
            pipeline_depth=msg.pipeline_depth,
            payload_ref=msg.payload_ref,
            metadata=metadata,
        )

    async def _publish_success(self, msg: StreamMessage, result: Any) -> None:
        """Override to handle retry vs. forward logic.

        If the validation result indicates a retry was published,
        skip the normal downstream publish.
        """
        if isinstance(result, dict) and not result.get("is_acceptable", True):
            attempt = int(msg.metadata.get("validation_attempt", "0"))
            if attempt < self._validation_config.max_retries:
                # Retry was already published in _process_message
                return

        # Normal forward to validated stream
        await super()._publish_success(msg, result)
