"""Chunking stream worker -- consumes parsed documents and produces chunks.

Extends :class:`BaseStreamWorker` which provides the common consumer loop
with deadline checks, retry/DLQ logic, pipeline depth enforcement, and
idempotency checks.  This worker only implements ``_process_message()``
and ``_publish_success()`` with chunking-specific domain logic.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from rag_common.redis.client import RedisClient
from rag_common.redis.streams import StreamMessage
from rag_common.redis.worker import BaseStreamWorker

from .config import ChunkingConfig
from .service import ChunkingService

logger = logging.getLogger(__name__)


class ChunkingWorker(BaseStreamWorker):
    """Reads tasks from the parsed stream and chunks them.

    Consumer group: ``chunking-agents``
    Input stream:   ``rag:stream:ingest:parsed``
    Output stream:  ``rag:stream:ingest:chunked``
    """

    def __init__(
        self,
        redis_client: RedisClient,
        service: ChunkingService,
        config: ChunkingConfig,
        consumer_id: str,
    ) -> None:
        super().__init__(
            redis_client,
            input_stream=config.input_stream,
            output_stream=config.output_stream,
            dlq_stream=config.dlq_stream,
            consumer_group=config.consumer_group,
            consumer_id=consumer_id,
            output_stage="chunks",
        )
        self._service = service
        self._chunking_config = config

    async def run(self) -> None:
        """Alias for :meth:`start` — preserves the existing call-site interface."""
        await self.start()

    async def _process_message(self, msg: StreamMessage) -> Any:
        """Chunk a parsed document and store chunk data in StateStore.

        Steps:
            1. Load parsed content from StateStore / payload_ref.
            2. Extract text, document_id, format, metadata.
            3. Call :meth:`ChunkingService.chunk_document`.
            4. Return result dict (must contain ``chunks_key``).
        """
        task_id = msg.task_id
        logger.info("Processing task_id=%s attempt=%d", task_id, msg.attempt)

        # -- Load parsed content from StateStore -----------------------------
        parsed = await self._load_parsed_content(msg)
        if parsed is None:
            raise ValueError(f"Parsed content not found in StateStore for task {task_id}")

        # -- Extract fields ---------------------------------------------------
        parsed_text = parsed.get("text", "")
        document_id = msg.metadata.get("document_id", parsed.get("document_id", ""))
        fmt = parsed.get("format", "txt")
        doc_metadata = parsed.get("parser_metadata", {})

        # -- Process ----------------------------------------------------------
        result = await self._service.chunk_document(
            task_id=task_id,
            document_id=document_id,
            parsed_content=parsed_text,
            fmt=fmt,
            metadata=doc_metadata,
        )

        if not result.get("success"):
            errors = "; ".join(result.get("errors", ["Unknown error"]))
            raise RuntimeError(f"Chunking failed for task {task_id}: {errors}")

        return result

    async def _publish_success(self, msg: StreamMessage, result: Any) -> None:
        """Publish to the chunked stream with chunking-specific metadata."""
        output_ref = result.get("chunks_key", "") if isinstance(result, dict) else ""
        await self._producer.publish(
            stream=self._chunking_config.output_stream,
            task_id=msg.task_id,
            trace_id=msg.trace_id,
            span_id=msg.span_id,
            attempt=0,
            max_attempts=msg.max_attempts,
            deadline_unix_ms=msg.deadline_unix_ms,
            pipeline_depth=msg.pipeline_depth + 1,
            payload_ref=output_ref,
            metadata={
                "document_id": result.get("document_id", ""),
                "chunk_count": str(result.get("chunk_count", 0)),
            },
        )
        await self._task_queue.complete_task(
            msg.task_id, result_ref=result.get("chunks_key", "")
        )
        logger.info(
            "Task %s chunked (%d chunks) and forwarded to chunked stream",
            msg.task_id,
            result.get("chunk_count", 0),
        )

    # ------------------------------------------------------------------ #
    #  Helpers                                                            #
    # ------------------------------------------------------------------ #

    async def _load_parsed_content(
        self, msg: StreamMessage
    ) -> dict[str, Any] | None:
        """Load parsed document data from StateStore."""
        parsed = await self._state.get_result(msg.task_id, "parsed")
        if parsed is not None:
            return parsed

        # Try the payload_ref key directly (it may point to the parsed stage)
        if msg.payload_ref:
            data = await self._redis.client.get(msg.payload_ref)
            if data is not None:
                try:
                    return json.loads(data)
                except (json.JSONDecodeError, TypeError):
                    logger.warning(
                        "Could not parse payload_ref data for task %s",
                        msg.task_id,
                    )
        return None
