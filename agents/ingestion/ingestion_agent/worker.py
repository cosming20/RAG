"""Ingestion stream worker -- consumes from Redis Streams and processes documents.

Extends :class:`BaseStreamWorker` which provides the common consumer loop
with deadline checks, retry/DLQ logic, pipeline depth enforcement, and
idempotency checks.  This worker only implements ``_process_message()``
and ``_publish_success()`` with ingestion-specific domain logic.
"""

from __future__ import annotations

import base64
import logging
from typing import Any

from rag_common.redis.client import RedisClient
from rag_common.redis.streams import StreamMessage
from rag_common.redis.worker import BaseStreamWorker

from .config import IngestionConfig
from .service import IngestionService

logger = logging.getLogger(__name__)


class IngestionWorker(BaseStreamWorker):
    """Reads tasks from the ingestion pending stream and processes them.

    Consumer group: ``ingestion-agents``
    Input stream:   ``rag:stream:ingest:pending``
    Output stream:  ``rag:stream:ingest:parsed``
    """

    def __init__(
        self,
        redis_client: RedisClient,
        service: IngestionService,
        config: IngestionConfig,
        consumer_id: str,
    ) -> None:
        super().__init__(
            redis_client,
            input_stream=config.input_stream,
            output_stream=config.output_stream,
            dlq_stream=config.dlq_stream,
            consumer_group=config.consumer_group,
            consumer_id=consumer_id,
            output_stage="parsed",
        )
        self._service = service
        self._ingestion_config = config

    async def run(self) -> None:
        """Alias for :meth:`start` — preserves the existing call-site interface."""
        await self.start()

    async def _process_message(self, msg: StreamMessage) -> Any:
        """Parse an ingested document and store parsed content in StateStore.

        Steps:
            1. Load raw content from StateStore / payload_ref.
            2. Load document metadata.
            3. Call :meth:`IngestionService.process_document`.
            4. Return result dict (must contain ``parsed_key``).
        """
        task_id = msg.task_id
        logger.info("Processing task_id=%s attempt=%d", task_id, msg.attempt)

        # -- Retrieve raw content from StateStore ----------------------------
        raw_content = await self._load_raw_content(msg)
        if raw_content is None:
            raise ValueError(f"Raw content not found in StateStore for task {task_id}")

        # -- Retrieve document metadata --------------------------------------
        meta = await self._state.get_result(task_id, "meta")
        document_id = msg.metadata.get("document_id", "")
        filename = msg.metadata.get("filename", "unknown")
        fmt = "txt"  # default
        if meta:
            document_id = document_id or meta.get("document_id", "")
            filename = meta.get("filename", filename)
            fmt = meta.get("format", fmt)

        # -- Process ----------------------------------------------------------
        result = await self._service.process_document(
            task_id=task_id,
            document_id=document_id,
            content=raw_content,
            filename=filename,
            fmt=fmt,
        )

        if not result.get("success"):
            errors = "; ".join(result.get("errors", ["Unknown error"]))
            raise RuntimeError(f"Ingestion failed for task {task_id}: {errors}")

        return result

    async def _publish_success(self, msg: StreamMessage, result: Any) -> None:
        """Publish to the parsed stream with ingestion-specific metadata."""
        output_ref = result.get("parsed_key", "") if isinstance(result, dict) else ""
        await self._producer.publish(
            stream=self._ingestion_config.output_stream,
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
                "filename": msg.metadata.get("filename", ""),
            },
        )
        await self._task_queue.complete_task(
            msg.task_id, result_ref=result.get("parsed_key", "")
        )
        logger.info("Task %s completed and forwarded to parsed stream", msg.task_id)

    # ------------------------------------------------------------------ #
    #  Helpers                                                            #
    # ------------------------------------------------------------------ #

    async def _load_raw_content(self, msg: StreamMessage) -> bytes | None:
        """Load raw document bytes from the StateStore."""
        raw_b64 = await self._state.get_result(msg.task_id, "raw")
        if raw_b64 is None:
            # Try the payload_ref key directly
            if msg.payload_ref:
                data = await self._redis.client.get(msg.payload_ref)
                if data is not None:
                    try:
                        return base64.b64decode(data)
                    except Exception:
                        if isinstance(data, bytes):
                            return data
                        return data.encode("utf-8") if isinstance(data, str) else None
            return None
        # StateStore.get_raw stores base64-encoded content
        if isinstance(raw_b64, str):
            try:
                return base64.b64decode(raw_b64)
            except Exception:
                return raw_b64.encode("utf-8")
        return raw_b64  # type: ignore[return-value]
