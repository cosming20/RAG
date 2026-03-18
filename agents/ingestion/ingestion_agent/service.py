"""Ingestion Agent business logic -- framework-agnostic service layer."""

from __future__ import annotations

import logging
from typing import Any

from rag_common.models.document import DocumentFormat
from rag_common.redis.client import RedisClient
from rag_common.redis.state import StateStore
from rag_common.redis.task_queue import TaskQueue

from .config import IngestionConfig
from .preflight import PreflightValidator
from .processor import DocumentProcessor

logger = logging.getLogger(__name__)


class IngestionService:
    """Orchestrates document ingestion: validate, parse, store.

    This service is framework-agnostic.  It is called by the stream
    worker and could equally be invoked from a gRPC handler.
    """

    def __init__(
        self,
        redis_client: RedisClient,
        config: IngestionConfig,
    ) -> None:
        self._redis = redis_client
        self._config = config
        self._state = StateStore(redis_client)
        self._task_queue = TaskQueue(redis_client)
        self._validator = PreflightValidator(config)
        self._processor = DocumentProcessor(ocr_enabled=config.ocr_enabled)

    async def process_document(
        self,
        task_id: str,
        document_id: str,
        content: bytes,
        filename: str,
        fmt: str,
    ) -> dict[str, Any]:
        """Run full ingestion pipeline for a single document.

        Steps:
            1. Preflight validation
            2. Document parsing (Docling or fallback)
            3. Store parsed result in StateStore
            4. Return success/failure dict

        Returns:
            A dict with ``success``, ``task_id``, and either ``parsed_key``
            on success or ``errors`` on failure.
        """
        # Resolve format enum
        try:
            doc_format = DocumentFormat(fmt)
        except ValueError:
            error_msg = f"Unknown document format: {fmt}"
            logger.error(error_msg)
            await self._task_queue.fail_task(task_id, error_msg)
            return {"success": False, "task_id": task_id, "errors": [error_msg]}

        # 1. Preflight validation
        is_valid, errors = self._validator.validate(content, filename, doc_format)
        if not is_valid:
            error_msg = f"Preflight failed: {'; '.join(errors)}"
            logger.warning("Task %s failed preflight: %s", task_id, error_msg)
            await self._task_queue.fail_task(task_id, error_msg)
            return {"success": False, "task_id": task_id, "errors": errors}

        # 2. Parse document
        try:
            parsed = self._processor.process(content, filename, doc_format)
        except Exception as exc:
            error_msg = f"Document parsing failed: {exc}"
            logger.error("Task %s parse error: %s", task_id, error_msg, exc_info=True)
            await self._task_queue.fail_task(task_id, error_msg)
            return {"success": False, "task_id": task_id, "errors": [error_msg]}

        # 3. Store parsed result
        parsed_data = {
            "document_id": document_id,
            "filename": filename,
            "format": fmt,
            "text": parsed["text"],
            "tables": parsed["tables"],
            "images": parsed["images"],
            "parser_metadata": parsed["metadata"],
        }
        parsed_key = await self._state.store_result(task_id, "parsed", parsed_data)

        logger.info(
            "Ingestion complete: task_id=%s document_id=%s text_len=%d tables=%d",
            task_id,
            document_id,
            len(parsed.get("text", "")),
            len(parsed.get("tables", [])),
        )

        return {
            "success": True,
            "task_id": task_id,
            "document_id": document_id,
            "parsed_key": parsed_key,
        }
