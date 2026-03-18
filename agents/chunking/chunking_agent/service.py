"""Chunking Agent business logic -- framework-agnostic service layer."""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from rag_common.llm.client import LLMClient
from rag_common.models.document import DocumentFormat
from rag_common.redis.client import RedisClient
from rag_common.redis.state import StateStore
from rag_common.redis.task_queue import TaskQueue

from .config import ChunkingConfig
from .normalizer import ChunkNormalizer
from .strategies import select_strategy
from .summarizer import DocumentSummarizer

logger = logging.getLogger(__name__)


class ChunkingService:
    """Orchestrates document chunking: strategy selection, normalization, summarization.

    This service is framework-agnostic.  It is called by the stream
    worker and could equally be invoked from a gRPC handler.
    """

    def __init__(
        self,
        redis_client: RedisClient,
        llm_client: LLMClient,
        config: ChunkingConfig,
    ) -> None:
        self._redis = redis_client
        self._config = config
        self._state = StateStore(redis_client)
        self._task_queue = TaskQueue(redis_client)
        self._normalizer = ChunkNormalizer(llm_client, config)
        self._summarizer = DocumentSummarizer(llm_client)

    async def chunk_document(
        self,
        task_id: str,
        document_id: str,
        parsed_content: str,
        fmt: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Run full chunking pipeline for a single parsed document.

        Steps:
            1. Select chunking strategy based on format
            2. Chunk the content
            3. Generate deterministic chunk IDs
            4. Optionally normalize chunks via LLM
            5. Optionally generate document summary
            6. Store results in StateStore
            7. Return success dict

        Returns:
            A dict with ``success``, ``task_id``, ``chunk_count``, and ``chunks_key``.
        """
        # Resolve format
        try:
            doc_format = DocumentFormat(fmt)
        except ValueError:
            doc_format = DocumentFormat.TXT

        # 1. Select strategy
        strategy = select_strategy(doc_format, self._config)
        logger.info(
            "Chunking task_id=%s with strategy=%s",
            task_id,
            type(strategy).__name__,
        )

        # 2. Chunk
        try:
            chunks = strategy.chunk(parsed_content, metadata)
        except Exception as exc:
            error_msg = f"Chunking failed: {exc}"
            logger.error("Task %s: %s", task_id, error_msg, exc_info=True)
            await self._task_queue.fail_task(task_id, error_msg)
            return {"success": False, "task_id": task_id, "errors": [error_msg]}

        if not chunks:
            error_msg = "Chunking produced zero chunks"
            logger.warning("Task %s: %s", task_id, error_msg)
            await self._task_queue.fail_task(task_id, error_msg)
            return {"success": False, "task_id": task_id, "errors": [error_msg]}

        # 3. Generate deterministic chunk IDs
        for chunk in chunks:
            chunk["chunk_id"] = self._compute_chunk_id(
                document_id, chunk["sequence_index"]
            )

        # 4. Normalize (optional)
        if self._config.normalize_enabled:
            try:
                chunks = await self._normalizer.normalize(chunks)
            except Exception:
                logger.warning(
                    "Task %s normalization failed, keeping raw chunks",
                    task_id,
                    exc_info=True,
                )

        # 5. Summarize (optional)
        summary_data: dict[str, Any] = {}
        if self._config.summarize_enabled:
            try:
                summary_data = await self._summarizer.summarize(
                    parsed_content, chunks
                )
            except Exception:
                logger.warning(
                    "Task %s summarization failed", task_id, exc_info=True
                )

        # 6. Store results
        result_payload = {
            "document_id": document_id,
            "format": fmt,
            "chunk_count": len(chunks),
            "chunks": chunks,
            "summary": summary_data.get("summary", ""),
            "key_topics": summary_data.get("key_topics", []),
        }
        chunks_key = await self._state.store_result(task_id, "chunks", result_payload)

        logger.info(
            "Chunking complete: task_id=%s document_id=%s chunk_count=%d",
            task_id,
            document_id,
            len(chunks),
        )

        return {
            "success": True,
            "task_id": task_id,
            "document_id": document_id,
            "chunk_count": len(chunks),
            "chunks_key": chunks_key,
        }

    @staticmethod
    def _compute_chunk_id(document_id: str, sequence_index: int) -> str:
        """Generate a deterministic chunk ID from document_id and sequence index."""
        raw = f"{document_id}:{sequence_index}"
        return hashlib.sha256(raw.encode()).hexdigest()[:24]
