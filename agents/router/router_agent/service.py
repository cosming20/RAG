"""Router Agent gRPC service implementation."""

from __future__ import annotations

import hashlib
import logging
import time
from uuid import uuid4

import grpc

from rag_common.config import CacheConfig, LLMConfig, RedisConfig, get_cache_config
from rag_common.llm.client import LLMClient
from rag_common.models.document import DocumentFormat, DocumentMeta
from rag_common.models.task import Task, TaskStatus
from rag_common.redis.client import RedisClient
from rag_common.redis.query_cache import QueryCache
from rag_common.redis.state import StateStore
from rag_common.redis.streams import StreamProducer
from rag_common.redis.task_queue import TaskQueue

from .classifier import IntentClassifier
from .config import RouterConfig

logger = logging.getLogger(__name__)

# Stream topics
INGEST_PENDING = "rag:stream:ingest:pending"
QUERY_PENDING = "rag:stream:query:pending"
QUERY_CLASSIFIED = "rag:stream:query:classified"

# Document format mapping
FORMAT_MAP = {
    "pdf": DocumentFormat.PDF,
    "docx": DocumentFormat.DOCX,
    "pptx": DocumentFormat.PPTX,
    "xlsx": DocumentFormat.XLSX,
    "html": DocumentFormat.HTML,
    "md": DocumentFormat.MARKDOWN,
    "markdown": DocumentFormat.MARKDOWN,
    "txt": DocumentFormat.TXT,
    "csv": DocumentFormat.CSV,
    "tex": DocumentFormat.LATEX,
    "latex": DocumentFormat.LATEX,
}


class RouterService:
    """Router Agent business logic — query routing and document ingestion dispatch.

    This service is framework-agnostic; the gRPC servicer wraps it.
    """

    def __init__(
        self,
        redis_client: RedisClient,
        llm_client: LLMClient,
        config: RouterConfig | None = None,
        cache_config: CacheConfig | None = None,
    ) -> None:
        self._config = config or RouterConfig()
        self._redis = redis_client
        self._llm = llm_client
        self._state = StateStore(redis_client)
        self._producer = StreamProducer(redis_client)
        self._task_queue = TaskQueue(redis_client)
        self._classifier = IntentClassifier(llm_client, model=self._config.classifier_model)

        # Query cache (opt-in via CACHE_ENABLED=true)
        self._cache_config = cache_config or get_cache_config()
        self._query_cache: QueryCache | None = None
        if self._cache_config.enabled:
            self._query_cache = QueryCache(
                redis_client,
                similarity_threshold=self._cache_config.similarity_threshold,
                max_cached=self._cache_config.max_cached_queries,
                ttl=self._cache_config.ttl_seconds,
            )

    async def handle_query(
        self,
        session_id: str,
        query_text: str,
        *,
        intent_hint: str | None = None,
        max_results: int = 10,
        timeout_ms: int = 30000,
        trace_id: str = "",
        span_id: str = "",
    ) -> dict:
        """Handle an incoming query — classify intent and dispatch to pipeline.

        When caching is enabled, embeds the query and checks for a similar
        cached result before running the full classification + dispatch pipeline.
        On a cache miss the result is stored after dispatch so subsequent
        similar queries can be served instantly.
        """
        start = time.monotonic()

        # ── Cache lookup (before any pipeline work) ────────────────
        query_embedding: list[float] | None = None
        if self._query_cache is not None:
            try:
                query_embedding = await self._llm.embed_single(query_text)
                cached = await self._query_cache.lookup(query_embedding)
                if cached is not None:
                    elapsed = (time.monotonic() - start) * 1000
                    logger.info("Query served from cache (%.1fms)", elapsed)
                    cached["cache_hit"] = True
                    return cached
            except Exception:
                logger.warning("Cache lookup failed, proceeding without cache", exc_info=True)

        # ── Normal pipeline ────────────────────────────────────────
        task_id = str(uuid4())

        # Classify intent
        if intent_hint:
            from rag_common.models.query import QueryIntent
            intent = QueryIntent(intent_hint)
            confidence = 1.0
        else:
            intent, confidence = await self._classifier.classify(query_text)

        # Store classification
        await self._state.store_result(task_id, "classification", {
            "intent": intent.value,
            "confidence": confidence,
            "query_text": query_text,
            "session_id": session_id,
            "max_results": max_results,
        })

        # Create task
        task = Task(
            task_id=task_id,
            status=TaskStatus.PENDING,
            trace_id=trace_id,
            payload_ref=f"rag:results:{task_id}:classification",
        )
        await self._task_queue.create_task(task)

        # Dispatch to query pipeline
        deadline_ms = int((time.time() + timeout_ms / 1000) * 1000)
        await self._producer.publish(
            QUERY_CLASSIFIED,
            task_id,
            trace_id=trace_id,
            span_id=span_id,
            deadline_unix_ms=deadline_ms,
            payload_ref=f"rag:results:{task_id}:classification",
        )

        result = {
            "task_id": task_id,
            "detected_intent": intent.value,
            "confidence": confidence,
        }

        # ── Cache store (after dispatch) ───────────────────────────
        if self._query_cache is not None and query_embedding is not None:
            try:
                cache_key = hashlib.sha256(query_text.encode()).hexdigest()[:16]
                await self._query_cache.store(cache_key, query_embedding, result)
            except Exception:
                logger.warning("Cache store failed", exc_info=True)

        elapsed = (time.monotonic() - start) * 1000
        logger.info(
            "Query dispatched: task_id=%s intent=%s confidence=%.2f (%.1fms)",
            task_id, intent, confidence, elapsed,
        )
        return result

    async def handle_ingest(
        self,
        content: bytes,
        filename: str,
        format_str: str,
        *,
        metadata: dict[str, str] | None = None,
        trace_id: str = "",
        span_id: str = "",
    ) -> dict:
        """Handle document ingestion — validate, store, and dispatch to pipeline."""
        task_id = str(uuid4())
        document_id = str(uuid4())

        # Validate size
        max_size = self._config.max_document_size_bytes
        if len(content) > max_size:
            return {
                "success": False,
                "error": f"Document too large: {len(content)} bytes (max {max_size})",
            }

        # Detect format
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        doc_format = FORMAT_MAP.get(format_str.lower(), FORMAT_MAP.get(ext, DocumentFormat.TXT))

        # Compute content hash for dedup
        content_hash = hashlib.sha256(content).hexdigest()

        # Store raw document
        await self._state.store_raw(task_id, content)

        # Store document metadata
        doc_meta = {
            "document_id": document_id,
            "filename": filename,
            "format": doc_format.value,
            "size_bytes": len(content),
            "content_hash": content_hash,
            "metadata": metadata or {},
        }
        await self._state.store_result(task_id, "meta", doc_meta)

        # Create task
        task = Task(
            task_id=task_id,
            status=TaskStatus.PENDING,
            trace_id=trace_id,
            payload_ref=f"rag:results:{task_id}:raw",
        )
        await self._task_queue.create_task(task)

        # Dispatch to ingestion pipeline
        deadline_ms = int((time.time() + self._config.default_timeout_ms / 1000) * 1000)
        await self._producer.publish(
            INGEST_PENDING,
            task_id,
            trace_id=trace_id,
            span_id=span_id,
            deadline_unix_ms=deadline_ms,
            payload_ref=f"rag:results:{task_id}:raw",
            metadata={"document_id": document_id, "filename": filename},
        )

        logger.info(
            "Ingestion dispatched: task_id=%s document_id=%s filename=%s format=%s size=%d",
            task_id, document_id, filename, doc_format, len(content),
        )
        return {
            "success": True,
            "task_id": task_id,
            "document_id": document_id,
        }
