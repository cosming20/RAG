"""Router Agent gRPC servicer -- wraps RouterService for gRPC transport.

Maps gRPC RPC methods to :class:`RouterService` calls.  When proto
generation is complete, this class should inherit from the generated
``RouterAgentServicer`` base class.

Proto service: ``rag.services.router.RouterAgent``
RPCs: Query, QueryStream, IngestDocument, IngestDocumentStream, GetTaskStatus
"""

from __future__ import annotations

import logging
import time
from typing import Any

import grpc

from .service import RouterService

logger = logging.getLogger(__name__)


class RouterServicer:
    """gRPC servicer for the Router Agent.

    Delegates all business logic to :class:`RouterService`.

    Once proto stubs are generated, update this class to inherit from
    ``router_pb2_grpc.RouterAgentServicer`` and return proper proto
    message objects instead of raw dicts.
    """

    def __init__(self, service: RouterService) -> None:
        self._service = service

    async def IngestDocument(
        self, request: Any, context: grpc.aio.ServicerContext,
    ) -> Any:
        """Handle document ingestion request.

        Proto: IngestDocumentRequest -> IngestDocumentResponse
        """
        start = time.monotonic()
        try:
            content: bytes = request.content if hasattr(request, "content") else b""
            filename: str = request.filename if hasattr(request, "filename") else ""
            format_str: str = str(request.format) if hasattr(request, "format") else ""
            metadata: dict[str, str] = (
                dict(request.metadata) if hasattr(request, "metadata") else {}
            )
            trace_id: str = (
                request.trace_context.trace_id
                if hasattr(request, "trace_context") and request.trace_context
                else ""
            )
            span_id: str = (
                request.trace_context.span_id
                if hasattr(request, "trace_context") and request.trace_context
                else ""
            )

            result = await self._service.handle_ingest(
                content=content,
                filename=filename,
                format_str=format_str,
                metadata=metadata,
                trace_id=trace_id,
                span_id=span_id,
            )

            elapsed = (time.monotonic() - start) * 1000
            logger.info("IngestDocument completed in %.1fms", elapsed)
            return result

        except Exception as exc:
            logger.exception("IngestDocument failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return None

    async def Query(
        self, request: Any, context: grpc.aio.ServicerContext,
    ) -> Any:
        """Handle query request.

        Proto: QueryRequest -> QueryResponse
        """
        start = time.monotonic()
        try:
            session_id: str = (
                request.session_id.value
                if hasattr(request, "session_id") and request.session_id
                else ""
            )
            query_text: str = request.query_text if hasattr(request, "query_text") else ""
            intent_hint: str | None = (
                str(request.intent_hint) if hasattr(request, "intent_hint") and request.intent_hint else None
            )
            max_results: int = request.max_results if hasattr(request, "max_results") else 10
            timeout_ms: int = request.timeout_ms if hasattr(request, "timeout_ms") else 30000
            trace_id: str = (
                request.trace_context.trace_id
                if hasattr(request, "trace_context") and request.trace_context
                else ""
            )
            span_id: str = (
                request.trace_context.span_id
                if hasattr(request, "trace_context") and request.trace_context
                else ""
            )

            result = await self._service.handle_query(
                session_id=session_id,
                query_text=query_text,
                intent_hint=intent_hint,
                max_results=max_results,
                timeout_ms=timeout_ms,
                trace_id=trace_id,
                span_id=span_id,
            )

            elapsed = (time.monotonic() - start) * 1000
            logger.info("Query completed in %.1fms", elapsed)
            return result

        except Exception as exc:
            logger.exception("Query failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return None

    async def QueryStream(
        self, request: Any, context: grpc.aio.ServicerContext,
    ) -> None:
        """Handle streaming query request.

        Proto: QueryRequest -> stream QueryStreamEvent

        Delegates to the same handle_query method but streams partial
        results.  Full streaming will be implemented when proto stubs
        and streaming generator support are wired.
        """
        try:
            session_id = (
                request.session_id.value
                if hasattr(request, "session_id") and request.session_id
                else ""
            )
            query_text = request.query_text if hasattr(request, "query_text") else ""

            result = await self._service.handle_query(
                session_id=session_id,
                query_text=query_text,
            )
            # Yield a single complete result; full streaming TBD
            yield result  # type: ignore[misc]

        except Exception as exc:
            logger.exception("QueryStream failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))

    async def GetTaskStatus(
        self, request: Any, context: grpc.aio.ServicerContext,
    ) -> Any:
        """Get task status by task ID.

        Proto: rag.common.TaskId -> rag.common.TaskMeta
        """
        try:
            task_id: str = request.value if hasattr(request, "value") else ""
            if not task_id:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("task_id is required")
                return None

            task = await self._service._task_queue.get_task(task_id)
            if task is None:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(f"Task {task_id} not found")
                return None

            return task

        except Exception as exc:
            logger.exception("GetTaskStatus failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return None
