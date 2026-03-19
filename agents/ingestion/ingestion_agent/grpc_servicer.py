"""Ingestion Agent gRPC servicer -- wraps IngestionService for gRPC transport.

Maps gRPC RPC methods to :class:`IngestionService` calls.  When proto
generation is complete, this class should inherit from the generated
``IngestionAgentServicer`` base class.

Proto service: ``rag.services.ingestion.IngestionAgent``
RPCs: ParseDocument, ValidateDocument
"""

from __future__ import annotations

import logging
import time
from typing import Any

import grpc

from .preflight import PreflightValidator
from .service import IngestionService

logger = logging.getLogger(__name__)


class IngestionServicer:
    """gRPC servicer for the Ingestion Agent.

    Delegates parsing and validation to :class:`IngestionService` and
    :class:`PreflightValidator`.

    Once proto stubs are generated, update this class to inherit from
    ``ingestion_pb2_grpc.IngestionAgentServicer`` and return proper
    proto message objects.
    """

    def __init__(self, service: IngestionService) -> None:
        self._service = service

    async def ParseDocument(
        self, request: Any, context: grpc.aio.ServicerContext,
    ) -> Any:
        """Parse a document into structured text content.

        Proto: ParseDocumentRequest -> ParseDocumentResponse
        """
        start = time.monotonic()
        try:
            document_id: str = (
                request.document_id.value
                if hasattr(request, "document_id") and request.document_id
                else ""
            )
            content: bytes = request.content if hasattr(request, "content") else b""
            filename: str = request.filename if hasattr(request, "filename") else ""
            fmt: str = str(request.format) if hasattr(request, "format") else "txt"
            trace_id: str = (
                request.trace_context.trace_id
                if hasattr(request, "trace_context") and request.trace_context
                else ""
            )

            # Generate a task_id for the parse operation
            import uuid
            task_id = str(uuid.uuid4())

            result = await self._service.process_document(
                task_id=task_id,
                document_id=document_id,
                content=content,
                filename=filename,
                fmt=fmt,
            )

            elapsed = (time.monotonic() - start) * 1000
            logger.info(
                "ParseDocument completed in %.1fms (task_id=%s, success=%s)",
                elapsed,
                task_id,
                result.get("success"),
            )
            return result

        except Exception as exc:
            logger.exception("ParseDocument failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return None

    async def ValidateDocument(
        self, request: Any, context: grpc.aio.ServicerContext,
    ) -> Any:
        """Validate a document before ingestion (preflight check).

        Proto: ValidateDocumentRequest -> ValidateDocumentResponse
        """
        start = time.monotonic()
        try:
            content: bytes = request.content if hasattr(request, "content") else b""
            filename: str = request.filename if hasattr(request, "filename") else ""
            fmt: str = str(request.format) if hasattr(request, "format") else "txt"

            from rag_common.models.document import DocumentFormat
            try:
                doc_format = DocumentFormat(fmt)
            except ValueError:
                doc_format = DocumentFormat.TXT

            is_valid, errors = self._service._validator.validate(
                content, filename, doc_format,
            )

            elapsed = (time.monotonic() - start) * 1000
            logger.info(
                "ValidateDocument completed in %.1fms (valid=%s)",
                elapsed,
                is_valid,
            )

            return {
                "is_valid": is_valid,
                "validation_errors": errors,
                "detected_format": doc_format.value,
                "size_bytes": len(content),
            }

        except Exception as exc:
            logger.exception("ValidateDocument failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return None
