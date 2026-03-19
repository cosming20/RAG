"""Validation Agent gRPC servicer -- wraps ValidationService for gRPC transport.

Maps gRPC RPC methods to :class:`ValidationService` calls.  When proto
generation is complete, this class should inherit from the generated
``ValidationAgentServicer`` base class.

Proto service: ``rag.services.validation.ValidationAgent``
RPCs: Validate, CheckHallucination, VerifyAttribution
"""

from __future__ import annotations

import logging
import time
from typing import Any

import grpc

from .service import ValidationService

logger = logging.getLogger(__name__)


class ValidationServicer:
    """gRPC servicer for the Validation Agent.

    Delegates hallucination checking, attribution verification, and
    composite validation to :class:`ValidationService`.

    Once proto stubs are generated, update this class to inherit from
    ``validation_pb2_grpc.ValidationAgentServicer`` and return proper
    proto message objects.
    """

    def __init__(self, service: ValidationService) -> None:
        self._service = service

    async def Validate(
        self, request: Any, context: grpc.aio.ServicerContext,
    ) -> Any:
        """Run full validation (hallucination + attribution) on an answer.

        Proto: ValidateRequest -> ValidateResponse
        """
        start = time.monotonic()
        try:
            # Extract answer data
            answer_data: dict[str, Any] = {}
            if hasattr(request, "answer") and request.answer:
                a = request.answer
                answer_data["text"] = getattr(a, "text", "")
                citations: list[dict[str, Any]] = []
                if hasattr(a, "citations"):
                    for cit in a.citations:
                        citations.append({
                            "document_id": getattr(cit.document_id, "value", "") if hasattr(cit, "document_id") else "",
                            "filename": getattr(cit, "filename", ""),
                            "page_number": getattr(cit, "page_number", 0),
                            "section": getattr(cit, "section", ""),
                            "quoted_text": getattr(cit, "quoted_text", ""),
                        })
                answer_data["citations"] = citations

            # Extract source chunks
            source_chunks: list[dict[str, Any]] = []
            if hasattr(request, "source_chunks"):
                for rc in request.source_chunks:
                    chunk_data: dict[str, Any] = {}
                    if hasattr(rc, "chunk") and rc.chunk:
                        c = rc.chunk
                        chunk_data["chunk_id"] = getattr(c.chunk_id, "value", "") if hasattr(c, "chunk_id") else ""
                        chunk_data["text"] = getattr(c, "text", "")
                        chunk_data["document_id"] = getattr(c.document_id, "value", "") if hasattr(c, "document_id") else ""
                    chunk_data["relevance_score"] = getattr(rc, "relevance_score", 0.0)
                    source_chunks.append(chunk_data)

            query_text: str = (
                request.original_query if hasattr(request, "original_query") else ""
            )

            import uuid
            task_id = str(uuid.uuid4())

            result = await self._service.validate(
                task_id=task_id,
                answer_data=answer_data,
                source_chunks=source_chunks,
                query_text=query_text,
            )

            elapsed = (time.monotonic() - start) * 1000
            logger.info(
                "Validate completed in %.1fms (quality=%.2f, acceptable=%s)",
                elapsed,
                result.get("quality_score", 0.0),
                result.get("is_acceptable"),
            )

            return {"success": True, **result}

        except Exception as exc:
            logger.exception("Validate failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return None

    async def CheckHallucination(
        self, request: Any, context: grpc.aio.ServicerContext,
    ) -> Any:
        """Run hallucination check only on an answer.

        Proto: HallucinationCheckRequest -> HallucinationCheckResponse
        """
        start = time.monotonic()
        try:
            answer_text: str = ""
            if hasattr(request, "answer") and request.answer:
                answer_text = getattr(request.answer, "text", "")

            source_chunks: list[dict[str, Any]] = []
            if hasattr(request, "source_chunks"):
                for rc in request.source_chunks:
                    chunk_data: dict[str, Any] = {}
                    if hasattr(rc, "chunk") and rc.chunk:
                        chunk_data["text"] = getattr(rc.chunk, "text", "")
                    source_chunks.append(chunk_data)

            hallucination_result = await self._service._hallucination_checker.check(
                answer_text, source_chunks,
            )

            elapsed = (time.monotonic() - start) * 1000
            logger.info(
                "CheckHallucination completed in %.1fms (grounding=%.2f)",
                elapsed,
                hallucination_result.get("grounding_score", 0.0),
            )

            return {
                "success": True,
                "claims": hallucination_result.get("claims", []),
                "grounding_score": hallucination_result.get("grounding_score", 0.0),
            }

        except Exception as exc:
            logger.exception("CheckHallucination failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return None

    async def VerifyAttribution(
        self, request: Any, context: grpc.aio.ServicerContext,
    ) -> Any:
        """Verify citation attribution for an answer.

        Proto: AttributionCheckRequest -> AttributionCheckResponse
        """
        start = time.monotonic()
        try:
            answer_text: str = ""
            citations: list[dict[str, Any]] = []
            if hasattr(request, "answer") and request.answer:
                answer_text = getattr(request.answer, "text", "")
                if hasattr(request.answer, "citations"):
                    for cit in request.answer.citations:
                        citations.append({
                            "document_id": getattr(cit.document_id, "value", "") if hasattr(cit, "document_id") else "",
                            "filename": getattr(cit, "filename", ""),
                            "quoted_text": getattr(cit, "quoted_text", ""),
                        })

            source_chunks: list[dict[str, Any]] = []
            if hasattr(request, "source_chunks"):
                for rc in request.source_chunks:
                    chunk_data: dict[str, Any] = {}
                    if hasattr(rc, "chunk") and rc.chunk:
                        chunk_data["text"] = getattr(rc.chunk, "text", "")
                        chunk_data["chunk_id"] = getattr(rc.chunk.chunk_id, "value", "") if hasattr(rc.chunk, "chunk_id") else ""
                    source_chunks.append(chunk_data)

            attribution_result = self._service._attribution_checker.check(
                answer_text, citations, source_chunks,
            )

            elapsed = (time.monotonic() - start) * 1000
            logger.info(
                "VerifyAttribution completed in %.1fms (score=%.2f)",
                elapsed,
                attribution_result.get("attribution_score", 0.0),
            )

            return {
                "success": True,
                "citations": attribution_result.get("citations", []),
                "attribution_score": attribution_result.get("attribution_score", 0.0),
            }

        except Exception as exc:
            logger.exception("VerifyAttribution failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return None
