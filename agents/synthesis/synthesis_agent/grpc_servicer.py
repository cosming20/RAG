"""Synthesis Agent gRPC servicer -- wraps SynthesisService for gRPC transport.

Maps gRPC RPC methods to :class:`SynthesisService` calls.  When proto
generation is complete, this class should inherit from the generated
``SynthesisAgentServicer`` base class.

Proto service: ``rag.services.synthesis.SynthesisAgent``
RPCs: Synthesize, SynthesizeStream
"""

from __future__ import annotations

import logging
import time
from typing import Any

import grpc

from .service import SynthesisService

logger = logging.getLogger(__name__)


class SynthesisServicer:
    """gRPC servicer for the Synthesis Agent.

    Delegates answer generation to :class:`SynthesisService`.

    Once proto stubs are generated, update this class to inherit from
    ``synthesis_pb2_grpc.SynthesisAgentServicer`` and return proper
    proto message objects.
    """

    def __init__(self, service: SynthesisService) -> None:
        self._service = service

    async def Synthesize(
        self, request: Any, context: grpc.aio.ServicerContext,
    ) -> Any:
        """Generate a cited answer from context chunks and query.

        Proto: SynthesizeRequest -> SynthesizeResponse
        """
        start = time.monotonic()
        try:
            query_text: str = request.query_text if hasattr(request, "query_text") else ""
            intent: str = str(request.intent) if hasattr(request, "intent") else ""
            graph_context: str | None = (
                request.graph_context if hasattr(request, "graph_context") and request.graph_context else None
            )
            document_summary: str | None = (
                request.document_summary if hasattr(request, "document_summary") and request.document_summary else None
            )
            system_prompt_override: str | None = (
                request.system_prompt_override
                if hasattr(request, "system_prompt_override") and request.system_prompt_override
                else None
            )

            # Build reranked_data dict from context_chunks
            context_chunks: list[dict[str, Any]] = []
            if hasattr(request, "context_chunks"):
                for rc in request.context_chunks:
                    chunk_data: dict[str, Any] = {}
                    if hasattr(rc, "chunk") and rc.chunk:
                        c = rc.chunk
                        chunk_data["chunk_id"] = getattr(c.chunk_id, "value", "") if hasattr(c, "chunk_id") else ""
                        chunk_data["text"] = getattr(c, "text", "")
                        chunk_data["document_id"] = getattr(c.document_id, "value", "") if hasattr(c, "document_id") else ""
                        chunk_data["sequence_index"] = getattr(c, "sequence_index", 0)
                    chunk_data["relevance_score"] = getattr(rc, "relevance_score", 0.0)
                    context_chunks.append(chunk_data)

            reranked_data: dict[str, Any] = {
                "primary_chunks": context_chunks,
                "referenced_chunks": [],
                "graph_context": graph_context,
                "summary": document_summary,
            }

            import uuid
            task_id = str(uuid.uuid4())

            result = await self._service.synthesize(
                task_id=task_id,
                query_text=query_text,
                reranked_data=reranked_data,
                intent=intent,
                feedback=system_prompt_override,
            )

            elapsed = (time.monotonic() - start) * 1000
            logger.info(
                "Synthesize completed in %.1fms (%d citations)",
                elapsed,
                result.get("citation_count", 0),
            )

            return {"success": True, **result}

        except Exception as exc:
            logger.exception("Synthesize failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return None

    async def SynthesizeStream(
        self, request: Any, context: grpc.aio.ServicerContext,
    ) -> None:
        """Generate a streaming answer with token-by-token output.

        Proto: SynthesizeRequest -> stream SynthesizeStreamEvent

        Full streaming generation will be implemented when the
        AnswerGenerator supports token callbacks.  For now, generates
        the full answer and yields a single complete event.
        """
        try:
            query_text: str = request.query_text if hasattr(request, "query_text") else ""
            intent: str = str(request.intent) if hasattr(request, "intent") else ""

            context_chunks: list[dict[str, Any]] = []
            if hasattr(request, "context_chunks"):
                for rc in request.context_chunks:
                    chunk_data: dict[str, Any] = {}
                    if hasattr(rc, "chunk") and rc.chunk:
                        c = rc.chunk
                        chunk_data["text"] = getattr(c, "text", "")
                    chunk_data["relevance_score"] = getattr(rc, "relevance_score", 0.0)
                    context_chunks.append(chunk_data)

            reranked_data: dict[str, Any] = {
                "primary_chunks": context_chunks,
                "referenced_chunks": [],
            }

            import uuid
            task_id = str(uuid.uuid4())

            result = await self._service.synthesize(
                task_id=task_id,
                query_text=query_text,
                reranked_data=reranked_data,
                intent=intent,
            )

            # Yield the complete answer as a single event
            yield {"complete_answer": result}  # type: ignore[misc]

        except Exception as exc:
            logger.exception("SynthesizeStream failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
