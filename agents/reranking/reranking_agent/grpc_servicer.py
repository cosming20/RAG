"""Reranking Agent gRPC servicer -- wraps RerankingService for gRPC transport.

Maps gRPC RPC methods to service layer calls.  When proto generation
is complete, this class should inherit from the generated
``RerankingAgentServicer`` base class.

Proto service: ``rag.services.reranking.RerankingAgent``
RPCs: Rerank, TrimRelevance, ResolveReferences
"""

from __future__ import annotations

import logging
import time
from typing import Any

import grpc

from .service import RerankingService

logger = logging.getLogger(__name__)


class RerankingServicer:
    """gRPC servicer for the Reranking Agent.

    Delegates reranking, trimming, and reference resolution to
    :class:`RerankingService` and its sub-components.

    Once proto stubs are generated, update this class to inherit from
    ``reranking_pb2_grpc.RerankingAgentServicer`` and return proper
    proto message objects.
    """

    def __init__(self, service: RerankingService) -> None:
        self._service = service

    async def Rerank(
        self, request: Any, context: grpc.aio.ServicerContext,
    ) -> Any:
        """Rerank retrieved chunks against a query using Jina.

        Proto: RerankRequest -> RerankResponse
        """
        start = time.monotonic()
        try:
            query_text: str = request.query_text if hasattr(request, "query_text") else ""
            top_n: int = request.top_n if hasattr(request, "top_n") and request.top_n else 10

            chunks: list[dict[str, Any]] = []
            if hasattr(request, "chunks"):
                for rc in request.chunks:
                    chunk_data: dict[str, Any] = {}
                    if hasattr(rc, "chunk") and rc.chunk:
                        c = rc.chunk
                        chunk_data["chunk_id"] = getattr(c.chunk_id, "value", "") if hasattr(c, "chunk_id") else ""
                        chunk_data["text"] = getattr(c, "text", "")
                        chunk_data["document_id"] = getattr(c.document_id, "value", "") if hasattr(c, "document_id") else ""
                    chunk_data["relevance_score"] = getattr(rc, "relevance_score", 0.0)
                    chunks.append(chunk_data)

            if not query_text:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("query_text is required")
                return None

            # Rerank via the Jina reranker
            reranked = await self._service._reranker.rerank(
                query_text, chunks, top_n=top_n,
            )

            elapsed = (time.monotonic() - start) * 1000
            logger.info("Rerank completed in %.1fms (%d results)", elapsed, len(reranked))

            return {"success": True, "ranked_chunks": reranked}

        except Exception as exc:
            logger.exception("Rerank failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return None

    async def TrimRelevance(
        self, request: Any, context: grpc.aio.ServicerContext,
    ) -> Any:
        """Trim low-relevance chunks using LLM-based relevance assessment.

        Proto: TrimRelevanceRequest -> TrimRelevanceResponse
        """
        start = time.monotonic()
        try:
            query_text: str = request.query_text if hasattr(request, "query_text") else ""
            threshold: float = (
                request.relevance_threshold
                if hasattr(request, "relevance_threshold") and request.relevance_threshold
                else self._service._config.relevance_threshold
            )

            ranked_chunks: list[dict[str, Any]] = []
            if hasattr(request, "ranked_chunks"):
                for rc in request.ranked_chunks:
                    chunk_data: dict[str, Any] = {}
                    if hasattr(rc, "chunk") and rc.chunk:
                        inner = rc.chunk
                        if hasattr(inner, "chunk") and inner.chunk:
                            c = inner.chunk
                            chunk_data["chunk_id"] = getattr(c.chunk_id, "value", "") if hasattr(c, "chunk_id") else ""
                            chunk_data["text"] = getattr(c, "text", "")
                    chunk_data["rerank_score"] = getattr(rc, "rerank_score", 0.0)
                    ranked_chunks.append(chunk_data)

            relevant, trimmed = await self._service._trimmer.trim(
                query_text,
                ranked_chunks,
                threshold=threshold,
                batch_size=self._service._config.trimmer_batch_size,
            )

            elapsed = (time.monotonic() - start) * 1000
            logger.info(
                "TrimRelevance completed in %.1fms (%d relevant, %d trimmed)",
                elapsed,
                len(relevant),
                len(trimmed),
            )

            return {
                "success": True,
                "relevant_chunks": relevant,
                "trimmed_chunks": trimmed,
            }

        except Exception as exc:
            logger.exception("TrimRelevance failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return None

    async def ResolveReferences(
        self, request: Any, context: grpc.aio.ServicerContext,
    ) -> Any:
        """Resolve cross-document references for context enrichment.

        Proto: ResolveReferencesRequest -> ResolveReferencesResponse
        """
        start = time.monotonic()
        try:
            max_depth: int = request.max_depth if hasattr(request, "max_depth") and request.max_depth else 1
            max_references: int = (
                request.max_references
                if hasattr(request, "max_references") and request.max_references
                else self._service._config.max_references
            )

            chunks: list[dict[str, Any]] = []
            if hasattr(request, "chunks"):
                for rc in request.chunks:
                    chunk_data: dict[str, Any] = {}
                    if hasattr(rc, "chunk") and rc.chunk:
                        inner = rc.chunk
                        if hasattr(inner, "chunk") and inner.chunk:
                            c = inner.chunk
                            chunk_data["chunk_id"] = getattr(c.chunk_id, "value", "") if hasattr(c, "chunk_id") else ""
                            chunk_data["text"] = getattr(c, "text", "")
                            chunk_data["document_id"] = getattr(c.document_id, "value", "") if hasattr(c, "document_id") else ""
                    chunks.append(chunk_data)

            referenced = await self._service._resolver.resolve(
                chunks,
                max_depth=max_depth,
                max_references=max_references,
            )

            elapsed = (time.monotonic() - start) * 1000
            logger.info(
                "ResolveReferences completed in %.1fms (%d referenced)",
                elapsed,
                len(referenced),
            )

            return {
                "success": True,
                "primary_chunks": chunks,
                "referenced_chunks": referenced,
            }

        except Exception as exc:
            logger.exception("ResolveReferences failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return None
