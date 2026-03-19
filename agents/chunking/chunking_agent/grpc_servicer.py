"""Chunking Agent gRPC servicer -- wraps ChunkingService for gRPC transport.

Maps gRPC RPC methods to :class:`ChunkingService` calls.  When proto
generation is complete, this class should inherit from the generated
``ChunkingAgentServicer`` base class.

Proto service: ``rag.services.chunking.ChunkingAgent``
RPCs: ChunkDocument, NormalizeChunks, SummarizeDocument
"""

from __future__ import annotations

import logging
import time
from typing import Any

import grpc

from .service import ChunkingService

logger = logging.getLogger(__name__)


class ChunkingServicer:
    """gRPC servicer for the Chunking Agent.

    Delegates chunking, normalization, and summarization to
    :class:`ChunkingService`.

    Once proto stubs are generated, update this class to inherit from
    ``chunking_pb2_grpc.ChunkingAgentServicer`` and return proper
    proto message objects.
    """

    def __init__(self, service: ChunkingService) -> None:
        self._service = service

    async def ChunkDocument(
        self, request: Any, context: grpc.aio.ServicerContext,
    ) -> Any:
        """Chunk a parsed document into smaller pieces.

        Proto: ChunkDocumentRequest -> ChunkDocumentResponse
        """
        start = time.monotonic()
        try:
            document_id: str = (
                request.document_id.value
                if hasattr(request, "document_id") and request.document_id
                else ""
            )
            parsed_content: str = (
                request.parsed_content if hasattr(request, "parsed_content") else ""
            )
            fmt: str = str(request.format) if hasattr(request, "format") else "txt"
            trace_id: str = (
                request.trace_context.trace_id
                if hasattr(request, "trace_context") and request.trace_context
                else ""
            )

            # Extract metadata from chunking config if provided
            metadata: dict[str, Any] = {}
            if hasattr(request, "config") and request.config:
                config = request.config
                metadata = {
                    "max_chunk_size": getattr(config, "max_chunk_size", 0),
                    "min_chunk_size": getattr(config, "min_chunk_size", 0),
                    "overlap_ratio": getattr(config, "overlap_ratio", 0.0),
                }

            # Generate a task_id for the operation
            import uuid
            task_id = str(uuid.uuid4())

            result = await self._service.chunk_document(
                task_id=task_id,
                document_id=document_id,
                parsed_content=parsed_content,
                fmt=fmt,
                metadata=metadata,
            )

            elapsed = (time.monotonic() - start) * 1000
            logger.info(
                "ChunkDocument completed in %.1fms (task_id=%s, chunks=%d)",
                elapsed,
                task_id,
                result.get("chunk_count", 0),
            )
            return result

        except Exception as exc:
            logger.exception("ChunkDocument failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return None

    async def NormalizeChunks(
        self, request: Any, context: grpc.aio.ServicerContext,
    ) -> Any:
        """Normalize chunks via LLM for consistency.

        Proto: NormalizeChunksRequest -> NormalizeChunksResponse
        """
        start = time.monotonic()
        try:
            chunks: list[dict[str, Any]] = []
            if hasattr(request, "chunks"):
                for chunk in request.chunks:
                    chunks.append({
                        "chunk_id": getattr(chunk.chunk_id, "value", "") if hasattr(chunk, "chunk_id") else "",
                        "text": getattr(chunk, "text", ""),
                        "sequence_index": getattr(chunk, "sequence_index", 0),
                    })

            normalized = await self._service._normalizer.normalize(chunks)

            elapsed = (time.monotonic() - start) * 1000
            logger.info(
                "NormalizeChunks completed in %.1fms (%d chunks)",
                elapsed,
                len(normalized),
            )
            return {"success": True, "normalized_chunks": normalized}

        except Exception as exc:
            logger.exception("NormalizeChunks failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return None

    async def SummarizeDocument(
        self, request: Any, context: grpc.aio.ServicerContext,
    ) -> Any:
        """Generate a document summary from parsed content and chunks.

        Proto: SummarizeDocumentRequest -> SummarizeDocumentResponse
        """
        start = time.monotonic()
        try:
            parsed_content: str = (
                request.parsed_content if hasattr(request, "parsed_content") else ""
            )
            chunks: list[dict[str, Any]] = []
            if hasattr(request, "chunks"):
                for chunk in request.chunks:
                    chunks.append({
                        "text": getattr(chunk, "text", ""),
                        "sequence_index": getattr(chunk, "sequence_index", 0),
                    })

            summary_data = await self._service._summarizer.summarize(
                parsed_content, chunks,
            )

            elapsed = (time.monotonic() - start) * 1000
            logger.info("SummarizeDocument completed in %.1fms", elapsed)

            return {
                "success": True,
                "summary": summary_data.get("summary", ""),
                "key_topics": summary_data.get("key_topics", []),
            }

        except Exception as exc:
            logger.exception("SummarizeDocument failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return None
