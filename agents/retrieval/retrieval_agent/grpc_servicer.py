"""Retrieval Agent gRPC servicer -- wraps RetrievalService and QdrantSearcher for gRPC transport.

Maps gRPC RPC methods to service layer calls.  When proto generation
is complete, this class should inherit from the generated
``RetrievalAgentServicer`` base class.

Proto service: ``rag.services.retrieval.RetrievalAgent``
RPCs: Search, HybridSearch, BatchSearch, StoreChunks
"""

from __future__ import annotations

import logging
import time
from typing import Any

import grpc

from .searcher import QdrantSearcher
from .service import RetrievalService

logger = logging.getLogger(__name__)


class RetrievalServicer:
    """gRPC servicer for the Retrieval Agent.

    Delegates search operations to :class:`QdrantSearcher` and storage
    operations to :class:`RetrievalService`.

    Once proto stubs are generated, update this class to inherit from
    ``retrieval_pb2_grpc.RetrievalAgentServicer`` and return proper
    proto message objects.
    """

    def __init__(
        self,
        service: RetrievalService,
        searcher: QdrantSearcher,
    ) -> None:
        self._service = service
        self._searcher = searcher

    async def Search(
        self, request: Any, context: grpc.aio.ServicerContext,
    ) -> Any:
        """Dense-only vector search.

        Proto: SearchRequest -> SearchResponse
        """
        start = time.monotonic()
        try:
            query_vector: list[float] = []
            if hasattr(request, "query_vector") and request.query_vector:
                query_vector = list(request.query_vector.values)

            limit: int = request.limit if hasattr(request, "limit") and request.limit else 10
            filters: dict[str, str] = (
                dict(request.filters) if hasattr(request, "filters") else {}
            )

            if not query_vector:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("query_vector is required for Search")
                return None

            results = await self._searcher.search(
                query_vector=query_vector,
                limit=limit,
                filters=filters if filters else None,
            )

            elapsed = (time.monotonic() - start) * 1000
            logger.info("Search completed in %.1fms (%d results)", elapsed, len(results))

            return {"success": True, "chunks": results}

        except Exception as exc:
            logger.exception("Search failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return None

    async def HybridSearch(
        self, request: Any, context: grpc.aio.ServicerContext,
    ) -> Any:
        """Hybrid dense + sparse search with fusion.

        Proto: HybridSearchRequest -> SearchResponse
        """
        start = time.monotonic()
        try:
            dense_vector: list[float] = []
            if hasattr(request, "dense_vector") and request.dense_vector:
                dense_vector = list(request.dense_vector.values)

            sparse_vector: dict[str, Any] = {}
            if hasattr(request, "sparse_vector") and request.sparse_vector:
                sparse_vector = {
                    "indices": list(request.sparse_vector.indices),
                    "values": list(request.sparse_vector.values),
                }

            limit: int = request.limit if hasattr(request, "limit") and request.limit else 10
            fusion: str = (
                "rrf" if not hasattr(request, "fusion_method") or request.fusion_method == 0
                else "rrf" if request.fusion_method == 1
                else "dbsf"
            )
            filters: dict[str, str] = (
                dict(request.filters) if hasattr(request, "filters") else {}
            )

            if not dense_vector:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("dense_vector is required for HybridSearch")
                return None

            results = await self._searcher.hybrid_search(
                dense_vector=dense_vector,
                sparse_vector=sparse_vector,
                limit=limit,
                fusion=fusion,
                filters=filters if filters else None,
            )

            elapsed = (time.monotonic() - start) * 1000
            logger.info("HybridSearch completed in %.1fms (%d results)", elapsed, len(results))

            return {"success": True, "chunks": results}

        except Exception as exc:
            logger.exception("HybridSearch failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return None

    async def StoreChunks(
        self, request: Any, context: grpc.aio.ServicerContext,
    ) -> Any:
        """Store chunk vectors in Qdrant.

        Proto: StoreChunksRequest -> StoreChunksResponse
        """
        start = time.monotonic()
        try:
            chunks_with_vectors: list[dict[str, Any]] = []
            if hasattr(request, "chunks"):
                for cwv in request.chunks:
                    chunk_data: dict[str, Any] = {}
                    if hasattr(cwv, "chunk") and cwv.chunk:
                        c = cwv.chunk
                        chunk_data["chunk_id"] = getattr(c.chunk_id, "value", "") if hasattr(c, "chunk_id") else ""
                        chunk_data["text"] = getattr(c, "text", "")
                        chunk_data["document_id"] = getattr(c.document_id, "value", "") if hasattr(c, "document_id") else ""
                        chunk_data["sequence_index"] = getattr(c, "sequence_index", 0)
                        chunk_data["metadata"] = dict(c.metadata) if hasattr(c, "metadata") else {}
                    if hasattr(cwv, "vectors") and cwv.vectors:
                        v = cwv.vectors
                        if hasattr(v, "dense") and v.dense:
                            chunk_data["dense"] = {
                                "values": list(v.dense.values),
                                "model": getattr(v.dense, "model", ""),
                                "dimensions": getattr(v.dense, "dimensions", 0),
                            }
                        if hasattr(v, "sparse") and v.sparse:
                            chunk_data["sparse"] = {
                                "indices": list(v.sparse.indices),
                                "values": list(v.sparse.values),
                            }
                    chunks_with_vectors.append(chunk_data)

            import uuid
            task_id = str(uuid.uuid4())

            result = await self._service.store_chunks(
                task_id=task_id,
                chunks_with_vectors=chunks_with_vectors,
            )

            elapsed = (time.monotonic() - start) * 1000
            logger.info(
                "StoreChunks completed in %.1fms (%d stored)",
                elapsed,
                result.get("stored_count", 0),
            )

            return result

        except Exception as exc:
            logger.exception("StoreChunks failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return None
