"""Graph Agent gRPC servicer -- wraps GraphService and GraphExpander for gRPC transport.

Maps gRPC RPC methods to service layer calls.  When proto generation
is complete, this class should inherit from the generated
``GraphAgentServicer`` base class.

Proto service: ``rag.services.graph.GraphAgent``
RPCs: ExtractEntities, StoreEntities, QueryGraph, ExpandContext
"""

from __future__ import annotations

import logging
import time
from typing import Any

import grpc

from .expansion import GraphExpander
from .extractor import EntityExtractor
from .service import GraphService

logger = logging.getLogger(__name__)


class GraphServicer:
    """gRPC servicer for the Graph Agent.

    Delegates entity extraction and storage to :class:`GraphService`,
    and query-time graph operations to :class:`GraphExpander`.

    Once proto stubs are generated, update this class to inherit from
    ``graph_pb2_grpc.GraphAgentServicer`` and return proper proto
    message objects.
    """

    def __init__(
        self,
        service: GraphService,
        expander: GraphExpander,
    ) -> None:
        self._service = service
        self._expander = expander

    async def ExtractEntities(
        self, request: Any, context: grpc.aio.ServicerContext,
    ) -> Any:
        """Extract entities and relationships from document chunks.

        Proto: ExtractEntitiesRequest -> ExtractEntitiesResponse
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

            extraction = await self._service._extractor.extract(chunks)

            elapsed = (time.monotonic() - start) * 1000
            logger.info(
                "ExtractEntities completed in %.1fms (%d entities, %d relationships)",
                elapsed,
                len(extraction.get("entities", [])),
                len(extraction.get("relationships", [])),
            )

            return {
                "success": True,
                "entities": extraction.get("entities", []),
                "relationships": extraction.get("relationships", []),
            }

        except Exception as exc:
            logger.exception("ExtractEntities failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return None

    async def StoreEntities(
        self, request: Any, context: grpc.aio.ServicerContext,
    ) -> Any:
        """Store extracted entities and relationships in Neo4j.

        Proto: StoreEntitiesRequest -> StoreEntitiesResponse
        """
        start = time.monotonic()
        try:
            document_id: str = (
                request.document_id.value
                if hasattr(request, "document_id") and request.document_id
                else ""
            )

            chunks: list[dict[str, Any]] = []
            if hasattr(request, "chunks"):
                for chunk in request.chunks:
                    chunks.append({
                        "chunk_id": getattr(chunk.chunk_id, "value", "") if hasattr(chunk, "chunk_id") else "",
                        "text": getattr(chunk, "text", ""),
                        "sequence_index": getattr(chunk, "sequence_index", 0),
                    })

            import uuid
            task_id = str(uuid.uuid4())

            result = await self._service.store_entities(
                task_id=task_id,
                document_id=document_id,
                chunks=chunks,
            )

            elapsed = (time.monotonic() - start) * 1000
            logger.info(
                "StoreEntities completed in %.1fms (%d nodes, %d rels)",
                elapsed,
                result.get("nodes_created", 0),
                result.get("relationships_created", 0),
            )

            return result

        except Exception as exc:
            logger.exception("StoreEntities failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return None

    async def QueryGraph(
        self, request: Any, context: grpc.aio.ServicerContext,
    ) -> Any:
        """Query the knowledge graph for entities related to a query.

        Proto: GraphQueryRequest -> GraphQueryResponse
        """
        start = time.monotonic()
        try:
            query_text: str = request.query_text if hasattr(request, "query_text") else ""
            entity_names: list[str] = (
                list(request.entity_names) if hasattr(request, "entity_names") else []
            )
            max_depth: int = request.max_depth if hasattr(request, "max_depth") and request.max_depth else 2
            max_results: int = request.max_results if hasattr(request, "max_results") and request.max_results else 20

            # If no entity names provided, extract from query
            if not entity_names and query_text:
                entities = await self._expander.extract_query_entities(query_text)
                entity_names = [e["name"] for e in entities]

            graph_context = await self._expander.expand_context(
                entity_names,
                max_depth=max_depth,
                max_results=max_results,
            )

            elapsed = (time.monotonic() - start) * 1000
            logger.info(
                "QueryGraph completed in %.1fms (%d entities, %d paths)",
                elapsed,
                len(graph_context.get("entities", [])),
                len(graph_context.get("paths", [])),
            )

            return {
                "success": True,
                "entities": graph_context.get("entities", []),
                "paths": graph_context.get("paths", []),
                "context_text": graph_context.get("context_text", ""),
            }

        except Exception as exc:
            logger.exception("QueryGraph failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return None

    async def ExpandContext(
        self, request: Any, context: grpc.aio.ServicerContext,
    ) -> Any:
        """Expand context from seed entities through the knowledge graph.

        Proto: ExpandContextRequest -> ExpandContextResponse
        """
        start = time.monotonic()
        try:
            entity_names: list[str] = (
                list(request.entity_names) if hasattr(request, "entity_names") else []
            )
            max_depth: int = request.max_depth if hasattr(request, "max_depth") and request.max_depth else 2
            max_expansions: int = request.max_expansions if hasattr(request, "max_expansions") and request.max_expansions else 20

            graph_context = await self._expander.expand_context(
                entity_names,
                max_depth=max_depth,
                max_results=max_expansions,
            )

            elapsed = (time.monotonic() - start) * 1000
            logger.info(
                "ExpandContext completed in %.1fms (%d chunks)",
                elapsed,
                len(graph_context.get("related_chunks", [])),
            )

            return {
                "success": True,
                "expanded_chunks": graph_context.get("related_chunks", []),
                "context_summary": graph_context.get("context_text", ""),
                "paths": graph_context.get("paths", []),
            }

        except Exception as exc:
            logger.exception("ExpandContext failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return None
