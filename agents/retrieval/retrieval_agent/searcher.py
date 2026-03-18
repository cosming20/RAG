"""Qdrant search operations — dense, sparse, and hybrid search with fusion.

Supports:
- Dense-only vector search
- Hybrid search with RRF (Reciprocal Rank Fusion) or Convex fusion
- Multi-collection search with deduplication

Inspired by Lawren's MultiCollectionSearchClient pattern.
"""

from __future__ import annotations

import logging
from typing import Any

from qdrant_client import models

from agents.retrieval.retrieval_agent.config import RetrievalConfig
from agents.retrieval.retrieval_agent.pool import QdrantPool

logger = logging.getLogger(__name__)

# Prefetch multiplier: fetch more candidates per sub-query than the final limit
# to give the fusion algorithm enough candidates to rank.
_PREFETCH_MULTIPLIER = 3


class QdrantSearcher:
    """Search interface over Qdrant supporting dense and hybrid retrieval.

    Uses Qdrant's ``query_points`` API with ``Prefetch`` and ``FusionQuery``
    for hybrid search, combining dense and sparse vectors via RRF or Convex
    fusion.
    """

    def __init__(self, pool: QdrantPool, config: RetrievalConfig) -> None:
        self._pool = pool
        self._config = config

    async def search(
        self,
        query_vector: list[float],
        *,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Dense-only vector search against Qdrant.

        Args:
            query_vector: Dense embedding vector for the query.
            limit: Maximum number of results to return.
            filters: Optional Qdrant filter conditions (converted to models.Filter).

        Returns:
            List of result dicts with chunk_id, text, document_id,
            relevance_score, and metadata.
        """
        client = self._pool.client
        query_filter = _build_filter(filters) if filters else None

        response = await client.query_points(
            collection_name=self._config.collection_name,
            query=query_vector,
            using=self._config.dense_vector_name,
            limit=limit,
            with_payload=True,
            query_filter=query_filter,
        )

        return [_point_to_result(point) for point in response.points]

    async def hybrid_search(
        self,
        dense_vector: list[float],
        sparse_vector: dict[str, Any],
        *,
        limit: int = 10,
        fusion: str = "rrf",
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Hybrid dense + sparse search with fusion.

        Uses Qdrant's prefetch + fusion API to combine results from both
        dense and sparse vector searches.

        Args:
            dense_vector: Dense embedding for the query.
            sparse_vector: Sparse vector dict with ``indices`` (list[int]) and
                ``values`` (list[float]).
            limit: Maximum number of final results.
            fusion: Fusion method - ``"rrf"`` (Reciprocal Rank Fusion) or
                ``"dbsf"`` (Distribution-Based Score Fusion / Convex).
            filters: Optional Qdrant filter conditions.

        Returns:
            List of result dicts with chunk_id, text, document_id,
            relevance_score, and metadata.
        """
        client = self._pool.client
        query_filter = _build_filter(filters) if filters else None
        prefetch_limit = limit * _PREFETCH_MULTIPLIER

        sparse_query = models.SparseVector(
            indices=sparse_vector.get("indices", []),
            values=sparse_vector.get("values", []),
        )

        prefetch = [
            models.Prefetch(
                query=dense_vector,
                using=self._config.dense_vector_name,
                limit=prefetch_limit,
                filter=query_filter,
            ),
            models.Prefetch(
                query=sparse_query,
                using=self._config.sparse_vector_name,
                limit=prefetch_limit,
                filter=query_filter,
            ),
        ]

        fusion_query = _resolve_fusion(fusion)

        response = await client.query_points(
            collection_name=self._config.collection_name,
            prefetch=prefetch,
            query=fusion_query,
            limit=limit,
            with_payload=True,
        )

        return [_point_to_result(point) for point in response.points]

    async def multi_collection_search(
        self,
        collections: list[str],
        dense_vector: list[float],
        sparse_vector: dict[str, Any],
        *,
        limit: int = 10,
        fusion: str = "rrf",
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search across multiple collections, merge and deduplicate results.

        Inspired by Lawren's MultiCollectionSearchClient pattern: runs hybrid
        search on each collection in parallel, merges results by relevance
        score, and deduplicates by chunk_id.

        Args:
            collections: List of Qdrant collection names to search.
            dense_vector: Dense embedding for the query.
            sparse_vector: Sparse vector dict with ``indices`` and ``values``.
            limit: Maximum total results after merge.
            fusion: Fusion method (``"rrf"`` or ``"dbsf"``).
            filters: Optional Qdrant filter conditions.

        Returns:
            Merged and deduplicated list of result dicts, sorted by
            descending relevance_score.
        """
        client = self._pool.client
        query_filter = _build_filter(filters) if filters else None
        prefetch_limit = limit * _PREFETCH_MULTIPLIER
        fusion_query = _resolve_fusion(fusion)

        sparse_query = models.SparseVector(
            indices=sparse_vector.get("indices", []),
            values=sparse_vector.get("values", []),
        )

        all_results: list[dict[str, Any]] = []
        for collection in collections:
            try:
                prefetch = [
                    models.Prefetch(
                        query=dense_vector,
                        using=self._config.dense_vector_name,
                        limit=prefetch_limit,
                        filter=query_filter,
                    ),
                    models.Prefetch(
                        query=sparse_query,
                        using=self._config.sparse_vector_name,
                        limit=prefetch_limit,
                        filter=query_filter,
                    ),
                ]

                response = await client.query_points(
                    collection_name=collection,
                    prefetch=prefetch,
                    query=fusion_query,
                    limit=limit,
                    with_payload=True,
                )

                for point in response.points:
                    result = _point_to_result(point)
                    result["collection"] = collection
                    all_results.append(result)
            except Exception:
                logger.exception(
                    "Failed to search collection %r, skipping", collection,
                )

        # Deduplicate by chunk_id, keeping the highest-scoring entry
        seen: dict[str, dict[str, Any]] = {}
        for result in all_results:
            cid = result["chunk_id"]
            if cid not in seen or result["relevance_score"] > seen[cid]["relevance_score"]:
                seen[cid] = result

        merged = sorted(seen.values(), key=lambda r: r["relevance_score"], reverse=True)
        return merged[:limit]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _point_to_result(point: Any) -> dict[str, Any]:
    """Convert a Qdrant ScoredPoint to a standardized result dict."""
    payload = point.payload or {}
    return {
        "chunk_id": str(point.id),
        "text": payload.get("text", ""),
        "document_id": payload.get("document_id", ""),
        "relevance_score": point.score,
        "metadata": payload.get("metadata", {}),
    }


def _resolve_fusion(fusion: str) -> models.FusionQuery:
    """Map a fusion name string to the Qdrant FusionQuery model."""
    fusion_lower = fusion.lower()
    if fusion_lower == "rrf":
        return models.FusionQuery(fusion=models.Fusion.RRF)
    if fusion_lower in ("dbsf", "convex"):
        return models.FusionQuery(fusion=models.Fusion.DBSF)
    logger.warning("Unknown fusion method %r, falling back to RRF", fusion)
    return models.FusionQuery(fusion=models.Fusion.RRF)


def _build_filter(filters: dict[str, Any]) -> models.Filter:
    """Build a simple Qdrant Filter from a flat key-value dict.

    Each key-value pair becomes a ``FieldCondition`` with ``MatchValue``.
    All conditions are combined with ``must`` (AND logic).
    """
    conditions = [
        models.FieldCondition(
            key=key,
            match=models.MatchValue(value=value),
        )
        for key, value in filters.items()
    ]
    return models.Filter(must=conditions)
