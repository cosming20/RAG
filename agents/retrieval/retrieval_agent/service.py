"""Retrieval storage service — upserts chunk vectors into Qdrant."""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from qdrant_client.models import PointStruct, SparseVector

from rag_common.redis.client import RedisClient
from rag_common.redis.state import StateStore

from agents.retrieval.retrieval_agent.config import RetrievalConfig
from agents.retrieval.retrieval_agent.pool import QdrantPool

logger = logging.getLogger(__name__)


class RetrievalService:
    """Stores chunk embeddings (dense + sparse) in a Qdrant collection.

    Receives chunk-level data produced by the Embedding agent and persists it as
    Qdrant ``PointStruct`` objects with named vectors for dense and sparse
    retrieval.
    """

    def __init__(
        self,
        redis_client: RedisClient,
        pool: QdrantPool,
        config: RetrievalConfig,
    ) -> None:
        self._state = StateStore(redis_client)
        self._pool = pool
        self._config = config

    async def store_chunks(
        self,
        task_id: str,
        chunks_with_vectors: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Upsert embedded chunks into Qdrant.

        Each element in *chunks_with_vectors* is expected to have:
        - ``chunk_id`` (str | UUID): unique identifier for the point.
        - ``dense_vector`` (list[float]): dense embedding values.
        - ``sparse_indices`` (list[int]): sparse vector indices.
        - ``sparse_values`` (list[float]): sparse vector values.
        - ``text`` (str): original chunk text stored as payload.
        - ``document_id`` (str | UUID): parent document identifier.
        - ``sequence_index`` (int): ordering within the document.
        - ``metadata`` (dict): arbitrary metadata.

        Args:
            task_id: Pipeline task identifier for logging/tracing.
            chunks_with_vectors: Embedded chunk dicts from the Embedding agent.

        Returns:
            A dict with ``success`` (bool) and ``stored_count`` (int).
        """
        await self._pool.ensure_collection()

        points = self._build_points(chunks_with_vectors)
        if not points:
            logger.warning("task_id=%s: no points to upsert", task_id)
            return {"success": True, "stored_count": 0}

        batch_size = self._config.batch_upsert_size
        total_stored = 0
        client = self._pool.client

        for batch_start in range(0, len(points), batch_size):
            batch = points[batch_start : batch_start + batch_size]
            await client.upsert(
                collection_name=self._config.collection_name,
                points=batch,
            )
            total_stored += len(batch)
            logger.debug(
                "task_id=%s: upserted batch %d-%d (%d points)",
                task_id,
                batch_start,
                batch_start + len(batch),
                len(batch),
            )

        logger.info("task_id=%s: stored %d chunks in Qdrant", task_id, total_stored)
        return {"success": True, "stored_count": total_stored}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_points(
        self,
        chunks_with_vectors: list[dict[str, Any]],
    ) -> list[PointStruct]:
        """Convert raw chunk dicts to Qdrant ``PointStruct`` objects."""
        dense_name = self._config.dense_vector_name
        sparse_name = self._config.sparse_vector_name
        points: list[PointStruct] = []

        for chunk in chunks_with_vectors:
            chunk_id = str(chunk.get("chunk_id", ""))
            if not chunk_id:
                logger.warning("Skipping chunk with missing chunk_id")
                continue

            dense_data = chunk.get("dense", {})
            sparse_data = chunk.get("sparse", {})
            dense_vec = dense_data.get("values", []) if isinstance(dense_data, dict) else chunk.get("dense_vector", [])
            sparse_indices = sparse_data.get("indices", []) if isinstance(sparse_data, dict) else chunk.get("sparse_indices", [])
            sparse_values = sparse_data.get("values", []) if isinstance(sparse_data, dict) else chunk.get("sparse_values", [])

            vectors: dict[str, Any] = {}
            if dense_vec:
                vectors[dense_name] = dense_vec
            if sparse_indices and sparse_values:
                vectors[sparse_name] = SparseVector(
                    indices=sparse_indices,
                    values=sparse_values,
                )

            payload = {
                "text": chunk.get("text", ""),
                "document_id": str(chunk.get("document_id", "")),
                "sequence_index": chunk.get("sequence_index", 0),
                "metadata": chunk.get("metadata", {}),
                "task_id": chunk.get("task_id", ""),
            }

            points.append(
                PointStruct(
                    id=chunk_id,
                    vector=vectors,
                    payload=payload,
                ),
            )

        return points
