"""Qdrant connection pool and collection management."""

from __future__ import annotations

import logging

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    SparseVectorParams,
    VectorParams,
)

from rag_common.config import QdrantConfig

from agents.retrieval.retrieval_agent.config import RetrievalConfig

logger = logging.getLogger(__name__)


class QdrantPool:
    """Async Qdrant client wrapper with connection lifecycle and collection management.

    Inspired by the Lawren QdrantPool pattern: thin wrapper that owns the
    connection, guarantees collection existence, and exposes the raw client
    for advanced callers.
    """

    def __init__(
        self,
        qdrant_config: QdrantConfig,
        retrieval_config: RetrievalConfig,
    ) -> None:
        self._qdrant_config = qdrant_config
        self._retrieval_config = retrieval_config
        self._client: AsyncQdrantClient | None = None

    @property
    def client(self) -> AsyncQdrantClient:
        """Return the connected async Qdrant client.

        Raises:
            RuntimeError: If ``connect()`` has not been called.
        """
        if self._client is None:
            msg = "QdrantPool not connected. Call connect() first."
            raise RuntimeError(msg)
        return self._client

    async def connect(self) -> None:
        """Create and verify the async Qdrant client connection."""
        api_key = (
            self._qdrant_config.api_key.get_secret_value()
            if self._qdrant_config.api_key
            else None
        )
        self._client = AsyncQdrantClient(
            host=self._qdrant_config.host,
            grpc_port=self._qdrant_config.grpc_port,
            port=self._qdrant_config.rest_port,
            api_key=api_key,
            prefer_grpc=self._qdrant_config.prefer_grpc,
            timeout=self._qdrant_config.timeout,
        )
        logger.info(
            "Connected to Qdrant at %s (grpc=%d, rest=%d)",
            self._qdrant_config.host,
            self._qdrant_config.grpc_port,
            self._qdrant_config.rest_port,
        )

    async def close(self) -> None:
        """Close the Qdrant client connection."""
        if self._client is not None:
            await self._client.close()
            self._client = None
            logger.info("Qdrant connection closed")

    async def ensure_collection(self) -> None:
        """Create the target collection if it does not already exist.

        Configures named vectors for both dense and sparse search:
        - ``dense``: standard vector with cosine distance.
        - ``sparse``: sparse vector for BM42/TF-IDF retrieval.
        """
        client = self.client
        cfg = self._retrieval_config

        collections = await client.get_collections()
        existing_names = {c.name for c in collections.collections}

        if cfg.collection_name in existing_names:
            logger.debug("Collection %r already exists", cfg.collection_name)
            return

        await client.create_collection(
            collection_name=cfg.collection_name,
            vectors_config={
                cfg.dense_vector_name: VectorParams(
                    size=cfg.vector_size,
                    distance=Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                cfg.sparse_vector_name: SparseVectorParams(),
            },
        )
        logger.info(
            "Created collection %r (dense=%d dims, sparse=%s)",
            cfg.collection_name,
            cfg.vector_size,
            cfg.sparse_vector_name,
        )
