"""Core embedding service: orchestrates dense + sparse embedding generation."""

from __future__ import annotations

import logging
from typing import Any

from rag_common.config import LLMConfig
from rag_common.llm.client import LLMClient
from rag_common.redis.client import RedisClient
from rag_common.redis.state import StateStore

from agents.embedding.embedding_agent.config import EmbeddingConfig
from agents.embedding.embedding_agent.dense import DenseEmbedder
from agents.embedding.embedding_agent.sparse import BM25Encoder

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Generates dense and sparse embeddings for a list of chunks.

    Reads chunk data from :class:`StateStore`, produces both dense (via
    :class:`DenseEmbedder`) and sparse (via :class:`SparseEmbedder`) vectors,
    and writes the paired results back to the state store.
    """

    def __init__(
        self,
        redis_client: RedisClient,
        llm_client: LLMClient,
        config: EmbeddingConfig,
    ) -> None:
        self._state = StateStore(redis_client)
        self._config = config
        self._dense = DenseEmbedder(llm_client, config)
        self._sparse = BM25Encoder(vocab_size=30_000)
        self._llm_config = LLMConfig()

    async def embed_chunks(
        self,
        task_id: str,
        chunks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Generate dense + sparse embeddings for every chunk.

        Args:
            task_id: The pipeline task identifier.
            chunks: A list of chunk dicts, each containing at least
                ``chunk_id`` and ``text``.

        Returns:
            A result dict with ``status``, ``task_id``, and ``embedding_count``.
        """
        texts = [chunk["text"] for chunk in chunks]

        logger.info("Generating embeddings for task %s: %d chunks", task_id, len(texts))

        # --- Dense embeddings (async) ---
        dense_vectors = await self._dense.embed_texts(texts)

        # --- Sparse embeddings (sync / CPU-bound) ---
        sparse_vectors = self._sparse.encode(texts)

        # --- Pair each chunk with its vectors ---
        embedding_results: list[dict[str, Any]] = []
        for chunk, dense_vec, sparse_vec in zip(chunks, dense_vectors, sparse_vectors):
            embedding_results.append({
                "chunk_id": str(chunk["chunk_id"]),
                "text": chunk.get("text", ""),
                "document_id": str(chunk.get("document_id", "")),
                "sequence_index": chunk.get("sequence_index", 0),
                "metadata": chunk.get("metadata", {}),
                "dense": {
                    "values": dense_vec,
                    "model": self._config.dense_model,
                    "dimensions": len(dense_vec),
                },
                "sparse": {
                    "indices": sparse_vec["indices"],
                    "values": sparse_vec["values"],
                },
            })

        # --- Persist to StateStore ---
        await self._state.store_result(task_id, "embeddings", embedding_results)

        logger.info("Stored %d embeddings for task %s", len(embedding_results), task_id)

        return {
            "status": "success",
            "task_id": task_id,
            "embedding_count": len(embedding_results),
        }
