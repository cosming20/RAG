"""Dense embedding generation via LLMClient."""

from __future__ import annotations

import logging

from rag_common.llm.client import LLMClient

from agents.embedding.embedding_agent.config import EmbeddingConfig

logger = logging.getLogger(__name__)


class DenseEmbedder:
    """Generates dense vector embeddings using the configured LLM provider.

    Delegates to :class:`LLMClient.embed_batch` which already handles batching
    and context-window overflow (split + pool).
    """

    def __init__(self, llm_client: LLMClient, config: EmbeddingConfig) -> None:
        self._llm = llm_client
        self._config = config

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts into dense vectors.

        Uses :meth:`LLMClient.embed_batch` with the configured ``batch_size``
        and ``dense_model``.  Context-window overflow is handled internally by
        the LLM client (split individual texts and pool).

        Args:
            texts: The texts to embed.

        Returns:
            A list of dense embedding vectors, one per input text.
        """
        if not texts:
            return []

        logger.debug("Embedding %d texts with model=%s, batch_size=%d",
                      len(texts), self._config.dense_model, self._config.batch_size)

        return await self._llm.embed_batch(
            texts,
            model=self._config.dense_model,
            batch_size=self._config.batch_size,
        )

    async def embed_single(self, text: str) -> list[float]:
        """Embed a single text into a dense vector.

        Args:
            text: The text to embed.

        Returns:
            A dense embedding vector.
        """
        results = await self.embed_texts([text])
        return results[0]
