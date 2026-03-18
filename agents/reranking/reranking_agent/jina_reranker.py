"""Jina Reranker client -- calls the Jina Reranker API for semantic reranking.

Uses httpx.AsyncClient for async HTTP. Falls back to original ordering on failure.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from rag_common.config import JinaConfig

logger = logging.getLogger(__name__)


class JinaReranker:
    """Calls the Jina Reranker API to reorder chunks by semantic relevance.

    On API failure, falls back to the original ordering so that the pipeline
    never loses data.
    """

    def __init__(self, config: JinaConfig) -> None:
        self._config = config

    async def rerank(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        top_n: int = 10,
    ) -> list[dict[str, Any]]:
        """Rerank *chunks* against *query* via the Jina Reranker API.

        Args:
            query: The user query to rank against.
            chunks: List of chunk dicts; each must contain a ``text`` field.
            top_n: Maximum number of results to return.

        Returns:
            Chunks sorted by descending ``rerank_score``, each enriched with
            ``rerank_score`` (float) and ``original_rank`` (int) fields.
        """
        if not chunks:
            return []

        documents = [chunk.get("text", "") for chunk in chunks]
        api_key = self._config.api_key.get_secret_value()

        request_body = {
            "model": self._config.model,
            "query": query,
            "documents": documents,
            "top_n": min(top_n, len(chunks)),
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=self._config.timeout_seconds) as client:
                response = await client.post(
                    self._config.api_url,
                    json=request_body,
                    headers=headers,
                )
                response.raise_for_status()
                data = response.json()

            return self._map_results(chunks, data)

        except Exception:
            logger.exception(
                "Jina reranker API call failed; falling back to original order",
            )
            return self._fallback_order(chunks, top_n)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _map_results(
        chunks: list[dict[str, Any]],
        api_response: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Map Jina API response back to original chunk dicts."""
        results: list[dict[str, Any]] = []

        for item in api_response.get("results", []):
            idx = item.get("index", 0)
            score = item.get("relevance_score", 0.0)

            if 0 <= idx < len(chunks):
                enriched = {**chunks[idx], "rerank_score": score, "original_rank": idx}
                results.append(enriched)

        # Jina already returns sorted by score; ensure descending order
        results.sort(key=lambda c: c.get("rerank_score", 0.0), reverse=True)
        return results

    @staticmethod
    def _fallback_order(
        chunks: list[dict[str, Any]],
        top_n: int,
    ) -> list[dict[str, Any]]:
        """Return chunks in their original order with synthetic scores."""
        result: list[dict[str, Any]] = []
        for idx, chunk in enumerate(chunks[:top_n]):
            enriched = {
                **chunk,
                "rerank_score": 0.0,
                "original_rank": idx,
            }
            result.append(enriched)
        return result
