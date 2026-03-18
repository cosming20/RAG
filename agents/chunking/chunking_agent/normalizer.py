"""LLM-based chunk normalization -- cleans OCR artifacts and formatting noise."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from rag_common.llm.client import LLMClient

from .config import ChunkingConfig

logger = logging.getLogger(__name__)

NORMALIZE_SYSTEM_PROMPT = (
    "Clean up the following text chunk. Fix OCR artifacts, broken words, "
    "and formatting issues. Remove noise and extraneous whitespace. "
    "Return only the cleaned text with no commentary."
)


class ChunkNormalizer:
    """Uses an LLM to clean up chunk text (OCR artifacts, formatting)."""

    def __init__(self, llm_client: LLMClient, config: ChunkingConfig) -> None:
        self._llm = llm_client
        self._batch_size = config.normalize_batch_size

    async def normalize(
        self, chunks: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Normalize all chunks, processing in batches to limit concurrency.

        Returns a new list of chunks with cleaned text.  Original chunk
        dicts are not mutated.
        """
        if not chunks:
            return []

        result: list[dict[str, Any]] = []
        for batch_start in range(0, len(chunks), self._batch_size):
            batch = chunks[batch_start : batch_start + self._batch_size]
            tasks = [self._normalize_single(chunk) for chunk in batch]
            normalized_batch = await asyncio.gather(*tasks, return_exceptions=True)

            for original, normalized in zip(batch, normalized_batch):
                if isinstance(normalized, Exception):
                    logger.warning(
                        "Normalization failed for chunk %d, keeping original: %s",
                        original.get("sequence_index", -1),
                        normalized,
                    )
                    result.append(original)
                else:
                    result.append(normalized)

        return result

    async def _normalize_single(
        self, chunk: dict[str, Any]
    ) -> dict[str, Any]:
        """Normalize a single chunk via LLM completion."""
        text = chunk.get("text", "")
        if not text.strip():
            return chunk

        messages = [
            {"role": "system", "content": NORMALIZE_SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]
        response = await self._llm.completion(
            messages, temperature=0.0, max_tokens=len(text) + 256
        )
        cleaned = response.choices[0].message.content.strip()

        return {
            **chunk,
            "text": cleaned,
            "metadata": {
                **chunk.get("metadata", {}),
                "normalized": "true",
            },
        }
