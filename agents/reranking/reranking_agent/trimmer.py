"""LLM-based relevance trimmer -- filters chunks by semantic relevance.

Inspired by Lawren's trimmer.rs: uses a lightweight LLM to score each chunk
against the query and removes irrelevant results. Falls back to keeping all
chunks if the LLM call fails (never loses data).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from rag_common.llm.client import LLMClient

logger = logging.getLogger(__name__)

_TRIMMER_SYSTEM_PROMPT = (
    "You are a relevance evaluator. Given a query and a batch of numbered texts, "
    "rate each text's relevance to the query on a scale of 0.0 to 1.0. "
    "Return ONLY valid JSON: a list of objects with keys \"index\", \"relevant\" (bool), "
    "and \"score\" (float). No explanation."
)

_TRIMMER_USER_TEMPLATE = (
    "Query: {query}\n\n"
    "Texts:\n{texts}\n\n"
    "Rate each text's relevance (0.0 to 1.0). A text is relevant if it directly "
    "helps answer the query. Return JSON: "
    '[{{"index": 0, "relevant": true, "score": 0.8}}, ...]'
)


class RelevanceTrimmer:
    """Uses a lightweight LLM to evaluate and filter chunk relevance.

    Batches chunks to reduce LLM calls. If the LLM fails, all chunks are
    preserved to avoid data loss.
    """

    def __init__(self, llm_client: LLMClient, model: str) -> None:
        self._llm = llm_client
        self._model = model

    async def trim(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        threshold: float = 0.3,
        batch_size: int = 5,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Evaluate relevance of each chunk and split into kept/trimmed.

        Args:
            query: The user query.
            chunks: Reranked chunks to evaluate.
            threshold: Minimum relevance score to keep a chunk.
            batch_size: Number of chunks per LLM call.

        Returns:
            A tuple of (relevant_chunks, trimmed_chunks).
        """
        if not chunks:
            return [], []

        relevance_scores: list[float] = [0.0] * len(chunks)

        for batch_start in range(0, len(chunks), batch_size):
            batch = chunks[batch_start : batch_start + batch_size]
            scores = await self._evaluate_batch(query, batch, batch_start)
            for offset, score in scores.items():
                global_idx = batch_start + offset
                if global_idx < len(relevance_scores):
                    relevance_scores[global_idx] = score

        relevant: list[dict[str, Any]] = []
        trimmed: list[dict[str, Any]] = []

        for idx, chunk in enumerate(chunks):
            enriched = {**chunk, "trimmer_score": relevance_scores[idx]}
            if relevance_scores[idx] >= threshold:
                relevant.append(enriched)
            else:
                trimmed.append(enriched)

        logger.info(
            "Trimmer: kept %d chunks, trimmed %d (threshold=%.2f)",
            len(relevant),
            len(trimmed),
            threshold,
        )
        return relevant, trimmed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _evaluate_batch(
        self,
        query: str,
        batch: list[dict[str, Any]],
        start_index: int,
    ) -> dict[int, float]:
        """Call LLM for a single batch and return {local_index: score}."""
        texts_block = "\n".join(
            f"[{i}] {chunk.get('text', '')[:500]}"
            for i, chunk in enumerate(batch)
        )
        user_message = _TRIMMER_USER_TEMPLATE.format(query=query, texts=texts_block)

        try:
            response = await self._llm.completion(
                messages=[
                    {"role": "system", "content": _TRIMMER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                model=self._model,
                temperature=0.0,
                max_tokens=512,
            )

            content = response.choices[0].message.content.strip()
            # Strip markdown code fences if present
            if content.startswith("```"):
                content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

            parsed: list[dict[str, Any]] = json.loads(content)
            return {
                item.get("index", i): float(item.get("score", 0.0))
                for i, item in enumerate(parsed)
                if isinstance(item, dict)
            }

        except Exception:
            logger.exception(
                "Trimmer LLM call failed for batch starting at %d; keeping all chunks",
                start_index,
            )
            # Fallback: assign score 1.0 so no data is lost
            return {i: 1.0 for i in range(len(batch))}
