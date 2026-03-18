"""LLM-based document summarization."""

from __future__ import annotations

import json
import logging
from typing import Any

from rag_common.llm.client import LLMClient

logger = logging.getLogger(__name__)

SUMMARIZE_SYSTEM_PROMPT = (
    "Summarize this document. Provide your response as JSON with two keys:\n"
    '1) "summary": A concise summary (2-3 paragraphs)\n'
    '2) "key_topics": A list of key topics as strings\n\n'
    "Return ONLY valid JSON, no markdown fences or commentary."
)


class DocumentSummarizer:
    """Generates a document-level summary and key topics via LLM."""

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client

    async def summarize(
        self,
        parsed_content: str,
        chunks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Produce a summary and key topics for the full document.

        Uses the parsed_content for context.  If the content is very long,
        uses the first and last chunks as a representative sample.

        Returns:
            A dict with ``summary`` (str) and ``key_topics`` (list[str]).
        """
        # Build input text -- use full content if short, else sample
        max_input_chars = 12_000
        if len(parsed_content) <= max_input_chars:
            input_text = parsed_content
        else:
            input_text = self._build_sample(parsed_content, chunks, max_input_chars)

        messages = [
            {"role": "system", "content": SUMMARIZE_SYSTEM_PROMPT},
            {"role": "user", "content": input_text},
        ]

        try:
            response = await self._llm.completion(
                messages, temperature=0.0, max_tokens=2048
            )
            raw_output = response.choices[0].message.content.strip()
            return self._parse_response(raw_output)
        except Exception:
            logger.warning("Summarization LLM call failed", exc_info=True)
            return {"summary": "", "key_topics": []}

    @staticmethod
    def _build_sample(
        parsed_content: str,
        chunks: list[dict[str, Any]],
        max_chars: int,
    ) -> str:
        """Build a representative sample from the beginning, middle, and end."""
        if not chunks:
            return parsed_content[:max_chars]

        third = max_chars // 3
        sample_parts: list[str] = []

        # First chunks
        accumulated = 0
        for c in chunks:
            text = c.get("text", "")
            if accumulated + len(text) > third:
                sample_parts.append(text[: third - accumulated])
                break
            sample_parts.append(text)
            accumulated += len(text)

        sample_parts.append("\n\n[...]\n\n")

        # Middle chunk
        mid_idx = len(chunks) // 2
        mid_text = chunks[mid_idx].get("text", "")
        sample_parts.append(mid_text[:third])

        sample_parts.append("\n\n[...]\n\n")

        # Last chunks
        accumulated = 0
        for c in reversed(chunks):
            text = c.get("text", "")
            if accumulated + len(text) > third:
                sample_parts.append(text[-(third - accumulated) :])
                break
            sample_parts.append(text)
            accumulated += len(text)

        return "".join(sample_parts)

    @staticmethod
    def _parse_response(raw: str) -> dict[str, Any]:
        """Parse the LLM JSON response, with fallback."""
        # Strip potential markdown fences
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:])
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)
            return {
                "summary": data.get("summary", ""),
                "key_topics": data.get("key_topics", []),
            }
        except json.JSONDecodeError:
            logger.warning("Could not parse summarizer JSON, using raw text")
            return {"summary": cleaned, "key_topics": []}
