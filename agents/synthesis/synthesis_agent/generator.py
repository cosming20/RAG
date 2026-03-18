"""Answer generation with citation extraction.

Uses LLMClient to produce answers from context, then parses [N] citation
references back to source chunk metadata.
"""

from __future__ import annotations

import logging
import re
from collections.abc import AsyncIterator
from typing import Any

from rag_common.llm.client import LLMClient

from agents.synthesis.synthesis_agent.config import SynthesisConfig
from agents.synthesis.synthesis_agent.prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)

# Regex to find citation markers like [1], [2], [13]
CITATION_PATTERN = re.compile(r"\[(\d+)\]")


class AnswerGenerator:
    """Generates answers via LLM and extracts structured citations."""

    def __init__(self, llm_client: LLMClient, config: SynthesisConfig) -> None:
        self._llm = llm_client
        self._config = config
        self._prompt_builder = PromptBuilder(config)

    async def generate(
        self,
        query: str,
        context_chunks: list[dict],
        graph_context: str | None,
        summary: str | None,
        intent: str,
        system_prompt_override: str | None = None,
    ) -> dict[str, Any]:
        """Generate an answer with citations.

        Args:
            query: The user's question.
            context_chunks: Ranked chunks with metadata.
            graph_context: Optional graph-derived context.
            summary: Optional document summary.
            intent: Detected query intent.
            system_prompt_override: Optional system prompt override
                (used for retry with feedback).

        Returns:
            Dict with keys ``text``, ``citations``, ``model``.
        """
        context = self._prompt_builder.build_context(
            context_chunks, graph_context, summary,
        )
        messages = self._prompt_builder.build_messages(
            query, context, intent, system_prompt=system_prompt_override,
        )

        response = await self._llm.completion(
            messages,
            model=self._config.synthesis_model,
        )
        answer_text = response.choices[0].message.content or ""

        citations = self._extract_citations(answer_text, context_chunks)

        return {
            "text": answer_text,
            "citations": citations,
            "model": self._config.synthesis_model,
        }

    async def generate_stream(
        self,
        query: str,
        context_chunks: list[dict],
        graph_context: str | None,
        summary: str | None,
        intent: str,
    ) -> AsyncIterator[str]:
        """Stream answer tokens one at a time.

        Yields individual token strings as they arrive from the LLM.
        Intended for the QueryStream gRPC endpoint.

        Args:
            query: The user's question.
            context_chunks: Ranked chunks with metadata.
            graph_context: Optional graph-derived context.
            summary: Optional document summary.
            intent: Detected query intent.

        Yields:
            Individual token strings.
        """
        context = self._prompt_builder.build_context(
            context_chunks, graph_context, summary,
        )
        messages = self._prompt_builder.build_messages(query, context, intent)

        response = await self._llm.completion(
            messages,
            model=self._config.synthesis_model,
            stream=True,
        )

        async for chunk in response:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content

    def _extract_citations(
        self,
        answer_text: str,
        context_chunks: list[dict],
    ) -> list[dict[str, Any]]:
        """Extract [N] citation references and map to source chunks.

        Args:
            answer_text: The generated answer text.
            context_chunks: The chunks that were provided as context
                (1-indexed to match [N] references).

        Returns:
            List of citation dicts with ``document_id``, ``filename``,
            ``page_number``, ``section``, ``quoted_text``.
        """
        limited_chunks = context_chunks[: self._config.max_context_chunks]
        found_indices = set()
        for match in CITATION_PATTERN.finditer(answer_text):
            idx = int(match.group(1))
            found_indices.add(idx)

        citations: list[dict[str, Any]] = []
        seen: set[int] = set()
        for idx in sorted(found_indices):
            if idx in seen:
                continue
            seen.add(idx)

            # [N] references are 1-indexed
            chunk_pos = idx - 1
            if 0 <= chunk_pos < len(limited_chunks):
                chunk = limited_chunks[chunk_pos]
                text_snippet = chunk.get("text", "")
                # Take first 200 chars as a representative quote
                if len(text_snippet) > 200:
                    text_snippet = text_snippet[:200] + "..."

                citations.append({
                    "document_id": chunk.get("document_id", ""),
                    "filename": chunk.get("filename", ""),
                    "page_number": chunk.get("page_number", 0),
                    "section": chunk.get("section", ""),
                    "quoted_text": text_snippet,
                })
            else:
                logger.warning(
                    "Citation [%d] references non-existent chunk (have %d)",
                    idx,
                    len(limited_chunks),
                )

        return citations
