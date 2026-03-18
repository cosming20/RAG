"""Prompt construction for the Synthesis agent.

Formats retrieved context chunks, graph context, and document summaries
into structured LLM messages with numbered source references.
"""

from __future__ import annotations

import logging

from agents.synthesis.synthesis_agent.config import DEFAULT_SYSTEM_PROMPT, SynthesisConfig

logger = logging.getLogger(__name__)

# Approximate chars-per-token ratio for truncation heuristic
CHARS_PER_TOKEN = 4


class PromptBuilder:
    """Builds context strings and message lists for LLM answer generation."""

    def __init__(self, config: SynthesisConfig) -> None:
        self._config = config

    def build_context(
        self,
        chunks: list[dict],
        graph_context: str | None,
        summary: str | None,
    ) -> str:
        """Format chunks as numbered references and append supplementary context.

        Each chunk is rendered as::

            [1] (source: filename.pdf, page 3)
            <chunk text>

        Graph context and document summaries are appended after the chunks.
        The total output is truncated to ``max_context_tokens`` (estimated via
        a chars * 0.25 heuristic).

        Args:
            chunks: List of chunk dicts with keys ``text``, ``filename``,
                ``page_number``, ``section``, ``document_id``, ``chunk_id``.
            graph_context: Optional knowledge-graph context string.
            summary: Optional document-level summary.

        Returns:
            Formatted context string ready for insertion into the user message.
        """
        max_chars = self._config.max_context_tokens * CHARS_PER_TOKEN
        parts: list[str] = []

        # Numbered chunk references
        limited_chunks = chunks[: self._config.max_context_chunks]
        for idx, chunk in enumerate(limited_chunks, start=1):
            source = chunk.get("filename", "unknown")
            page = chunk.get("page_number", 0)
            section = chunk.get("section", "")

            source_label = f"source: {source}"
            if page:
                source_label += f", page {page}"
            if section:
                source_label += f", {section}"

            text = chunk.get("text", "")
            parts.append(f"[{idx}] ({source_label})\n{text}")

        # Supplementary sections
        if graph_context:
            parts.append(f"--- Knowledge Graph Context ---\n{graph_context}")

        if summary:
            parts.append(f"--- Document Summary ---\n{summary}")

        context = "\n\n".join(parts)

        # Truncate to approximate token budget
        if len(context) > max_chars:
            logger.warning(
                "Context length %d chars exceeds budget %d chars, truncating",
                len(context),
                max_chars,
            )
            context = context[:max_chars]

        return context

    def build_messages(
        self,
        query: str,
        context: str,
        intent: str,
        system_prompt: str | None = None,
    ) -> list[dict[str, str]]:
        """Build the chat message list for the LLM completion call.

        Args:
            query: The user's original question.
            context: Pre-formatted context string from ``build_context``.
            intent: Detected query intent (factual, analytical, etc.).
            system_prompt: Optional override for the default system prompt.

        Returns:
            List of message dicts with ``role`` and ``content`` keys.
        """
        sys_prompt = system_prompt or self._config.system_prompt or DEFAULT_SYSTEM_PROMPT

        # Append intent hint so the model can adjust response style
        if intent:
            sys_prompt += f"\n\nThe user's question is classified as: {intent}."

        user_content = f"Context:\n{context}\n\n---\n\nQuestion: {query}"

        return [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_content},
        ]
