"""Reference resolver -- detects and fetches cross-document references.

Inspired by Lawren's reference_resolver.rs: uses an LLM to detect when a chunk
references other documents or sections, then fetches those referenced chunks
from the state store. Caps traversal depth and total references to prevent
explosion.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from rag_common.llm.client import LLMClient
from rag_common.redis.state import StateStore

logger = logging.getLogger(__name__)

_REF_DETECTION_SYSTEM_PROMPT = (
    "You detect cross-references in text. Given a text passage, identify any "
    "explicit references to other documents, sections, appendices, tables, "
    "figures, or named entities that the reader would need for full context. "
    "Return ONLY valid JSON: {\"references\": [\"document_name or section\"]}. "
    "If no references, return {\"references\": []}."
)

_REF_DETECTION_USER_TEMPLATE = "Text:\n{text}\n\nIdentify any references to other documents or sections."


class ReferenceResolver:
    """Detects and resolves cross-document references in chunks.

    For each chunk, asks an LLM whether it references other documents/sections,
    then attempts to fetch those referenced items from the state store.
    Traverses up to ``max_depth`` levels and returns at most ``max_references``
    additional chunks.
    """

    def __init__(
        self,
        state_store: StateStore,
        llm_client: LLMClient,
        *,
        model: str = "openai/gpt-4o-mini",
    ) -> None:
        self._state = state_store
        self._llm = llm_client
        self._model = model

    # Hard cap on total LLM calls per resolve() invocation to prevent runaway cost.
    _MAX_LLM_CALLS: int = 10

    async def resolve(
        self,
        chunks: list[dict[str, Any]],
        max_depth: int = 2,
        max_references: int = 10,
    ) -> list[dict[str, Any]]:
        """Resolve references found in *chunks*.

        Args:
            chunks: Primary chunks to scan for references.
            max_depth: Maximum recursive depth for following references.
            max_references: Hard cap on total referenced chunks returned.

        Returns:
            A list of additional referenced chunks (does NOT include the
            original *chunks*). May be empty if no references are found.
        """
        if not chunks:
            return []

        referenced: list[dict[str, Any]] = []
        seen_refs: set[str] = set()
        llm_call_count: list[int] = [0]  # mutable counter shared across recursion

        await self._resolve_recursive(
            chunks=chunks,
            depth=0,
            max_depth=max_depth,
            max_references=max_references,
            referenced=referenced,
            seen_refs=seen_refs,
            llm_call_count=llm_call_count,
        )

        logger.info(
            "Reference resolver: found %d referenced chunks (max_depth=%d)",
            len(referenced),
            max_depth,
        )
        return referenced

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _resolve_recursive(
        self,
        chunks: list[dict[str, Any]],
        depth: int,
        max_depth: int,
        max_references: int,
        referenced: list[dict[str, Any]],
        seen_refs: set[str],
        llm_call_count: list[int],
    ) -> None:
        """Recursively detect and fetch references up to max_depth."""
        if depth >= max_depth or len(referenced) >= max_references:
            return
        if llm_call_count[0] >= self._MAX_LLM_CALLS:
            logger.info("LLM call cap (%d) reached, stopping reference resolution", self._MAX_LLM_CALLS)
            return

        for chunk in chunks:
            if len(referenced) >= max_references:
                break
            if llm_call_count[0] >= self._MAX_LLM_CALLS:
                break

            llm_call_count[0] += 1
            ref_names = await self._detect_references(chunk)
            new_chunks: list[dict[str, Any]] = []

            for ref_name in ref_names:
                if len(referenced) >= max_references:
                    break
                if ref_name in seen_refs:
                    continue
                seen_refs.add(ref_name)

                fetched = await self._fetch_reference(ref_name, chunk)
                if fetched is not None:
                    fetched["reference_source"] = chunk.get("chunk_id", "unknown")
                    fetched["reference_depth"] = depth + 1
                    referenced.append(fetched)
                    new_chunks.append(fetched)

            # Recurse into newly found chunks
            if new_chunks and depth + 1 < max_depth:
                await self._resolve_recursive(
                    chunks=new_chunks,
                    depth=depth + 1,
                    max_depth=max_depth,
                    max_references=max_references,
                    referenced=referenced,
                    seen_refs=seen_refs,
                    llm_call_count=llm_call_count,
                )

    async def _detect_references(self, chunk: dict[str, Any]) -> list[str]:
        """Use LLM to detect references in a single chunk."""
        text = chunk.get("text", "")
        if not text:
            return []

        user_message = _REF_DETECTION_USER_TEMPLATE.format(text=text[:1000])

        try:
            response = await self._llm.completion(
                messages=[
                    {"role": "system", "content": _REF_DETECTION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                model=self._model,
                temperature=0.0,
                max_tokens=256,
            )

            content = response.choices[0].message.content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

            parsed = json.loads(content)
            refs = parsed.get("references", [])
            return [r for r in refs if isinstance(r, str) and r.strip()]

        except Exception:
            logger.exception("Reference detection failed for chunk %s", chunk.get("chunk_id", "?"))
            return []

    async def _fetch_reference(
        self,
        ref_name: str,
        source_chunk: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Attempt to fetch a referenced chunk from the state store.

        Uses the task_id from the source chunk to look up related results.
        Falls back gracefully if the reference cannot be resolved.
        """
        task_id = source_chunk.get("task_id", "")
        if not task_id:
            return None

        # Try to find referenced content in state store under a lookup key
        lookup_key = f"ref:{ref_name}"
        result = await self._state.get_result(task_id, lookup_key)
        if result is not None and isinstance(result, dict):
            return result

        # Try document-level lookup
        doc_id = source_chunk.get("document_id", "")
        if doc_id:
            doc_result = await self._state.get_result(doc_id, f"section:{ref_name}")
            if doc_result is not None and isinstance(doc_result, dict):
                return doc_result

        logger.debug(
            "Could not resolve reference '%s' from chunk %s",
            ref_name,
            source_chunk.get("chunk_id", "?"),
        )
        return None
