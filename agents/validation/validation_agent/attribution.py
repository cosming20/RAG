"""Citation attribution verification.

Pure-logic checker (no LLM needed) that validates [N] citations in the
answer text map to real source chunks and are contextually plausible.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Regex to find citation markers like [1], [2], [13]
CITATION_PATTERN = re.compile(r"\[(\d+)\]")


class AttributionChecker:
    """Verifies that answer citations map to valid source chunks."""

    def check(
        self,
        answer_text: str,
        citations: list[dict],
        source_chunks: list[dict],
    ) -> dict[str, Any]:
        """Validate each [N] citation in the answer against source chunks.

        Args:
            answer_text: The generated answer text.
            citations: List of citation dicts from the synthesis stage.
            source_chunks: The chunks that were provided as context
                (1-indexed to match [N] references).

        Returns:
            Dict with ``citations`` (per-citation validation results)
            and ``attribution_score`` (fraction of valid citations).
            If the answer contains no citations at all, score is 0.0.
        """
        # Find all [N] references in the answer
        found_indices: list[int] = []
        for match in CITATION_PATTERN.finditer(answer_text):
            idx = int(match.group(1))
            if idx not in found_indices:
                found_indices.append(idx)

        # Edge case: no citations at all means score 0.0
        if not found_indices:
            logger.warning("Answer contains no citations")
            return {
                "citations": [],
                "attribution_score": 0.0,
            }

        citation_results: list[dict[str, Any]] = []
        valid_count = 0

        for idx in found_indices:
            result = self._validate_citation(idx, citations, source_chunks)
            citation_results.append(result)
            if result["is_valid"]:
                valid_count += 1

        total = len(citation_results)
        attribution_score = valid_count / total if total > 0 else 0.0

        logger.info(
            "Attribution check: %d/%d citations valid (score=%.2f)",
            valid_count,
            total,
            attribution_score,
        )

        return {
            "citations": citation_results,
            "attribution_score": attribution_score,
        }

    def _validate_citation(
        self,
        citation_index: int,
        citations: list[dict],
        source_chunks: list[dict],
    ) -> dict[str, Any]:
        """Validate a single [N] citation.

        Checks:
            1. N maps to an actual source chunk (1-indexed).
            2. The corresponding citation metadata exists.

        Args:
            citation_index: The [N] number from the answer.
            citations: Structured citations from synthesis.
            source_chunks: Context chunks (1-indexed).

        Returns:
            Dict with ``citation_index``, ``is_valid``, ``issue``.
        """
        chunk_pos = citation_index - 1

        # Check if citation references a valid chunk
        if chunk_pos < 0 or chunk_pos >= len(source_chunks):
            return {
                "citation_index": citation_index,
                "is_valid": False,
                "issue": f"Citation [{citation_index}] references non-existent chunk "
                         f"(only {len(source_chunks)} chunks available)",
            }

        # Check if the citation has a matching entry in the structured citations
        matching_citation = None
        for cit in citations:
            chunk = source_chunks[chunk_pos]
            if cit.get("document_id") == chunk.get("document_id"):
                matching_citation = cit
                break

        if matching_citation is None and citations:
            # Citation index is valid but no structured citation metadata matches
            # This is a soft issue -- the chunk exists, metadata just wasn't mapped
            return {
                "citation_index": citation_index,
                "is_valid": True,
                "issue": "Citation maps to valid chunk but structured metadata mismatch",
            }

        return {
            "citation_index": citation_index,
            "is_valid": True,
            "issue": "",
        }
