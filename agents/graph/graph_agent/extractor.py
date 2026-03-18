"""Entity and relationship extraction from text chunks via LLM."""

from __future__ import annotations

import json
import logging
from typing import Any

from rag_common.llm.client import LLMClient

from agents.graph.graph_agent.config import GraphConfig

logger = logging.getLogger(__name__)

EXTRACTION_SYSTEM_PROMPT = (
    "Extract entities and relationships from the text. "
    "Return JSON with format: "
    '{"entities": [{"name": "...", "type": "PERSON|ORG|CONCEPT|DATE|LOCATION", "properties": {}}], '
    '"relationships": [{"source": "...", "target": "...", '
    '"type": "RELATED_TO|MENTIONS|CONTAINS|DEPENDS_ON|PART_OF|AUTHORED_BY", "properties": {}}]}'
    "\n\nRules:\n"
    "- Entity names should be normalized (title case for proper nouns).\n"
    "- Deduplicate entities that refer to the same real-world object.\n"
    "- Only extract clearly stated relationships, do not infer.\n"
    "- Return valid JSON only, no markdown fences."
)

CHUNK_BATCH_SIZE = 5


class EntityExtractor:
    """Extracts named entities and relationships from text chunks using an LLM.

    Chunks are batched (default 5 at a time) to reduce the number of LLM calls.
    Entities are deduplicated across all chunks by (name, type) pair.
    """

    def __init__(self, llm_client: LLMClient, config: GraphConfig) -> None:
        self._llm = llm_client
        self._config = config

    async def extract(
        self,
        chunks: list[dict[str, Any]],
    ) -> dict[str, list[dict[str, Any]]]:
        """Extract entities and relationships from a list of chunk dicts.

        Each chunk dict must have at least a ``text`` key.

        Args:
            chunks: Chunk dicts with ``text`` field.

        Returns:
            A dict with ``entities`` and ``relationships`` lists.
        """
        all_entities: list[dict[str, Any]] = []
        all_relationships: list[dict[str, Any]] = []

        for batch_start in range(0, len(chunks), CHUNK_BATCH_SIZE):
            batch = chunks[batch_start : batch_start + CHUNK_BATCH_SIZE]
            combined_text = "\n\n---\n\n".join(
                c.get("text", "") for c in batch
            )

            result = await self._extract_from_text(combined_text)
            all_entities.extend(result.get("entities", []))
            all_relationships.extend(result.get("relationships", []))

        # Deduplicate entities by (name, type)
        deduped_entities = self._deduplicate_entities(all_entities)

        logger.info(
            "Extracted %d unique entities, %d relationships from %d chunks",
            len(deduped_entities),
            len(all_relationships),
            len(chunks),
        )

        return {
            "entities": deduped_entities,
            "relationships": all_relationships,
        }

    async def _extract_from_text(self, text: str) -> dict[str, Any]:
        """Call the LLM to extract entities/relationships from a text block."""
        messages = [
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]

        try:
            response = await self._llm.completion(
                messages,
                model=self._config.relationship_extraction_model,
                temperature=0.0,
            )
            content = response.choices[0].message.content
            return self._parse_json_response(content)
        except Exception:
            logger.exception("LLM extraction failed for text (len=%d)", len(text))
            return {"entities": [], "relationships": []}

    @staticmethod
    def _parse_json_response(content: str) -> dict[str, Any]:
        """Parse JSON from LLM response, handling markdown fences if present."""
        cleaned = content.strip()
        if cleaned.startswith("```"):
            # Strip markdown code fence
            lines = cleaned.split("\n")
            # Remove first and last lines (fences)
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines)

        try:
            parsed = json.loads(cleaned)
            if not isinstance(parsed, dict):
                return {"entities": [], "relationships": []}
            return {
                "entities": parsed.get("entities", []),
                "relationships": parsed.get("relationships", []),
            }
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM response as JSON: %s...", cleaned[:200])
            return {"entities": [], "relationships": []}

    @staticmethod
    def _deduplicate_entities(
        entities: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Deduplicate entities by (name, type) pair, merging properties."""
        seen: dict[tuple[str, str], dict[str, Any]] = {}
        for entity in entities:
            name = entity.get("name", "").strip()
            etype = entity.get("type", "CONCEPT").strip()
            if not name:
                continue
            key = (name.lower(), etype.upper())
            if key in seen:
                # Merge properties from duplicate
                existing_props = seen[key].get("properties", {})
                new_props = entity.get("properties", {})
                existing_props.update(new_props)
                seen[key]["properties"] = existing_props
            else:
                seen[key] = {
                    "name": name,
                    "type": etype.upper(),
                    "properties": entity.get("properties", {}),
                }
        return list(seen.values())
