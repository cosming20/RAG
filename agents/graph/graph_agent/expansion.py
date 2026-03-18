"""Graph expansion — multi-hop traversal for query-time context enrichment.

Extracts entities from a natural-language query via LLM, then expands
outward through the knowledge graph to gather contextually relevant
subgraphs. Inspired by Lawren's graph_expansion.rs BFS approach.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from agents.graph.graph_agent.config import GraphConfig
from agents.graph.graph_agent.neo4j_client import Neo4jClient
from rag_common.llm.client import LLMClient

logger = logging.getLogger(__name__)

_ENTITY_EXTRACTION_PROMPT = (
    "Extract entity names from this query. "
    "Return ONLY a JSON object with this exact format: "
    '{"entities": [{"name": "entity1", "type": "PERSON"}, '
    '{"name": "entity2", "type": "CONCEPT"}]}. '
    "Use types: PERSON, ORGANIZATION, LOCATION, CONCEPT, EVENT, TECHNOLOGY, OTHER. "
    "If no entities are found, return {\"entities\": []}.\n\n"
    "Query: {query}"
)


class GraphExpander:
    """Expand seed entities through the knowledge graph to gather context.

    Supports two modes:
    - ``expand_context``: BFS traversal up to ``max_depth`` hops, gathering
      related entities, paths, and associated chunks.
    - ``query_subgraph``: Shallow (1-hop) lookup for FACTUAL queries that
      don't need deep traversal.
    """

    def __init__(
        self,
        neo4j_client: Neo4jClient,
        llm_client: LLMClient,
        config: GraphConfig,
    ) -> None:
        self._neo4j = neo4j_client
        self._llm = llm_client
        self._config = config

    async def extract_query_entities(
        self, query_text: str,
    ) -> list[dict[str, str]]:
        """Use the LLM to extract entity names and types from a query.

        Args:
            query_text: The user's natural-language query.

        Returns:
            List of dicts with ``name`` (str) and ``type`` (str).
        """
        prompt = _ENTITY_EXTRACTION_PROMPT.format(query=query_text)
        response = await self._llm.completion(
            messages=[{"role": "user", "content": prompt}],
            model=self._config.relationship_extraction_model,
            temperature=0.0,
        )

        raw_text = response.choices[0].message.content.strip()
        entities = _parse_entities_json(raw_text)

        logger.info(
            "Extracted %d entities from query: %s",
            len(entities),
            [e["name"] for e in entities],
        )
        return entities

    async def expand_context(
        self,
        entity_names: list[str],
        *,
        max_depth: int | None = None,
        max_results: int = 20,
    ) -> dict[str, Any]:
        """BFS expansion from seed entities through the knowledge graph.

        Traverses up to ``max_depth`` hops (default from config) and collects
        related entities, relationship paths, and associated chunks.

        Args:
            entity_names: Seed entity names to expand from.
            max_depth: Maximum traversal depth (hops). Defaults to
                ``GraphConfig.max_depth``.
            max_results: Maximum number of related records to return.

        Returns:
            Dict with:
            - ``entities``: list of related entity dicts
            - ``paths``: list of path descriptions
            - ``context_text``: human-readable summary of the graph context
            - ``related_chunks``: list of chunk dicts linked to found entities
        """
        raw_depth = max_depth if max_depth is not None else self._config.max_depth
        depth = max(1, min(int(raw_depth), 5))  # Clamp 1-5

        if not entity_names:
            return _empty_context()

        # 1. BFS traversal to find related entities and paths
        traversal_query = (
            f"MATCH path = (e:Entity)-[r*1..{depth}]-(related) "
            "WHERE e.name IN $entity_names "
            "RETURN e.name AS source, "
            "       [rel IN r | type(rel)] AS rel_types, "
            "       [node IN nodes(path) | node.name] AS path_names, "
            "       related.name AS related_name, "
            "       related.type AS related_type, "
            "       labels(related) AS related_labels "
            "LIMIT $max_results"
        )

        traversal_records = await self._neo4j.execute_query(
            traversal_query,
            params={"entity_names": entity_names, "max_results": max_results},
        )

        # 2. Collect related entities (deduplicated)
        entities: list[dict[str, Any]] = []
        seen_entities: set[str] = set()
        paths: list[str] = []

        for record in traversal_records:
            related_name = record.get("related_name", "")
            if related_name and related_name not in seen_entities:
                seen_entities.add(related_name)
                entities.append({
                    "name": related_name,
                    "type": record.get("related_type", "UNKNOWN"),
                    "labels": record.get("related_labels", []),
                })

            # Build path description
            path_names = record.get("path_names", [])
            rel_types = record.get("rel_types", [])
            if path_names and rel_types:
                path_desc = _format_path(path_names, rel_types)
                if path_desc not in paths:
                    paths.append(path_desc)

        # 3. Find chunks linked to the discovered entities
        all_entity_names = list(seen_entities | set(entity_names))
        related_chunks = await self._find_related_chunks(all_entity_names)

        # 4. Build context text summary
        context_text = _build_context_text(entity_names, entities, paths)

        logger.info(
            "Graph expansion: %d seed entities -> %d related, %d paths, %d chunks",
            len(entity_names),
            len(entities),
            len(paths),
            len(related_chunks),
        )

        return {
            "entities": entities,
            "paths": paths,
            "context_text": context_text,
            "related_chunks": related_chunks,
        }

    async def query_subgraph(
        self, entity_names: list[str],
    ) -> dict[str, Any]:
        """Shallow 1-hop lookup for FACTUAL queries.

        Finds directly connected entities and their associated chunks
        without deep traversal.

        Args:
            entity_names: Entity names to look up.

        Returns:
            Dict with ``entities``, ``paths``, ``context_text``,
            ``related_chunks``.
        """
        if not entity_names:
            return _empty_context()

        query = (
            "MATCH (e:Entity)-[r]-(related) "
            "WHERE e.name IN $entity_names "
            "RETURN e.name AS source, "
            "       type(r) AS rel_type, "
            "       related.name AS related_name, "
            "       related.type AS related_type, "
            "       labels(related) AS related_labels "
            "LIMIT 50"
        )

        records = await self._neo4j.execute_query(
            query, params={"entity_names": entity_names},
        )

        entities: list[dict[str, Any]] = []
        seen: set[str] = set()
        paths: list[str] = []

        for record in records:
            related_name = record.get("related_name", "")
            if related_name and related_name not in seen:
                seen.add(related_name)
                entities.append({
                    "name": related_name,
                    "type": record.get("related_type", "UNKNOWN"),
                    "labels": record.get("related_labels", []),
                })

            source = record.get("source", "")
            rel_type = record.get("rel_type", "")
            if source and related_name and rel_type:
                path_desc = f"{source} -[{rel_type}]-> {related_name}"
                if path_desc not in paths:
                    paths.append(path_desc)

        all_names = list(seen | set(entity_names))
        related_chunks = await self._find_related_chunks(all_names)
        context_text = _build_context_text(entity_names, entities, paths)

        return {
            "entities": entities,
            "paths": paths,
            "context_text": context_text,
            "related_chunks": related_chunks,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _find_related_chunks(
        self, entity_names: list[str],
    ) -> list[dict[str, Any]]:
        """Find Chunk nodes linked to the given entities via MENTIONS."""
        if not entity_names:
            return []

        query = (
            "MATCH (c:Chunk)-[:MENTIONS]->(e:Entity) "
            "WHERE e.name IN $entity_names "
            "RETURN DISTINCT c.id AS chunk_id, "
            "       c.text AS text, "
            "       c.sequence_index AS sequence_index "
            "LIMIT 50"
        )

        records = await self._neo4j.execute_query(
            query, params={"entity_names": entity_names},
        )

        return [
            {
                "chunk_id": record.get("chunk_id", ""),
                "text": record.get("text", ""),
                "sequence_index": record.get("sequence_index", 0),
            }
            for record in records
            if record.get("chunk_id")
        ]


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _empty_context() -> dict[str, Any]:
    """Return an empty graph context result."""
    return {
        "entities": [],
        "paths": [],
        "context_text": "",
        "related_chunks": [],
    }


def _parse_entities_json(raw_text: str) -> list[dict[str, str]]:
    """Parse the LLM's JSON response into a list of entity dicts.

    Handles common LLM output quirks (markdown fences, extra whitespace).
    """
    cleaned = raw_text.strip()
    # Strip markdown code fences if present
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Remove first and last fence lines
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()

    try:
        parsed = json.loads(cleaned)
        entities_raw = parsed.get("entities", [])
        return [
            {
                "name": e.get("name", ""),
                "type": e.get("type", "OTHER"),
            }
            for e in entities_raw
            if e.get("name")
        ]
    except (json.JSONDecodeError, AttributeError, TypeError):
        logger.warning("Failed to parse entity extraction response: %.200s", raw_text)
        return []


def _format_path(path_names: list[str], rel_types: list[str]) -> str:
    """Format a graph path as a human-readable string."""
    parts: list[str] = []
    for i, name in enumerate(path_names):
        parts.append(str(name) if name else "?")
        if i < len(rel_types):
            parts.append(f"-[{rel_types[i]}]->")
    return " ".join(parts)


def _build_context_text(
    seed_names: list[str],
    entities: list[dict[str, Any]],
    paths: list[str],
) -> str:
    """Build a human-readable context summary from graph expansion results."""
    lines: list[str] = []
    lines.append(f"Seed entities: {', '.join(seed_names)}")

    if entities:
        entity_strs = [
            f"{e['name']} ({e.get('type', 'UNKNOWN')})" for e in entities[:15]
        ]
        lines.append(f"Related entities: {', '.join(entity_strs)}")

    if paths:
        lines.append("Graph paths:")
        for path in paths[:10]:
            lines.append(f"  - {path}")

    return "\n".join(lines)
