"""Graph storage service — extracts entities and persists them in Neo4j."""

from __future__ import annotations

import logging
from typing import Any

from rag_common.llm.client import LLMClient
from rag_common.redis.client import RedisClient
from rag_common.redis.state import StateStore

from agents.graph.graph_agent.config import GraphConfig
from agents.graph.graph_agent.extractor import EntityExtractor
from agents.graph.graph_agent.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


class GraphService:
    """Extracts entities from document chunks and stores them as a knowledge graph.

    Creates the following node types and relationships in Neo4j:
    - ``(:Document {id, filename, format})``
    - ``(:Chunk {id, text, sequence_index})`` linked to Document via ``[:CONTAINS]``
    - ``(:Entity {name, type})`` linked to Chunk via ``[:MENTIONS]``
    - ``(:Entity)-[:RELATED_TO]->(:Entity)`` for cross-entity relationships
    """

    def __init__(
        self,
        redis_client: RedisClient,
        neo4j_client: Neo4jClient,
        llm_client: LLMClient,
        config: GraphConfig,
    ) -> None:
        self._state = StateStore(redis_client)
        self._neo4j = neo4j_client
        self._extractor = EntityExtractor(llm_client, config)
        self._config = config

    @staticmethod
    def _deduplicate(
        entities: list[dict[str, Any]],
        relationships: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Remove duplicate entities and relationships before Neo4j merge.

        Entities are keyed by (lowercased name, type).
        Relationships are keyed by (source, target, type).
        """
        seen_entities: dict[tuple[str, str], bool] = {}
        deduped_entities: list[dict[str, Any]] = []
        for e in entities:
            key = (e.get("name", "").lower(), e.get("type", ""))
            if key not in seen_entities:
                seen_entities[key] = True
                deduped_entities.append(e)

        seen_rels: set[tuple[str, str, str]] = set()
        deduped_rels: list[dict[str, Any]] = []
        for r in relationships:
            key = (r.get("source", ""), r.get("target", ""), r.get("type", ""))
            if key not in seen_rels:
                seen_rels.add(key)
                deduped_rels.append(r)

        return deduped_entities, deduped_rels

    async def store_entities(
        self,
        task_id: str,
        document_id: str,
        chunks: list[dict[str, Any]],
        *,
        filename: str = "",
        doc_format: str = "",
    ) -> dict[str, Any]:
        """Extract entities from chunks and persist the knowledge graph.

        Args:
            task_id: Pipeline task identifier for logging/tracing.
            document_id: The parent document UUID.
            chunks: List of chunk dicts with at least ``chunk_id``, ``text``,
                ``sequence_index``.
            filename: Original filename of the document.
            doc_format: Document format (e.g. ``"pdf"``, ``"markdown"``).

        Returns:
            A dict with ``success``, ``nodes_created``, ``relationships_created``.
        """
        nodes_created = 0
        relationships_created = 0

        # 1. Create Document node
        await self._neo4j.merge_node("Document", {
            "id": document_id,
            "filename": filename,
            "format": doc_format,
        })
        nodes_created += 1

        # 2. Create Chunk nodes and link to Document
        for chunk in chunks:
            chunk_id = str(chunk.get("chunk_id", ""))
            if not chunk_id:
                continue

            await self._neo4j.merge_node("Chunk", {
                "id": chunk_id,
                "text": chunk.get("text", "")[:5000],  # Truncate for Neo4j storage
                "sequence_index": chunk.get("sequence_index", 0),
            })
            nodes_created += 1

            await self._neo4j.merge_relationship(
                source_label="Document",
                source_props={"id": document_id},
                rel_type="CONTAINS",
                target_label="Chunk",
                target_props={"id": chunk_id},
                rel_props={"sequence_index": chunk.get("sequence_index", 0)},
            )
            relationships_created += 1

        # 3. Extract entities via LLM
        extraction = await self._extractor.extract(chunks)
        entities = extraction.get("entities", [])
        relationships = extraction.get("relationships", [])

        # 3.5. Deduplicate entities and relationships before Neo4j merge
        entities, relationships = self._deduplicate(entities, relationships)

        # 4. Create Entity nodes and link to Chunks
        entity_name_set: set[str] = set()
        for entity in entities:
            name = entity.get("name", "")
            etype = entity.get("type", "CONCEPT")
            if not name:
                continue

            await self._neo4j.merge_node("Entity", {
                "name": name,
                "type": etype,
                **entity.get("properties", {}),
            })
            nodes_created += 1
            entity_name_set.add(name)

            # Link entity to chunks that mention it
            for chunk in chunks:
                chunk_text = chunk.get("text", "").lower()
                if name.lower() in chunk_text:
                    chunk_id = str(chunk.get("chunk_id", ""))
                    if chunk_id:
                        await self._neo4j.merge_relationship(
                            source_label="Chunk",
                            source_props={"id": chunk_id},
                            rel_type="MENTIONS",
                            target_label="Entity",
                            target_props={"name": name},
                        )
                        relationships_created += 1

        # 5. Create cross-entity relationships
        for rel in relationships:
            source = rel.get("source", "")
            target = rel.get("target", "")
            rel_type = rel.get("type", "RELATED_TO")
            if not source or not target:
                continue

            await self._neo4j.merge_relationship(
                source_label="Entity",
                source_props={"name": source},
                rel_type=rel_type,
                target_label="Entity",
                target_props={"name": target},
                rel_props=rel.get("properties"),
            )
            relationships_created += 1

        logger.info(
            "task_id=%s: created %d nodes, %d relationships in Neo4j",
            task_id,
            nodes_created,
            relationships_created,
        )

        return {
            "success": True,
            "nodes_created": nodes_created,
            "relationships_created": relationships_created,
        }
