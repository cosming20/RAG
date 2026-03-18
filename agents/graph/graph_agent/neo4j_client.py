"""Async Neo4j client wrapper with connection lifecycle."""

from __future__ import annotations

import logging
import re
from typing import Any

from neo4j import AsyncGraphDatabase, AsyncDriver

from rag_common.config import Neo4jConfig

logger = logging.getLogger(__name__)

_VALID_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _sanitize_cypher_identifier(value: str) -> str:
    """Sanitize a Cypher label or relationship type to prevent injection."""
    cleaned = re.sub(r"[^A-Za-z0-9_]", "", value)
    if not cleaned or not _VALID_IDENTIFIER.match(cleaned):
        raise ValueError(f"Invalid Cypher identifier: {value!r}")
    return cleaned


class Neo4jClient:
    """Thin async wrapper around the Neo4j Python driver.

    Manages the driver lifecycle and exposes convenience methods for
    common graph operations (execute_query, merge_node, merge_relationship).
    """

    def __init__(self, config: Neo4jConfig) -> None:
        self._config = config
        self._driver: AsyncDriver | None = None

    @property
    def driver(self) -> AsyncDriver:
        """Return the connected async Neo4j driver.

        Raises:
            RuntimeError: If ``connect()`` has not been called.
        """
        if self._driver is None:
            msg = "Neo4jClient not connected. Call connect() first."
            raise RuntimeError(msg)
        return self._driver

    async def connect(self) -> None:
        """Create the async Neo4j driver and verify connectivity."""
        password = self._config.password.get_secret_value()
        self._driver = AsyncGraphDatabase.driver(
            self._config.uri,
            auth=(self._config.username, password),
            max_connection_pool_size=self._config.max_connection_pool_size,
        )
        await self._driver.verify_connectivity()
        logger.info("Connected to Neo4j at %s", self._config.uri)

    async def close(self) -> None:
        """Close the Neo4j driver."""
        if self._driver is not None:
            await self._driver.close()
            self._driver = None
            logger.info("Neo4j connection closed")

    async def execute_query(
        self,
        query: str,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute a Cypher query and return all result records as dicts.

        Args:
            query: Cypher query string.
            params: Optional parameter dict for the query.

        Returns:
            A list of record dicts.
        """
        async with self.driver.session(database=self._config.database) as session:
            result = await session.run(query, parameters=params or {})
            records = await result.data()
            return records

    async def merge_node(
        self,
        label: str,
        properties: dict[str, Any],
    ) -> None:
        """MERGE a node with the given label, keyed on ``id`` or ``name``.

        If the properties contain an ``id`` field it is used as the merge key;
        otherwise ``name`` is used.

        Args:
            label: The node label (e.g. ``"Entity"``, ``"Document"``).
            properties: Property dict for the node.
        """
        label = _sanitize_cypher_identifier(label)
        merge_key = "id" if "id" in properties else "name"
        merge_value = properties[merge_key]

        set_clause = ", ".join(
            f"n.{k} = ${k}" for k in properties if k != merge_key
        )
        query = f"MERGE (n:{label} {{{merge_key}: ${merge_key}}})"
        if set_clause:
            query += f" SET {set_clause}"

        await self.execute_query(query, properties)

    async def merge_relationship(
        self,
        source_label: str,
        source_props: dict[str, Any],
        rel_type: str,
        target_label: str,
        target_props: dict[str, Any],
        rel_props: dict[str, Any] | None = None,
    ) -> None:
        """MERGE a relationship between two nodes.

        Both source and target nodes are matched/merged by their ``id`` or
        ``name`` property (whichever is present).

        Args:
            source_label: Label of the source node.
            source_props: Properties to identify the source node.
            rel_type: Relationship type (e.g. ``"CONTAINS"``, ``"MENTIONS"``).
            target_label: Label of the target node.
            target_props: Properties to identify the target node.
            rel_props: Optional properties set on the relationship.
        """
        source_label = _sanitize_cypher_identifier(source_label)
        target_label = _sanitize_cypher_identifier(target_label)
        rel_type = _sanitize_cypher_identifier(rel_type)
        src_key = "id" if "id" in source_props else "name"
        tgt_key = "id" if "id" in target_props else "name"

        params: dict[str, Any] = {
            f"src_{src_key}": source_props[src_key],
            f"tgt_{tgt_key}": target_props[tgt_key],
        }

        rel_set = ""
        if rel_props:
            for k, v in rel_props.items():
                params[f"rel_{k}"] = v
            rel_set_parts = ", ".join(f"r.{k} = $rel_{k}" for k in rel_props)
            rel_set = f" SET {rel_set_parts}"

        query = (
            f"MERGE (a:{source_label} {{{src_key}: $src_{src_key}}})"
            f" MERGE (b:{target_label} {{{tgt_key}: $tgt_{tgt_key}}})"
            f" MERGE (a)-[r:{rel_type}]->(b)"
            f"{rel_set}"
        )

        await self.execute_query(query, params)
