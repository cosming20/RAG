"""Graph query worker — enriches queries with knowledge graph context.

Consumes from ``rag:stream:query:classified`` (fan-out with the Retrieval
query worker), extracts entities from the query via LLM, expands through
the knowledge graph, and stores the graph context for downstream synthesis.
"""

from __future__ import annotations

import logging
from typing import Any

from agents.graph.graph_agent.config import GraphConfig
from agents.graph.graph_agent.expansion import GraphExpander
from rag_common.config import AgentConfig
from rag_common.redis.client import RedisClient
from rag_common.redis.streams import StreamMessage
from rag_common.redis.worker import BaseStreamWorker

logger = logging.getLogger(__name__)

INPUT_STREAM = "rag:stream:query:classified"
OUTPUT_STREAM = "rag:stream:query:graph_enriched"
DLQ_STREAM = "rag:stream:graph-query:dlq"
CONSUMER_GROUP = "graph-query-agents"


class GraphQueryWorker(BaseStreamWorker):
    """Stream worker that enriches queries with knowledge graph context.

    Pipeline:
        1. Read classification from StateStore (``rag:results:{task_id}:classification``)
        2. Extract entities from query via ``GraphExpander.extract_query_entities()``
        3. Expand context via ``GraphExpander.expand_context()`` or
           ``query_subgraph()`` depending on query type
        4. Store at ``rag:results:{task_id}:graph_context``
        5. Forward envelope to ``rag:stream:query:graph_enriched``
    """

    def __init__(
        self,
        redis_client: RedisClient,
        expander: GraphExpander,
        graph_config: GraphConfig,
        *,
        consumer_id: str,
        agent_config: AgentConfig | None = None,
    ) -> None:
        super().__init__(
            redis_client,
            input_stream=INPUT_STREAM,
            output_stream=OUTPUT_STREAM,
            dlq_stream=DLQ_STREAM,
            consumer_group=CONSUMER_GROUP,
            consumer_id=consumer_id,
            agent_config=agent_config,
        )
        self._expander = expander
        self._graph_config = graph_config

    async def _process_message(self, msg: StreamMessage) -> Any:
        """Extract entities, expand graph context, store results.

        Returns:
            Dict with ``output_ref`` pointing to the state store key.
        """
        task_id = msg.task_id
        logger.info("GraphQueryWorker processing task_id=%s", task_id)

        # 1. Read classification
        classification = await self._state.get_result(task_id, "classification")
        if classification is None:
            raise ValueError(
                f"task_id={task_id}: no classification found in state store"
            )

        query_text: str = classification.get("query_text", "")
        if not query_text:
            raise ValueError(
                f"task_id={task_id}: classification missing query_text"
            )

        query_type: str = classification.get("query_type", "")
        max_depth = classification.get(
            "graph_depth", self._graph_config.max_depth,
        )

        # 2. Extract entities from query
        entities = await self._expander.extract_query_entities(query_text)
        entity_names = [e["name"] for e in entities]

        # 3. Expand context — use shallow subgraph for FACTUAL, deep for others
        if query_type.upper() == "FACTUAL" and entity_names:
            graph_context = await self._expander.query_subgraph(entity_names)
        elif entity_names:
            graph_context = await self._expander.expand_context(
                entity_names, max_depth=max_depth,
            )
        else:
            graph_context = {
                "entities": [],
                "paths": [],
                "context_text": "",
                "related_chunks": [],
            }

        # 4. Store results
        result_data = {
            "query_text": query_text,
            "extracted_entities": entities,
            "graph_context": graph_context,
            "entity_count": len(entities),
            "related_entity_count": len(graph_context.get("entities", [])),
            "chunk_count": len(graph_context.get("related_chunks", [])),
        }
        output_ref = await self._state.store_result(
            task_id, "graph_context", result_data,
        )

        logger.info(
            "task_id=%s: extracted %d entities, expanded to %d related, %d chunks",
            task_id,
            len(entities),
            len(graph_context.get("entities", [])),
            len(graph_context.get("related_chunks", [])),
        )

        return {"output_ref": output_ref}
