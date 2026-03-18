"""Graph agent entry point — storage + query paths.

Starts the gRPC health server, connects to Neo4j, and runs both:
- GraphStorageWorker: consumes from ``rag:stream:ingest:embedded``
  and persists entities/relationships in Neo4j.
- GraphQueryWorker: consumes from ``rag:stream:query:classified``
  and enriches queries with knowledge graph context.
"""

from __future__ import annotations

import asyncio
import logging

from rag_common.config import AgentConfig, LLMConfig, Neo4jConfig
from rag_common.grpc_utils.server import BaseGrpcServer
from rag_common.llm.client import LLMClient
from rag_common.models.agent import AgentRole

from agents.graph.graph_agent.config import GraphConfig
from agents.graph.graph_agent.expansion import GraphExpander
from agents.graph.graph_agent.neo4j_client import Neo4jClient
from agents.graph.graph_agent.query_worker import GraphQueryWorker
from agents.graph.graph_agent.service import GraphService
from agents.graph.graph_agent.worker import GraphStorageWorker

logger = logging.getLogger(__name__)


async def main() -> None:
    """Bootstrap and run the Graph agent (storage + query)."""
    graph_config = GraphConfig()
    neo4j_config = Neo4jConfig()
    llm_config = LLMConfig()
    agent_config = AgentConfig()

    # --- gRPC server ---
    server = BaseGrpcServer(
        role=AgentRole.GRAPH,
        agent_config=agent_config,
    )
    await server.start()

    # --- Neo4j client ---
    neo4j_client = Neo4jClient(neo4j_config)
    await neo4j_client.connect()

    # --- LLM client ---
    llm_client = LLMClient(llm_config)

    # --- Storage path ---
    service = GraphService(
        server.redis_client,
        neo4j_client,
        llm_client,
        graph_config,
    )
    storage_worker = GraphStorageWorker(
        server.redis_client,
        service,
        consumer_id=server.agent_id,
    )

    # --- Query path ---
    expander = GraphExpander(neo4j_client, llm_client, graph_config)
    query_worker = GraphQueryWorker(
        server.redis_client,
        expander,
        graph_config,
        consumer_id=f"{server.agent_id}-query",
        agent_config=agent_config,
    )

    # Run both workers as background tasks
    storage_task = asyncio.create_task(storage_worker.start())
    query_task = asyncio.create_task(query_worker.start())

    try:
        await server.wait_for_shutdown()
    finally:
        # StorageWorker has async stop; QueryWorker (BaseStreamWorker) has sync stop
        await storage_worker.stop()
        query_worker.stop()
        storage_task.cancel()
        query_task.cancel()
        for task in (storage_task, query_task):
            try:
                await task
            except asyncio.CancelledError:
                pass
        await neo4j_client.close()

    logger.info("Graph agent shut down")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    asyncio.run(main())
