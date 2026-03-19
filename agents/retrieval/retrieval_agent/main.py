"""Retrieval agent entry point — storage + query paths.

Starts the gRPC health server, connects to Qdrant, and runs both:
- RetrievalStorageWorker: consumes from ``rag:stream:ingest:embedded``
  and upserts chunk vectors into Qdrant.
- RetrievalQueryWorker: consumes from ``rag:stream:query:classified``
  and performs hybrid search over Qdrant.
"""

from __future__ import annotations

import asyncio
import logging

from rag_common.config import AgentConfig, LLMConfig, QdrantConfig, RedisConfig
from rag_common.grpc_utils.server import BaseGrpcServer
from rag_common.llm.client import LLMClient
from rag_common.models.agent import AgentRole

from agents.retrieval.retrieval_agent.config import RetrievalConfig
from agents.retrieval.retrieval_agent.grpc_servicer import RetrievalServicer
from agents.retrieval.retrieval_agent.pool import QdrantPool
from agents.retrieval.retrieval_agent.query_worker import RetrievalQueryWorker
from agents.retrieval.retrieval_agent.searcher import QdrantSearcher
from agents.retrieval.retrieval_agent.service import RetrievalService
from agents.retrieval.retrieval_agent.worker import RetrievalStorageWorker

logger = logging.getLogger(__name__)


async def main() -> None:
    """Bootstrap and run the Retrieval agent (storage + query)."""
    retrieval_config = RetrievalConfig()
    qdrant_config = QdrantConfig()
    llm_config = LLMConfig()
    agent_config = AgentConfig()

    # --- gRPC server ---
    server = BaseGrpcServer(
        role=AgentRole.RETRIEVAL,
        agent_config=agent_config,
    )
    await server.start()

    # --- Qdrant pool ---
    pool = QdrantPool(qdrant_config, retrieval_config)
    await pool.connect()

    # --- LLM client (for query-time embedding) ---
    llm_client = LLMClient(llm_config)

    # --- Storage path ---
    service = RetrievalService(server.redis_client, pool, retrieval_config)

    # --- Query path (searcher created early for servicer) ---
    searcher = QdrantSearcher(pool, retrieval_config)

    # --- gRPC servicer (ready for proto registration) ---
    servicer = RetrievalServicer(service, searcher)
    # TODO: Register when proto generation is complete:
    # from rag_common.generated.services import retrieval_pb2_grpc
    # retrieval_pb2_grpc.add_RetrievalAgentServicer_to_server(servicer, server._server)
    logger.info("gRPC servicer created for %s (awaiting proto registration)", AgentRole.RETRIEVAL)

    storage_worker = RetrievalStorageWorker(
        server.redis_client,
        service,
        consumer_id=server.agent_id,
    )

    # --- Query worker (reuses searcher from above) ---
    query_worker = RetrievalQueryWorker(
        server.redis_client,
        searcher,
        llm_client,
        retrieval_config,
        consumer_id=f"{server.agent_id}-query",
        agent_config=agent_config,
    )

    # Run both workers as background tasks
    storage_task = asyncio.create_task(storage_worker.start())
    query_task = asyncio.create_task(query_worker.start())

    try:
        await server.wait_for_shutdown()
    finally:
        # Both workers extend BaseStreamWorker with sync stop()
        storage_worker.stop()
        query_worker.stop()
        storage_task.cancel()
        query_task.cancel()
        for task in (storage_task, query_task):
            try:
                await task
            except asyncio.CancelledError:
                pass
        await pool.close()

    logger.info("Retrieval agent shut down")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    asyncio.run(main())
