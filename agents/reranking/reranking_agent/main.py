"""Reranking agent entry point.

Starts the gRPC health server, initializes the Jina reranker and LLM client,
and runs the RerankingWorker that consumes from ``rag:stream:query:retrieved``
and publishes reranked results to ``rag:stream:query:reranked``.
"""

from __future__ import annotations

import asyncio
import logging

from rag_common.config import AgentConfig, JinaConfig, LLMConfig
from rag_common.grpc_utils.server import BaseGrpcServer
from rag_common.llm.client import LLMClient
from rag_common.models.agent import AgentRole

from agents.reranking.reranking_agent.config import RerankingConfig
from agents.reranking.reranking_agent.service import RerankingService
from agents.reranking.reranking_agent.worker import RerankingWorker

logger = logging.getLogger(__name__)


async def main() -> None:
    """Bootstrap and run the Reranking agent."""
    reranking_config = RerankingConfig()
    jina_config = JinaConfig()
    llm_config = LLMConfig()
    agent_config = AgentConfig()

    # --- gRPC server ---
    server = BaseGrpcServer(
        role=AgentRole.RERANKING,
        agent_config=agent_config,
    )
    await server.start()

    # --- LLM client ---
    llm_client = LLMClient(llm_config)

    # --- Service + Worker ---
    service = RerankingService(
        redis_client=server.redis_client,
        llm_client=llm_client,
        config=reranking_config,
        jina_config=jina_config,
    )
    worker = RerankingWorker(
        redis_client=server.redis_client,
        service=service,
        config=reranking_config,
        consumer_id=server.agent_id,
    )

    # Run worker as a background task
    worker_task = asyncio.create_task(worker.start())

    try:
        await server.wait_for_shutdown()
    finally:
        worker.stop()
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass

    logger.info("Reranking agent shut down")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    asyncio.run(main())
