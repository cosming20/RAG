"""Entry point for the Embedding agent."""

from __future__ import annotations

import asyncio
import logging

from rag_common.config import AgentConfig, LLMConfig, ObservabilityConfig, RedisConfig
from rag_common.grpc_utils.server import BaseGrpcServer
from rag_common.llm.client import LLMClient
from rag_common.models.agent import AgentRole

from agents.embedding.embedding_agent.config import EmbeddingConfig
from agents.embedding.embedding_agent.service import EmbeddingService
from agents.embedding.embedding_agent.grpc_servicer import EmbeddingServicer
from agents.embedding.embedding_agent.worker import EmbeddingWorker

logger = logging.getLogger(__name__)


async def main() -> None:
    """Bootstrap the Embedding agent: gRPC server + stream worker."""
    embedding_config = EmbeddingConfig()
    llm_config = LLMConfig()
    agent_config = AgentConfig()

    # --- gRPC server ---
    server = BaseGrpcServer(
        role=AgentRole.EMBEDDING,
        agent_config=agent_config,
        redis_config=RedisConfig(),
        otel_config=ObservabilityConfig(service_name="embedding-agent"),
    )
    await server.start()

    # --- LLM client ---
    llm_client = LLMClient(llm_config)

    # --- gRPC servicer (ready for proto registration) ---
    embedding_service = EmbeddingService(server.redis_client, llm_client, embedding_config)
    servicer = EmbeddingServicer(embedding_service)
    # TODO: Register when proto generation is complete:
    # from rag_common.generated.services import embedding_pb2_grpc
    # embedding_pb2_grpc.add_EmbeddingAgentServicer_to_server(servicer, server._server)
    logger.info("gRPC servicer created for %s (awaiting proto registration)", AgentRole.EMBEDDING)

    # --- Stream worker ---
    worker = EmbeddingWorker(
        redis_client=server.redis_client,
        llm_client=llm_client,
        config=embedding_config,
        consumer_id=server.agent_id,
    )

    worker_task = asyncio.create_task(worker.start())

    logger.info("Embedding agent running on port %d", embedding_config.grpc_port)

    try:
        await server.wait_for_shutdown()
    finally:
        worker.stop()
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass

    logger.info("Embedding agent shut down")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    asyncio.run(main())
