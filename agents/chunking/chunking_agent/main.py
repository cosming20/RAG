"""Chunking Agent entry point -- starts gRPC server and stream worker."""

from __future__ import annotations

import asyncio
import logging

from rag_common.config import AgentConfig, LLMConfig, ObservabilityConfig, RedisConfig
from rag_common.grpc_utils.server import BaseGrpcServer
from rag_common.llm.client import LLMClient
from rag_common.models.agent import AgentRole
from rag_common.observability import setup_logging, setup_metrics, setup_tracing

from .config import ChunkingConfig
from .grpc_servicer import ChunkingServicer
from .service import ChunkingService
from .worker import ChunkingWorker

logger = logging.getLogger(__name__)


async def serve() -> None:
    """Initialize and run the Chunking Agent (gRPC + stream worker)."""
    # Load configuration
    agent_config = AgentConfig()
    redis_config = RedisConfig()
    llm_config = LLMConfig()
    otel_config = ObservabilityConfig(service_name="rag-chunking")
    chunking_config = ChunkingConfig()

    # Setup observability
    setup_logging(otel_config)
    setup_metrics(otel_config)
    setup_tracing(otel_config)

    # Create LLM client (used by normalizer and summarizer)
    llm_client = LLMClient(llm_config)

    # Create base gRPC server (handles Redis, registration, heartbeat)
    server = BaseGrpcServer(
        role=AgentRole.CHUNKING,
        agent_config=agent_config,
        redis_config=redis_config,
        otel_config=otel_config,
    )
    await server.start(service_names=[])

    # Build service layer using the server's Redis connection
    redis_client = server.redis_client
    chunking_service = ChunkingService(
        redis_client=redis_client,
        llm_client=llm_client,
        config=chunking_config,
    )

    # gRPC servicer (ready for proto registration)
    servicer = ChunkingServicer(chunking_service)
    # TODO: Register when proto generation is complete:
    # from rag_common.generated.services import chunking_pb2_grpc
    # chunking_pb2_grpc.add_ChunkingAgentServicer_to_server(servicer, server._server)
    logger.info("gRPC servicer created for %s (awaiting proto registration)", AgentRole.CHUNKING)

    # Build and start stream worker
    worker = ChunkingWorker(
        redis_client=redis_client,
        service=chunking_service,
        config=chunking_config,
        consumer_id=server.agent_id,
    )

    logger.info("Chunking Agent starting: port=%d", chunking_config.grpc_port)

    # Run worker and gRPC server concurrently
    worker_task = asyncio.create_task(worker.run())
    try:
        await server.wait_for_shutdown()
    finally:
        worker.stop()
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass
        logger.info("Chunking Agent shut down")


def main() -> None:
    """CLI entry point."""
    asyncio.run(serve())


if __name__ == "__main__":
    main()
