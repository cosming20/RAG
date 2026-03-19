"""Ingestion Agent entry point -- starts gRPC server and stream worker."""

from __future__ import annotations

import asyncio
import logging

from rag_common.config import AgentConfig, ObservabilityConfig, RedisConfig
from rag_common.grpc_utils.server import BaseGrpcServer
from rag_common.models.agent import AgentRole
from rag_common.observability import setup_logging, setup_metrics, setup_tracing

from .config import IngestionConfig
from .grpc_servicer import IngestionServicer
from .service import IngestionService
from .worker import IngestionWorker

logger = logging.getLogger(__name__)


async def serve() -> None:
    """Initialize and run the Ingestion Agent (gRPC + stream worker)."""
    # Load configuration
    agent_config = AgentConfig()
    redis_config = RedisConfig()
    otel_config = ObservabilityConfig(service_name="rag-ingestion")
    ingestion_config = IngestionConfig()

    # Setup observability
    setup_logging(otel_config)
    setup_metrics(otel_config)
    setup_tracing(otel_config)

    # Create base gRPC server (handles Redis, registration, heartbeat)
    server = BaseGrpcServer(
        role=AgentRole.INGESTION,
        agent_config=agent_config,
        redis_config=redis_config,
        otel_config=otel_config,
    )
    await server.start(service_names=[])

    # Build service layer using the server's Redis connection
    redis_client = server.redis_client
    ingestion_service = IngestionService(
        redis_client=redis_client,
        config=ingestion_config,
    )

    # gRPC servicer (ready for proto registration)
    servicer = IngestionServicer(ingestion_service)
    # TODO: Register when proto generation is complete:
    # from rag_common.generated.services import ingestion_pb2_grpc
    # ingestion_pb2_grpc.add_IngestionAgentServicer_to_server(servicer, server._server)
    logger.info("gRPC servicer created for %s (awaiting proto registration)", AgentRole.INGESTION)

    # Build and start stream worker
    worker = IngestionWorker(
        redis_client=redis_client,
        service=ingestion_service,
        config=ingestion_config,
        consumer_id=server.agent_id,
    )

    logger.info("Ingestion Agent starting: port=%d", ingestion_config.grpc_port)

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
        logger.info("Ingestion Agent shut down")


def main() -> None:
    """CLI entry point."""
    asyncio.run(serve())


if __name__ == "__main__":
    main()
