"""Router Agent entry point — starts the gRPC server with all services."""

from __future__ import annotations

import asyncio
import logging

from rag_common.config import AgentConfig, LLMConfig, ObservabilityConfig, RedisConfig
from rag_common.grpc_utils.server import BaseGrpcServer
from rag_common.llm.client import LLMClient
from rag_common.models.agent import AgentRole
from rag_common.observability import setup_logging, setup_metrics, setup_tracing

from .config import RouterConfig
from .service import RouterService

logger = logging.getLogger(__name__)


async def serve() -> None:
    """Initialize and run the Router Agent."""
    # Load config
    agent_config = AgentConfig()
    redis_config = RedisConfig()
    llm_config = LLMConfig()
    otel_config = ObservabilityConfig(service_name="rag-router")
    router_config = RouterConfig()

    # Setup observability
    setup_logging(otel_config)
    setup_metrics(otel_config)
    setup_tracing(otel_config)

    # Create LLM client
    llm_client = LLMClient(llm_config)

    # Create base server
    server = BaseGrpcServer(
        role=AgentRole.ROUTER,
        agent_config=agent_config,
        redis_config=redis_config,
        otel_config=otel_config,
    )

    # Start server (registers agent, starts heartbeat)
    await server.start(service_names=[])

    # Create the service
    _router_service = RouterService(
        redis_client=server.redis_client,
        llm_client=llm_client,
        config=router_config,
    )

    # TODO: Register the gRPC servicer once proto generation is complete
    # from rag_common.generated.services import router_pb2_grpc
    # router_pb2_grpc.add_RouterAgentServicer_to_server(servicer, server._server)

    logger.info(
        "Router Agent ready — gRPC port=%d, role=%s",
        agent_config.grpc_port,
        AgentRole.ROUTER,
    )

    # Wait for shutdown
    await server.wait_for_shutdown()


def main() -> None:
    asyncio.run(serve())


if __name__ == "__main__":
    main()
