"""Synthesis agent entry point.

Starts the gRPC health server, connects to Redis, and runs the
SynthesisWorker that consumes from ``rag:stream:query:reranked``
and generates cited LLM answers.
"""

from __future__ import annotations

import asyncio
import logging

from rag_common.config import AgentConfig, LLMConfig
from rag_common.grpc_utils.server import BaseGrpcServer
from rag_common.llm.client import LLMClient
from rag_common.models.agent import AgentRole

from agents.synthesis.synthesis_agent.config import SynthesisConfig
from agents.synthesis.synthesis_agent.grpc_servicer import SynthesisServicer
from agents.synthesis.synthesis_agent.service import SynthesisService
from agents.synthesis.synthesis_agent.worker import SynthesisWorker

logger = logging.getLogger(__name__)


async def main() -> None:
    """Bootstrap and run the Synthesis agent."""
    synthesis_config = SynthesisConfig()
    llm_config = LLMConfig()
    agent_config = AgentConfig()

    # --- gRPC server ---
    server = BaseGrpcServer(
        role=AgentRole.SYNTHESIS,
        agent_config=agent_config,
    )
    await server.start()

    # --- LLM client ---
    llm_client = LLMClient(llm_config)

    # --- Service ---
    service = SynthesisService(
        server.redis_client,
        llm_client,
        synthesis_config,
    )

    # --- gRPC servicer (ready for proto registration) ---
    servicer = SynthesisServicer(service)
    # TODO: Register when proto generation is complete:
    # from rag_common.generated.services import synthesis_pb2_grpc
    # synthesis_pb2_grpc.add_SynthesisAgentServicer_to_server(servicer, server._server)
    logger.info("gRPC servicer created for %s (awaiting proto registration)", AgentRole.SYNTHESIS)

    # --- Worker ---
    worker = SynthesisWorker(
        server.redis_client,
        service,
        consumer_id=server.agent_id,
        agent_config=agent_config,
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

    logger.info("Synthesis agent shut down")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    asyncio.run(main())
