"""Validation agent entry point.

Starts the gRPC health server, connects to Redis, and runs the
ValidationWorker that consumes from ``rag:stream:query:synthesized``
and validates answer quality via hallucination and attribution checks.
"""

from __future__ import annotations

import asyncio
import logging

from rag_common.config import AgentConfig, LLMConfig
from rag_common.grpc_utils.server import BaseGrpcServer
from rag_common.llm.client import LLMClient
from rag_common.models.agent import AgentRole

from agents.validation.validation_agent.config import ValidationConfig
from agents.validation.validation_agent.service import ValidationService
from agents.validation.validation_agent.worker import ValidationWorker

logger = logging.getLogger(__name__)


async def main() -> None:
    """Bootstrap and run the Validation agent."""
    validation_config = ValidationConfig()
    llm_config = LLMConfig()
    agent_config = AgentConfig()

    # --- gRPC server ---
    server = BaseGrpcServer(
        role=AgentRole.VALIDATION,
        agent_config=agent_config,
    )
    await server.start()

    # --- LLM client ---
    llm_client = LLMClient(llm_config)

    # --- Service + Worker ---
    service = ValidationService(
        server.redis_client,
        llm_client,
        validation_config,
    )
    worker = ValidationWorker(
        server.redis_client,
        service,
        validation_config,
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

    logger.info("Validation agent shut down")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    asyncio.run(main())
