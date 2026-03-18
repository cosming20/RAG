"""Shared agent runner -- eliminates boilerplate across agent main.py files."""

from __future__ import annotations

import asyncio
import logging
from typing import Callable

from rag_common.config import AgentConfig, RedisConfig, LLMConfig, ObservabilityConfig
from rag_common.grpc_utils.server import BaseGrpcServer
from rag_common.llm.client import LLMClient
from rag_common.models.agent import AgentRole
from rag_common.observability import setup_logging, setup_metrics, setup_tracing
from rag_common.redis.worker import BaseStreamWorker

logger = logging.getLogger(__name__)


async def run_agent(
    role: AgentRole,
    worker_factory: Callable[[BaseGrpcServer, LLMClient], list[BaseStreamWorker]],
    service_name: str | None = None,
) -> None:
    """Shared entry point for all agents. Handles server lifecycle + workers."""
    agent_config = AgentConfig()
    redis_config = RedisConfig()
    llm_config = LLMConfig()
    otel_config = ObservabilityConfig(service_name=service_name or f"rag-{role}")

    setup_logging(otel_config)
    setup_metrics(otel_config)
    setup_tracing(otel_config)

    llm_client = LLMClient(llm_config)

    server = BaseGrpcServer(
        role=role,
        agent_config=agent_config,
        redis_config=redis_config,
        otel_config=otel_config,
    )
    await server.start(service_names=[])

    workers = worker_factory(server, llm_client)
    worker_tasks = [asyncio.create_task(w.start()) for w in workers]

    logger.info("Agent %s ready with %d workers", role, len(workers))

    try:
        await server.wait_for_shutdown()
    finally:
        for w in workers:
            w.stop()
        for t in worker_tasks:
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass
