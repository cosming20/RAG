"""Base gRPC server with health check, reflection, and observability."""

from __future__ import annotations

import asyncio
import logging
import signal
from concurrent import futures
from typing import Any

import grpc
from grpc_health.v1 import health, health_pb2, health_pb2_grpc
from grpc_reflection.v1alpha import reflection

from rag_common.config import AgentConfig, ObservabilityConfig, RedisConfig
from rag_common.grpc_utils.interceptors import LoggingInterceptor, MetricsInterceptor
from rag_common.models.agent import AgentInfo, AgentRole, AgentStatus
from rag_common.redis.client import RedisClient
from rag_common.redis.registry import AgentRegistry

logger = logging.getLogger(__name__)


class BaseGrpcServer:
    """Base class for all agent gRPC servers.

    Handles: server creation, health checks, reflection, agent registration,
    heartbeat, graceful shutdown.
    """

    def __init__(
        self,
        role: AgentRole,
        agent_config: AgentConfig | None = None,
        redis_config: RedisConfig | None = None,
        otel_config: ObservabilityConfig | None = None,
    ) -> None:
        self._role = role
        self._agent_config = agent_config or AgentConfig()
        self._redis_config = redis_config or RedisConfig()
        self._otel_config = otel_config or ObservabilityConfig()
        self._server: grpc.aio.Server | None = None
        self._redis_client: RedisClient | None = None
        self._registry: AgentRegistry | None = None
        self._agent_info: AgentInfo | None = None
        self._shutdown_event = asyncio.Event()

    @property
    def redis_client(self) -> RedisClient:
        if self._redis_client is None:
            msg = "Server not started. Call start() first."
            raise RuntimeError(msg)
        return self._redis_client

    @property
    def agent_id(self) -> str:
        import os
        hostname = os.environ.get("HOSTNAME", os.environ.get("POD_NAME", "local"))
        pid = os.getpid()
        return f"{self._role}-{hostname}-{pid}"

    async def start(self, service_names: list[str] | None = None) -> None:
        # Connect to Redis
        self._redis_client = RedisClient(self._redis_config)
        await self._redis_client.connect()
        self._registry = AgentRegistry(self._redis_client)

        # Create gRPC server
        interceptors = [LoggingInterceptor(), MetricsInterceptor()]
        self._server = grpc.aio.server(
            futures.ThreadPoolExecutor(max_workers=self._agent_config.grpc_max_workers),
            interceptors=interceptors,
        )

        # Add health service
        health_servicer = health.aio.HealthServicer()
        health_pb2_grpc.add_HealthServicer_to_server(health_servicer, self._server)
        await health_servicer.set(
            "", health_pb2.HealthCheckResponse.SERVING,
        )

        # Add reflection
        all_services = [
            health_pb2.DESCRIPTOR.services_by_name["Health"].full_name,
            *(service_names or []),
            reflection.SERVICE_NAME,
        ]
        reflection.enable_server_reflection(all_services, self._server)

        # Bind port
        port = self._agent_config.grpc_port
        self._server.add_insecure_port(f"[::]:{port}")
        await self._server.start()

        # Register agent
        self._agent_info = AgentInfo(
            agent_id=self.agent_id,
            role=self._role,
            grpc_address=f"0.0.0.0:{port}",
            status=AgentStatus.SERVING,
        )
        await self._registry.register(self._agent_info)
        await self._registry.start_heartbeat_loop(
            self.agent_id,
            interval_seconds=self._agent_config.heartbeat_interval_seconds,
            ttl_seconds=int(self._agent_config.heartbeat_ttl_seconds),
        )

        logger.info(
            "Agent %s (%s) serving on port %s",
            self.agent_id,
            self._role,
            port,
        )

        # Handle shutdown signals
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(self.shutdown()))

    async def shutdown(self) -> None:
        logger.info("Shutting down agent %s...", self.agent_id)

        # Stop heartbeat
        if self._registry:
            await self._registry.stop_heartbeat_loop()

        # Deregister
        if self._registry and self._agent_info:
            await self._registry.deregister(self._agent_info.agent_id, self._role)

        # Graceful stop
        if self._server:
            grace = self._agent_config.graceful_shutdown_seconds
            await self._server.stop(grace)

        # Close Redis
        if self._redis_client:
            await self._redis_client.close()

        self._shutdown_event.set()
        logger.info("Agent %s shut down cleanly", self.agent_id)

    async def wait_for_shutdown(self) -> None:
        await self._shutdown_event.wait()

    def register_service(self, add_servicer_fn: Any, servicer: Any) -> None:
        if self._server is None:
            msg = "Server not started. Call start() first."
            raise RuntimeError(msg)
        add_servicer_fn(servicer, self._server)
