#!/usr/bin/env python3
"""Health check script — verifies all agents and infrastructure are reachable."""

from __future__ import annotations

import asyncio
import sys

import grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc


AGENTS = {
    "router": "localhost:50051",
    "ingestion": "localhost:50052",
    "chunking": "localhost:50053",
    "embedding": "localhost:50054",
    "retrieval": "localhost:50055",
    "graph": "localhost:50056",
    "reranking": "localhost:50057",
    "synthesis": "localhost:50058",
    "validation": "localhost:50059",
}


async def check_agent(name: str, address: str) -> bool:
    try:
        async with grpc.aio.insecure_channel(address) as channel:
            stub = health_pb2_grpc.HealthStub(channel)
            response = await stub.Check(
                health_pb2.HealthCheckRequest(service=""),
                timeout=5,
            )
            status = health_pb2.HealthCheckResponse.ServingStatus.Name(response.status)
            print(f"  {name:12s} @ {address:20s} -> {status}")
            return response.status == health_pb2.HealthCheckResponse.SERVING
    except grpc.aio.AioRpcError as e:
        print(f"  {name:12s} @ {address:20s} -> UNREACHABLE ({e.code().name})")
        return False


async def check_redis() -> bool:
    try:
        import redis.asyncio as aioredis
        r = aioredis.Redis(host="localhost", port=6379)
        await r.ping()
        await r.aclose()
        print("  redis        @ localhost:6379       -> OK")
        return True
    except Exception as e:
        print(f"  redis        @ localhost:6379       -> UNREACHABLE ({e})")
        return False


async def check_qdrant() -> bool:
    try:
        from qdrant_client import AsyncQdrantClient
        client = AsyncQdrantClient(host="localhost", grpc_port=6334, prefer_grpc=True)
        await client.get_collections()
        await client.close()
        print("  qdrant       @ localhost:6334       -> OK")
        return True
    except Exception as e:
        print(f"  qdrant       @ localhost:6334       -> UNREACHABLE ({e})")
        return False


async def main() -> None:
    print("RAG Swarm Health Check")
    print("=" * 60)

    print("\nInfrastructure:")
    infra_ok = all(await asyncio.gather(check_redis(), check_qdrant()))

    print("\nAgents:")
    agent_results = await asyncio.gather(
        *(check_agent(name, addr) for name, addr in AGENTS.items())
    )
    agents_ok = all(agent_results)

    healthy = sum(agent_results)
    total = len(agent_results)
    print(f"\nSummary: {healthy}/{total} agents healthy, infra {'OK' if infra_ok else 'DEGRADED'}")

    sys.exit(0 if (infra_ok and agents_ok) else 1)


if __name__ == "__main__":
    asyncio.run(main())
