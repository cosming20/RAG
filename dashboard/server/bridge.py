"""WebSocket bridge — reads Redis agent state and streams to pixel-agents frontend.

Maps RAG swarm agent lifecycle events to the pixel-agents message protocol:
- Agent registered → agentCreated
- Agent heartbeat → agentStatus (idle/typing/walking)
- Task claimed → agentToolStart
- Task completed → agentToolDone
- Task failed → agentToolPermission (shows error bubble)
- Agent deregistered → agentClosed

Run: python dashboard/server/bridge.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any

import redis.asyncio as aioredis
from aiohttp import web

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("rag-dashboard")

REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "2050"))

# Agent role → pixel character palette (0-5) + seat assignment
AGENT_PALETTE = {
    "router": {"palette": 0, "seat": 0, "label": "Router"},
    "ingestion": {"palette": 1, "seat": 1, "label": "Ingestion"},
    "chunking": {"palette": 2, "seat": 2, "label": "Chunking"},
    "embedding": {"palette": 3, "seat": 3, "label": "Embedding"},
    "retrieval": {"palette": 4, "seat": 4, "label": "Retrieval"},
    "graph": {"palette": 5, "seat": 5, "label": "Graph"},
    "reranking": {"palette": 0, "seat": 6, "label": "Reranking"},
    "synthesis": {"palette": 1, "seat": 7, "label": "Synthesis"},
    "validation": {"palette": 2, "seat": 8, "label": "Validation"},
}

# Stream names to monitor for activity
STREAMS = [
    "rag:stream:ingest:pending",
    "rag:stream:ingest:parsed",
    "rag:stream:ingest:chunked",
    "rag:stream:ingest:embedded",
    "rag:stream:ingest:stored",
    "rag:stream:query:classified",
    "rag:stream:query:retrieved",
    "rag:stream:query:graph_enriched",
    "rag:stream:query:reranked",
    "rag:stream:query:synthesized",
    "rag:stream:query:validated",
]


class DashboardState:
    """Aggregates Redis state into a dashboard-friendly format."""

    def __init__(self, redis_url: str) -> None:
        self._redis_url = redis_url
        self._redis: aioredis.Redis | None = None
        self._ws_clients: list[web.WebSocketResponse] = []

    async def connect(self) -> None:
        self._redis = aioredis.Redis(
            host=REDIS_HOST, port=REDIS_PORT, decode_responses=True,
        )
        await self._redis.ping()
        logger.info("Connected to Redis at %s:%s", REDIS_HOST, REDIS_PORT)

    async def close(self) -> None:
        if self._redis:
            await self._redis.aclose()

    async def get_agents(self) -> list[dict[str, Any]]:
        """Get all registered agents from Redis."""
        agents = []
        for role, config in AGENT_PALETTE.items():
            role_key = f"rag:agents:by_role:{role}"
            members = await self._redis.smembers(role_key)
            for agent_id in members:
                agent_key = f"rag:agents:{agent_id}"
                data = await self._redis.hgetall(agent_key)
                if data:
                    agents.append({
                        "id": agent_id,
                        "role": role,
                        "status": data.get("status", "unknown"),
                        "grpc_address": data.get("grpc_address", ""),
                        "last_heartbeat": data.get("last_heartbeat", "0"),
                        **config,
                    })
        return agents

    async def get_stream_stats(self) -> dict[str, Any]:
        """Get message counts and consumer group info for all streams."""
        stats = {}
        for stream in STREAMS:
            try:
                length = await self._redis.xlen(stream)
                groups = []
                try:
                    group_info = await self._redis.xinfo_groups(stream)
                    for g in group_info:
                        groups.append({
                            "name": g.get("name", ""),
                            "consumers": g.get("consumers", 0),
                            "pending": g.get("pending", 0),
                            "last_delivered": g.get("last-delivered-id", ""),
                        })
                except Exception:
                    pass
                stats[stream] = {"length": length, "groups": groups}
            except Exception:
                stats[stream] = {"length": 0, "groups": []}
        return stats

    async def get_recent_tasks(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get recent tasks from Redis."""
        tasks = []
        # Scan for task keys
        cursor = 0
        found = 0
        while found < limit:
            cursor, keys = await self._redis.scan(cursor, match="rag:tasks:*", count=50)
            for key in keys:
                if found >= limit:
                    break
                data = await self._redis.hgetall(key)
                if data:
                    task_id = key.replace("rag:tasks:", "")
                    tasks.append({"task_id": task_id, **data})
                    found += 1
            if cursor == 0:
                break
        return sorted(tasks, key=lambda t: t.get("updated_at", "0"), reverse=True)

    async def get_intermediate_results(self, task_id: str) -> dict[str, Any]:
        """Get all intermediate results for a specific task."""
        results = {}
        stages = [
            "raw", "meta", "parsed", "chunks", "embeddings", "classification",
            "retrieved", "graph_context", "reranked", "answer", "validation",
        ]
        for stage in stages:
            key = f"rag:results:{task_id}:{stage}"
            data = await self._redis.get(key)
            if data:
                try:
                    parsed = json.loads(data)
                    # Truncate large values for display
                    if isinstance(parsed, dict):
                        for k, v in parsed.items():
                            if isinstance(v, list) and len(v) > 5:
                                parsed[k] = v[:5] + [f"... ({len(v) - 5} more)"]
                            elif isinstance(v, str) and len(v) > 500:
                                parsed[k] = v[:500] + "..."
                    results[stage] = parsed
                except json.JSONDecodeError:
                    results[stage] = f"(binary, {len(data)} bytes)"
        return results

    async def get_full_state(self) -> dict[str, Any]:
        """Get complete dashboard state."""
        agents = await self.get_agents()
        streams = await self.get_stream_stats()
        tasks = await self.get_recent_tasks()
        return {
            "timestamp": int(time.time() * 1000),
            "agents": agents,
            "streams": streams,
            "tasks": tasks,
        }

    async def broadcast(self, message: dict) -> None:
        """Send message to all connected WebSocket clients."""
        dead = []
        for ws in self._ws_clients:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._ws_clients.remove(ws)


# ── Web Server ────────────────────────────────────────────

async def handle_ws(request: web.Request) -> web.WebSocketResponse:
    """WebSocket endpoint for real-time dashboard updates."""
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    state: DashboardState = request.app["state"]
    state._ws_clients.append(ws)
    logger.info("WebSocket client connected (%d total)", len(state._ws_clients))

    # Send initial state
    full_state = await state.get_full_state()
    await ws.send_json({"type": "fullState", **full_state})

    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                data = json.loads(msg.data)
                if data.get("type") == "getTaskDetails":
                    task_id = data.get("taskId", "")
                    results = await state.get_intermediate_results(task_id)
                    await ws.send_json({"type": "taskDetails", "taskId": task_id, "results": results})
            elif msg.type == web.WSMsgType.ERROR:
                break
    finally:
        state._ws_clients.remove(ws)
        logger.info("WebSocket client disconnected (%d remaining)", len(state._ws_clients))

    return ws


async def handle_index(request: web.Request) -> web.FileResponse:
    """Serve the dashboard HTML."""
    return web.FileResponse(
        os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html"),
    )


async def handle_api_state(request: web.Request) -> web.Response:
    """REST endpoint for full state (fallback if WebSocket not available)."""
    state: DashboardState = request.app["state"]
    full_state = await state.get_full_state()
    return web.json_response(full_state)


async def handle_api_task(request: web.Request) -> web.Response:
    """REST endpoint for task details."""
    task_id = request.match_info["task_id"]
    state: DashboardState = request.app["state"]
    results = await state.get_intermediate_results(task_id)
    return web.json_response({"taskId": task_id, "results": results})


async def poll_loop(app: web.Application) -> None:
    """Background task that polls Redis and broadcasts updates."""
    state: DashboardState = app["state"]
    while True:
        try:
            full_state = await state.get_full_state()
            await state.broadcast({"type": "fullState", **full_state})
        except Exception:
            logger.exception("Poll loop error")
        await asyncio.sleep(2)


async def on_startup(app: web.Application) -> None:
    state = DashboardState(f"redis://{REDIS_HOST}:{REDIS_PORT}")
    await state.connect()
    app["state"] = state
    app["poll_task"] = asyncio.create_task(poll_loop(app))
    logger.info("Dashboard server started")


async def on_cleanup(app: web.Application) -> None:
    app["poll_task"].cancel()
    await app["state"].close()


def create_app() -> web.Application:
    app = web.Application()
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)

    # Routes
    app.router.add_get("/", handle_index)
    app.router.add_get("/ws", handle_ws)
    app.router.add_get("/api/state", handle_api_state)
    app.router.add_get("/api/task/{task_id}", handle_api_task)

    # Static files for frontend assets
    frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
    if os.path.isdir(frontend_dir):
        app.router.add_static("/static/", frontend_dir)

    return app


if __name__ == "__main__":
    port = int(os.environ.get("DASHBOARD_PORT", "2070"))
    logger.info("Starting dashboard on http://localhost:%d", port)
    web.run_app(create_app(), host="0.0.0.0", port=port)
