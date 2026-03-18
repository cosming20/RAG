"""Agent lifecycle models."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, Field


class AgentRole(StrEnum):
    ROUTER = "router"
    INGESTION = "ingestion"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    RETRIEVAL = "retrieval"
    GRAPH = "graph"
    RERANKING = "reranking"
    SYNTHESIS = "synthesis"
    VALIDATION = "validation"


class AgentStatus(StrEnum):
    STARTING = "starting"
    SERVING = "serving"
    DRAINING = "draining"
    STOPPED = "stopped"


class AgentInfo(BaseModel):
    agent_id: str
    role: AgentRole
    grpc_address: str
    status: AgentStatus = AgentStatus.STARTING
    registered_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_heartbeat: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    capabilities: dict[str, str] = Field(default_factory=dict)
