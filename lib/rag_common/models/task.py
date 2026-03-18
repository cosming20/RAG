"""Task lifecycle models."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class TaskStatus(StrEnum):
    PENDING = "pending"
    CLAIMED = "claimed"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    POISON = "poison"


class Task(BaseModel):
    task_id: UUID = Field(default_factory=uuid4)
    parent_task_id: UUID | None = None
    agent_id: str = ""
    status: TaskStatus = TaskStatus.PENDING
    attempt: int = 0
    max_attempts: int = 3
    pipeline_depth: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    deadline: datetime | None = None
    trace_id: str = ""
    span_id: str = ""
    payload_ref: str = ""
    error: str | None = None
    source_pipeline: str = ""
    source_agent: str = ""


class TaskResult(BaseModel):
    task_id: UUID
    success: bool
    output_ref: str = ""
    error_message: str | None = None
    processing_time_ms: float = 0.0
