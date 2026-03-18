"""Query and response domain models."""

from __future__ import annotations

from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, Field


class QueryIntent(StrEnum):
    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    COMPARATIVE = "comparative"
    PROCEDURAL = "procedural"
    EXPLORATORY = "exploratory"


class Citation(BaseModel):
    document_id: UUID
    filename: str = ""
    page_number: int = 0
    section: str = ""
    quoted_text: str = ""


class RetrievedChunk(BaseModel):
    chunk_id: UUID
    document_id: UUID
    text: str
    relevance_score: float
    source_description: str = ""
    metadata: dict[str, str] = Field(default_factory=dict)


class Answer(BaseModel):
    text: str
    citations: list[Citation] = Field(default_factory=list)
    confidence_score: float = 0.0
    metadata: dict[str, str] = Field(default_factory=dict)


class QueryResult(BaseModel):
    answer: Answer
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    detected_intent: QueryIntent = QueryIntent.FACTUAL
    processing_time_ms: float = 0.0
