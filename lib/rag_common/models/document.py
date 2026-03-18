"""Document domain models."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class DocumentFormat(StrEnum):
    PDF = "pdf"
    DOCX = "docx"
    PPTX = "pptx"
    XLSX = "xlsx"
    HTML = "html"
    MARKDOWN = "markdown"
    IMAGE = "image"
    LATEX = "latex"
    CSV = "csv"
    CODE = "code"
    TXT = "txt"


class DocumentMeta(BaseModel):
    document_id: UUID = Field(default_factory=uuid4)
    filename: str
    format: DocumentFormat
    size_bytes: int
    content_hash: str = ""
    metadata: dict[str, str] = Field(default_factory=dict)
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @staticmethod
    def compute_hash(content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()


class Chunk(BaseModel):
    chunk_id: UUID = Field(default_factory=uuid4)
    document_id: UUID
    text: str
    sequence_index: int
    start_offset: int = 0
    end_offset: int = 0
    metadata: dict[str, str] = Field(default_factory=dict)


class EmbeddingVector(BaseModel):
    values: list[float]
    model: str
    dimensions: int


class SparseVector(BaseModel):
    indices: list[int]
    values: list[float]
