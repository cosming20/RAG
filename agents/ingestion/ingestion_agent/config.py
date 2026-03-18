"""Ingestion Agent configuration."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict

from rag_common.models.document import DocumentFormat


class IngestionConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="INGESTION_")

    grpc_port: int = 50052
    max_document_size_bytes: int = 50 * 1024 * 1024  # 50MB
    supported_formats: list[str] = [
        DocumentFormat.PDF,
        DocumentFormat.DOCX,
        DocumentFormat.PPTX,
        DocumentFormat.XLSX,
        DocumentFormat.HTML,
        DocumentFormat.MARKDOWN,
        DocumentFormat.IMAGE,
        DocumentFormat.LATEX,
        DocumentFormat.CSV,
        DocumentFormat.CODE,
        DocumentFormat.TXT,
    ]
    ocr_enabled: bool = False
    consumer_group: str = "ingestion-agents"
    input_stream: str = "rag:stream:ingest:pending"
    output_stream: str = "rag:stream:ingest:parsed"
    dlq_stream: str = "rag:stream:ingest:dlq"
    batch_size: int = 1
    block_ms: int = 5000
