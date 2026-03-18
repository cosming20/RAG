"""Chunking Agent configuration.

Defaults sourced from:
- Cognee: chunk_size=1500, chunk_overlap=10 (cognee/infrastructure/data/chunking/config.py)
- Cognee LangchainChunker: chunk_size=1024, chunk_overlap=10
- Docling: paragraph-based chunking with heading awareness
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class ChunkingConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CHUNKING_")

    grpc_port: int = 50053

    # Cognee production default: 1500 chars for paragraph strategy
    default_chunk_size: int = 1500
    # Minimum viable chunk — below this, merge with adjacent
    min_chunk_size: int = 64
    # Cognee: overlap=10 chars. For token-based, ~24 tokens ≈ 10% of 256-token chunks
    chunk_overlap: int = 24
    # LLM normalization for OCR cleanup
    normalize_enabled: bool = True
    # Document-level summary generation
    summarize_enabled: bool = True
    # Cognee processes triplets in batches of 100; normalize is lighter
    normalize_batch_size: int = 20

    # Stream config
    consumer_group: str = "chunking-agents"
    input_stream: str = "rag:stream:ingest:parsed"
    output_stream: str = "rag:stream:ingest:chunked"
    dlq_stream: str = "rag:stream:chunking:dlq"
    batch_size: int = 1
    block_ms: int = 5000
