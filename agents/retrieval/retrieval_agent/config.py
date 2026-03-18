"""Retrieval agent configuration.

Defaults sourced from:
- Lawren API: search_ef=128, qdrant_timeout=30s (config.rs)
- OpenAI: text-embedding-3-large = 3072 dimensions
- Cognee: batch operations capped at 100
- Qdrant: Cosine distance for normalized embeddings
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class RetrievalConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RETRIEVAL_")

    grpc_port: int = 50055
    collection_name: str = "rag_chunks"
    # OpenAI text-embedding-3-large output dimensions
    vector_size: int = 3072
    sparse_vector_name: str = "sparse"
    dense_vector_name: str = "dense"
    # Cognee batch size for storage operations
    batch_upsert_size: int = 100
    # Lawren API: search_ef=128 for HNSW index (higher = more accurate, slower)
    search_ef: int = 128
    # Qdrant timeout from Lawren config
    qdrant_timeout_secs: float = 30.0
