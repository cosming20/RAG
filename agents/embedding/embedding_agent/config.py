"""Embedding agent configuration.

Defaults sourced from:
- Cognee: batch_size=100 (cognee/tasks/memify/get_triplet_datapoints.py)
- OpenAI: text-embedding-3-large = 3072 dimensions
- BM25: standard k1=1.2, b=0.75 (Robertson & Walker, 1994)
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict

class EmbeddingConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EMBEDDING_")

    grpc_port: int = 50054
    # Cognee uses batch_size=100 for triplet processing
    batch_size: int = 100
    dense_model: str = "openai/text-embedding-3-large"
    # BM25 built into Qdrant; client-side encoder for query-time
    sparse_encoder: str = "bm25"
    max_retries: int = 3
