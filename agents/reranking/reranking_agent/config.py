"""Reranking agent configuration.

Defaults sourced from:
- Jina: reranker-v2-base-multilingual model (8192 token context, multilingual)
- Lawren API: reference_resolver max_depth=2, max_references=10 (reference_resolver.rs)
- OpenAI: gpt-4o-mini for lightweight relevance filtering
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class RerankingConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RERANKING_")

    grpc_port: int = 50057
    # Jina reranker-v2: multilingual cross-encoder, 8192 token context
    jina_model: str = "jina-reranker-v2-base-multilingual"
    # Return top N chunks after reranking
    top_n: int = 10
    # Jina score threshold: chunks below this are trimmed
    relevance_threshold: float = 0.3
    # Lawren API: recursive reference resolution depth (reference_resolver.rs)
    max_reference_depth: int = 2
    # Lawren API: cap total referenced chunks to prevent explosion
    max_references: int = 10
    # Lightweight model for LLM-based relevance filtering
    trimmer_model: str = "openai/gpt-4o-mini"
    # Seconds to wait for graph context before proceeding without it
    graph_context_wait_seconds: float = 5.0
    # Number of chunks per LLM trimming batch
    trimmer_batch_size: int = 5
