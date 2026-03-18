"""Centralized configuration via Pydantic Settings. All config from environment variables.

Value sources documented per field. Key references:
- Cognee: cognee/infrastructure/data/chunking/config.py, cognee/cli/commands/config_command.py
- Lawren API: src/lawren_api/config.rs (Qdrant, Neo4j, LLM settings)
- Qdrant defaults: lib/segment/src/common/defaults.rs
- OpenAI API: model specs for text-embedding-3-large, gpt-4o
- Docling: docling/datamodel/vlm_model_specs.py
"""

from __future__ import annotations

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class RedisConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="REDIS_")

    host: str = "localhost"
    port: int = 6379
    password: SecretStr | None = None
    db: int = 0
    # Redis default socket timeout
    socket_timeout: float = 30.0
    # Lawren API: connection pool size ~50 for production gRPC service
    max_connections: int = 50
    decode_responses: bool = True

    @property
    def url(self) -> str:
        if self.password:
            auth = f":{self.password.get_secret_value()}@"
        else:
            auth = ""
        return f"redis://{auth}{self.host}:{self.port}/{self.db}"


class QdrantConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="QDRANT_")

    host: str = "localhost"
    grpc_port: int = 6334
    rest_port: int = 6333
    api_key: SecretStr | None = None
    collection_name: str = "rag_chunks"
    prefer_grpc: bool = True
    # Lawren API: graphrag_qdrant_timeout_secs = 30
    timeout: float = 30.0
    # Lawren API: graphrag_qdrant_search_ef = 128
    # Qdrant default ef=128; higher improves recall at cost of latency
    search_ef: int = 128


class Neo4jConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NEO4J_")

    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: SecretStr = SecretStr("")
    database: str = "neo4j"
    # Neo4j driver default pool size
    max_connection_pool_size: int = 50


class LLMConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LLM_")

    # Cognee default: openai/gpt-4o-mini; we use gpt-4o for better extraction
    default_model: str = "openai/gpt-4o"
    # OpenAI: 3072-dim, best quality/cost for RAG
    embedding_model: str = "openai/text-embedding-3-large"
    embedding_dimensions: int = 3072
    api_key: SecretStr = SecretStr("")
    # Cognee/Lawren: 3 retries standard
    max_retries: int = 3
    # Lawren API: LLM timeout 60s
    timeout_seconds: float = 60.0
    # Docling VLM: max_tokens=4096; GPT-4o supports up to 16K output
    max_tokens: int = 4096
    # Low temperature for factual extraction
    temperature: float = 0.1


class JinaConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="JINA_")

    api_key: SecretStr = SecretStr("")
    # Jina v2: multilingual cross-encoder, 8192 token context
    model: str = "jina-reranker-v2-base-multilingual"
    api_url: str = "https://api.jina.ai/v1/rerank"
    timeout_seconds: float = 30.0


class AgentConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AGENT_")

    grpc_port: int = 50051
    # Matches typical gRPC server thread pool for async Python
    grpc_max_workers: int = 10
    # Standard heartbeat: 5s interval, 30s TTL (6 missed beats = dead)
    heartbeat_interval_seconds: float = 5.0
    heartbeat_ttl_seconds: float = 30.0
    # Lawren API: request timeout 300s for complex document processing
    task_timeout_seconds: float = 300.0
    max_task_retries: int = 3
    # Swarm loop prevention: max 10 pipeline hops
    max_pipeline_depth: int = 10
    # 90s grace for in-flight LLM calls (GPT-4o can take 30-60s)
    graceful_shutdown_seconds: float = 90.0
    # Lawren API: 50MB max document size
    max_document_size_bytes: int = 50 * 1024 * 1024


class CacheConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CACHE_")

    enabled: bool = False
    similarity_threshold: float = 0.95
    max_cached_queries: int = 1000
    ttl_seconds: int = 604800  # 7 days (Cognee pattern)


class ObservabilityConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OTEL_")

    service_name: str = "rag-agent"
    # OTLP gRPC default port
    exporter_endpoint: str = "http://localhost:4317"
    log_level: str = "INFO"
    log_format: str = "json"
    # Prometheus default scrape port
    metrics_port: int = 9090
    enable_tracing: bool = True
    enable_metrics: bool = True


from functools import lru_cache


@lru_cache
def get_redis_config() -> RedisConfig:
    return RedisConfig()


@lru_cache
def get_qdrant_config() -> QdrantConfig:
    return QdrantConfig()


@lru_cache
def get_neo4j_config() -> Neo4jConfig:
    return Neo4jConfig()


@lru_cache
def get_llm_config() -> LLMConfig:
    return LLMConfig()


@lru_cache
def get_jina_config() -> JinaConfig:
    return JinaConfig()


@lru_cache
def get_agent_config() -> AgentConfig:
    return AgentConfig()


@lru_cache
def get_cache_config() -> CacheConfig:
    return CacheConfig()


@lru_cache
def get_otel_config() -> ObservabilityConfig:
    return ObservabilityConfig()
