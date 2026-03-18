"""Router Agent configuration."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class RouterConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ROUTER_")

    grpc_port: int = 50051
    default_max_results: int = 10
    default_timeout_ms: int = 30000
    classifier_model: str = "openai/gpt-4o-mini"
    max_document_size_bytes: int = 50 * 1024 * 1024  # 50MB
