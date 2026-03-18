"""Validation agent configuration.

Defaults sourced from:
- OpenAI: gpt-4o for reliable hallucination judgment
- RAG evaluation best practices: 0.7 quality threshold, 1 retry
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class ValidationConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VALIDATION_")

    grpc_port: int = 50059
    # Minimum acceptable quality score (0.0-1.0)
    quality_threshold: float = 0.7
    # Maximum synthesis retries when quality is below threshold
    max_retries: int = 1
    # Strong model required for accurate hallucination judgment
    hallucination_model: str = "openai/gpt-4o"
    # Whether to run attribution verification on citations
    attribution_check_enabled: bool = True
