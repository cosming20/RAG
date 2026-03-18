"""LLM provider configuration helpers."""

from __future__ import annotations

from rag_common.config import LLMConfig


def get_embedding_model(config: LLMConfig | None = None) -> str:
    return (config or LLMConfig()).embedding_model


def get_default_model(config: LLMConfig | None = None) -> str:
    return (config or LLMConfig()).default_model
