"""Synthesis agent configuration.

Defaults sourced from:
- OpenAI: gpt-4o for high-quality answer generation
- RAG best practices: 12K token context budget, 15 max chunks
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict

# Default system prompt for RAG answer generation
DEFAULT_SYSTEM_PROMPT = (
    "You are a precise research assistant. Answer questions based ONLY on the "
    "provided context. Cite sources using [N] notation matching the reference "
    "numbers. If the context doesn't contain enough information, say so clearly. "
    "Do not fabricate information."
)


class SynthesisConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SYNTHESIS_")

    grpc_port: int = 50058
    # OpenAI gpt-4o: strong generation model for accurate, cited answers
    synthesis_model: str = "openai/gpt-4o"
    # Maximum number of context chunks to include in the LLM prompt
    max_context_chunks: int = 15
    # Approximate token budget for context (chars * 0.25 heuristic)
    max_context_tokens: int = 12000
    # Configurable system prompt for RAG generation
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
