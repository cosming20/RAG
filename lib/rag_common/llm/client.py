"""LiteLLM wrapper for completions and embeddings with retry and rate limiting."""

from __future__ import annotations

import logging
from typing import Any

import litellm
from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_delay,
    wait_exponential_jitter,
)

from rag_common.config import LLMConfig
from rag_common.llm.rate_limiter import MovingWindowRateLimiter, RateLimitConfig

logger = logging.getLogger(__name__)


class LLMClient:
    """Multi-provider LLM client via LiteLLM.

    Supports completions (chat) and embeddings.
    Handles retry with exponential backoff, context window overflow (split+pool).
    Inspired by Cognee's LiteLLMEmbeddingEngine.
    """

    def __init__(self, config: LLMConfig | None = None) -> None:
        self._config = config or LLMConfig()
        api_key = self._config.api_key.get_secret_value()
        if api_key:
            litellm.api_key = api_key
        litellm.set_verbose = False

        # Rate limiter (disabled by default, enable via LLM_RATE_LIMIT_ENABLED=true)
        rl_config = RateLimitConfig()
        self._rate_limiter: MovingWindowRateLimiter | None = None
        if rl_config.enabled:
            self._rate_limiter = MovingWindowRateLimiter(
                max_requests=rl_config.requests,
                window_seconds=rl_config.interval,
            )
            logger.info(
                "Rate limiter enabled: %d requests / %.0fs window",
                rl_config.requests,
                rl_config.interval,
            )

    @retry(
        stop=stop_after_delay(128),
        wait=wait_exponential_jitter(initial=2, max=128, jitter=2),
        retry=retry_if_not_exception_type(litellm.NotFoundError),
        reraise=True,
    )
    async def completion(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Any:
        if self._rate_limiter:
            await self._rate_limiter.acquire()
        response = await litellm.acompletion(
            model=model or self._config.default_model,
            messages=messages,
            max_tokens=max_tokens or self._config.max_tokens,
            temperature=temperature if temperature is not None else self._config.temperature,
            stream=stream,
            timeout=self._config.timeout_seconds,
            **kwargs,
        )
        return response

    @retry(
        stop=stop_after_delay(128),
        wait=wait_exponential_jitter(initial=2, max=128, jitter=2),
        retry=retry_if_not_exception_type(litellm.NotFoundError),
        reraise=True,
    )
    async def embed(
        self,
        texts: list[str],
        *,
        model: str | None = None,
    ) -> list[list[float]]:
        if self._rate_limiter:
            await self._rate_limiter.acquire()
        response = await litellm.aembedding(
            model=model or self._config.embedding_model,
            input=texts,
            timeout=self._config.timeout_seconds,
        )
        return [item["embedding"] for item in response.data]

    async def embed_single(self, text: str, *, model: str | None = None) -> list[float]:
        results = await self.embed([text], model=model)
        return results[0]

    async def embed_batch(
        self,
        texts: list[str],
        *,
        model: str | None = None,
        batch_size: int = 64,
    ) -> list[list[float]]:
        """Embed texts in batches. Handles context window overflow by splitting."""
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                embeddings = await self.embed(batch, model=model)
                all_embeddings.extend(embeddings)
            except Exception as e:
                if "context_length_exceeded" in str(e).lower():
                    logger.warning("Context window overflow, splitting batch at index %d", i)
                    for text in batch:
                        try:
                            emb = await self.embed([text], model=model)
                            all_embeddings.extend(emb)
                        except Exception:
                            logger.error("Failed to embed text (len=%d), using zero vector", len(text))
                            dims = self._config.embedding_dimensions
                            all_embeddings.append([0.0] * dims)
                else:
                    raise
        return all_embeddings
