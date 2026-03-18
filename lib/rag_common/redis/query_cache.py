"""Query result cache using embedding similarity.

Inspired by Cognee's cache pattern (7-day TTL) + Lawren's cache config
(CACHE_DIRECT_HIT_THRESHOLD=0.8, CACHE_MAX_RESULTS=10).
"""

from __future__ import annotations

import json
import logging

import numpy as np

from rag_common.redis.client import RedisClient

logger = logging.getLogger(__name__)

CACHE_KEY_PREFIX = "rag:query_cache"
# Cognee uses 7-day TTL for cache entries
DEFAULT_CACHE_TTL = 604800  # 7 days in seconds


class QueryCache:
    """Caches query results by embedding similarity.

    On query:
    1. Embed the query
    2. Compare against cached query embeddings (cosine similarity)
    3. If similarity > threshold: return cached result
    4. If miss: return None, caller processes normally then calls store()
    """

    def __init__(
        self,
        redis_client: RedisClient,
        similarity_threshold: float = 0.95,
        max_cached: int = 1000,
        ttl: int = DEFAULT_CACHE_TTL,
    ) -> None:
        self._redis = redis_client
        self._threshold = similarity_threshold
        self._max_cached = max_cached
        self._ttl = ttl

    async def lookup(self, query_embedding: list[float]) -> dict | None:
        """Check if a similar query exists in cache. Returns cached result or None."""
        # Get all cached query keys
        cache_index_key = f"{CACHE_KEY_PREFIX}:index"
        cached_keys = await self._redis.client.lrange(cache_index_key, 0, self._max_cached)

        if not cached_keys:
            return None

        query_vec = np.array(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return None
        query_vec = query_vec / query_norm

        best_score = 0.0
        best_key = None

        for key in cached_keys:
            cache_key = key if isinstance(key, str) else key.decode()
            cached_data = await self._redis.client.get(f"{CACHE_KEY_PREFIX}:{cache_key}:embedding")
            if cached_data is None:
                continue
            cached_vec = np.array(json.loads(cached_data), dtype=np.float32)
            cached_norm = np.linalg.norm(cached_vec)
            if cached_norm == 0:
                continue
            cached_vec = cached_vec / cached_norm

            similarity = float(np.dot(query_vec, cached_vec))
            if similarity > best_score:
                best_score = similarity
                best_key = cache_key

        if best_score >= self._threshold and best_key:
            result_data = await self._redis.client.get(f"{CACHE_KEY_PREFIX}:{best_key}:result")
            if result_data:
                logger.info("Cache hit: similarity=%.4f key=%s", best_score, best_key)
                return json.loads(result_data)

        return None

    async def store(
        self,
        cache_key: str,
        query_embedding: list[float],
        result: dict,
    ) -> None:
        """Store a query result in cache."""
        pipe = self._redis.client.pipeline()
        pipe.set(
            f"{CACHE_KEY_PREFIX}:{cache_key}:embedding",
            json.dumps(query_embedding),
            ex=self._ttl,
        )
        pipe.set(
            f"{CACHE_KEY_PREFIX}:{cache_key}:result",
            json.dumps(result, default=str),
            ex=self._ttl,
        )
        pipe.lpush(f"{CACHE_KEY_PREFIX}:index", cache_key)
        pipe.ltrim(f"{CACHE_KEY_PREFIX}:index", 0, self._max_cached - 1)
        await pipe.execute()
        logger.debug("Cached query result: key=%s", cache_key)

    async def invalidate(self, cache_key: str) -> None:
        """Remove a specific cache entry."""
        pipe = self._redis.client.pipeline()
        pipe.delete(f"{CACHE_KEY_PREFIX}:{cache_key}:embedding")
        pipe.delete(f"{CACHE_KEY_PREFIX}:{cache_key}:result")
        pipe.lrem(f"{CACHE_KEY_PREFIX}:index", 0, cache_key)
        await pipe.execute()

    async def clear(self) -> None:
        """Clear all cached queries."""
        index_key = f"{CACHE_KEY_PREFIX}:index"
        cached_keys = await self._redis.client.lrange(index_key, 0, -1)
        if cached_keys:
            pipe = self._redis.client.pipeline()
            for key in cached_keys:
                cache_key = key if isinstance(key, str) else key.decode()
                pipe.delete(f"{CACHE_KEY_PREFIX}:{cache_key}:embedding")
                pipe.delete(f"{CACHE_KEY_PREFIX}:{cache_key}:result")
            pipe.delete(index_key)
            await pipe.execute()
