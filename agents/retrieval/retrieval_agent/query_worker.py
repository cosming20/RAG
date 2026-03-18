"""Retrieval query worker — performs hybrid search over Qdrant for incoming queries.

Consumes from ``rag:stream:query:classified`` (fan-out with the Graph query worker),
embeds the query text (dense + sparse), runs hybrid search, and stores retrieved
chunks in the state store for downstream synthesis.
"""

from __future__ import annotations

import logging
from typing import Any

from agents.embedding.embedding_agent.sparse import BM25Encoder
from agents.retrieval.retrieval_agent.config import RetrievalConfig
from agents.retrieval.retrieval_agent.pool import QdrantPool
from agents.retrieval.retrieval_agent.searcher import QdrantSearcher
from rag_common.config import AgentConfig
from rag_common.llm.client import LLMClient
from rag_common.redis.client import RedisClient
from rag_common.redis.streams import StreamMessage
from rag_common.redis.worker import BaseStreamWorker

logger = logging.getLogger(__name__)

INPUT_STREAM = "rag:stream:query:classified"
OUTPUT_STREAM = "rag:stream:query:retrieved"
DLQ_STREAM = "rag:stream:retrieval:dlq"
CONSUMER_GROUP = "retrieval-query-agents"

# Default number of results to retrieve when not specified in classification
_DEFAULT_SEARCH_LIMIT = 10


class RetrievalQueryWorker(BaseStreamWorker):
    """Stream worker that performs hybrid search for classified queries.

    Pipeline:
        1. Read classification from StateStore (``rag:results:{task_id}:classification``)
        2. Embed the query text (dense via LLMClient, sparse via BM25Encoder)
        3. Execute hybrid search via QdrantSearcher
        4. Store results at ``rag:results:{task_id}:retrieved``
        5. Forward envelope to ``rag:stream:query:retrieved``
    """

    def __init__(
        self,
        redis_client: RedisClient,
        searcher: QdrantSearcher,
        llm_client: LLMClient,
        retrieval_config: RetrievalConfig,
        *,
        consumer_id: str,
        agent_config: AgentConfig | None = None,
    ) -> None:
        super().__init__(
            redis_client,
            input_stream=INPUT_STREAM,
            output_stream=OUTPUT_STREAM,
            dlq_stream=DLQ_STREAM,
            consumer_group=CONSUMER_GROUP,
            consumer_id=consumer_id,
            agent_config=agent_config,
        )
        self._searcher = searcher
        self._llm = llm_client
        self._bm25 = BM25Encoder()
        self._retrieval_config = retrieval_config

    async def _process_message(self, msg: StreamMessage) -> Any:
        """Embed the query, run hybrid search, store results.

        Returns:
            Dict with ``output_ref`` pointing to the state store key.
        """
        task_id = msg.task_id
        logger.info("RetrievalQueryWorker processing task_id=%s", task_id)

        # 1. Read classification
        classification = await self._state.get_result(task_id, "classification")
        if classification is None:
            raise ValueError(
                f"task_id={task_id}: no classification found in state store"
            )

        query_text: str = classification.get("query_text", "")
        if not query_text:
            raise ValueError(
                f"task_id={task_id}: classification missing query_text"
            )

        search_limit = classification.get("search_limit", _DEFAULT_SEARCH_LIMIT)
        fusion_method = classification.get("fusion", "rrf")
        filters = classification.get("filters")

        # 2. Embed query: dense + sparse
        dense_vector = await self._llm.embed_single(query_text)
        sparse_entry = self._bm25.encode_query(query_text)
        sparse_vector = {
            "indices": sparse_entry["indices"],
            "values": sparse_entry["values"],
        }

        # 3. Hybrid search
        results = await self._searcher.hybrid_search(
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            limit=search_limit,
            fusion=fusion_method,
            filters=filters,
        )

        # 4. Store results
        retrieved_data = {
            "query_text": query_text,
            "results": results,
            "result_count": len(results),
            "fusion_method": fusion_method,
        }
        output_ref = await self._state.store_result(
            task_id, "retrieved", retrieved_data,
        )

        logger.info(
            "task_id=%s: retrieved %d chunks via hybrid search (%s)",
            task_id,
            len(results),
            fusion_method,
        )

        return {"output_ref": output_ref}
