"""Graph agent configuration.

Defaults sourced from:
- Cognee: extract_graph_from_data processes entities per chunk, batch_size=5
- Lawren API: graph expansion max_depth=3 (graphrag_search/graph_expansion.rs)
- Cognee: triplet extraction batch_size=100
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict

class GraphConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GRAPH_")

    grpc_port: int = 50056
    # Cognee's entity extraction typically yields 5-15 entities per chunk
    max_entities_per_chunk: int = 15
    relationship_extraction_model: str = "openai/gpt-4o"
    # Lawren: graph expansion BFS max depth=3 hops
    max_depth: int = 3
    # Cognee: processes chunks in batches of 5 for LLM entity extraction
    extraction_batch_size: int = 5
    # Stream config
    consumer_group: str = "graph-storage-agents"
    input_stream: str = "rag:stream:ingest:embedded"
    output_stream: str = "rag:stream:ingest:stored"
    dlq_stream: str = "rag:stream:graph:dlq"
    block_ms: int = 5000
