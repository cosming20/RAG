# RAG Swarm — System Architecture

## High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          RAG SWARM SYSTEM                                   │
│                   9 Autonomous gRPC Microservices                           │
│              Coordinated via Redis Streams (No Central Orchestrator)        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐            │
│   │  Qdrant  │    │  Neo4j   │    │  Redis   │    │  Jaeger  │            │
│   │ (Vectors)│    │ (Graph)  │    │ (Streams)│    │ (Traces) │            │
│   │ :6334    │    │ :7687    │    │ :6379    │    │ :16686   │            │
│   └──────────┘    └──────────┘    └──────────┘    └──────────┘            │
│                                                                             │
│   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐           │
│   │ Router  │ │Ingestion│ │Chunking │ │Embedding│ │Retrieval│           │
│   │  :2060  │ │  :2061  │ │  :2062  │ │  :2063  │ │  :2064  │           │
│   └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘           │
│   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐                       │
│   │  Graph  │ │Reranking│ │Synthesis│ │Validation│                       │
│   │  :2065  │ │  :2066  │ │  :2067  │ │  :2068  │                       │
│   └─────────┘ └─────────┘ └─────────┘ └─────────┘                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Document Ingestion Pipeline

```
                    ┌──────────────┐
                    │   Client     │
                    │ (gRPC call)  │
                    └──────┬───────┘
                           │ IngestDocument(bytes, filename)
                           ▼
                  ┌─────────────────┐
                  │  Router :2060   │
                  │                 │
                  │ • Validate size │
                  │ • SHA-256 dedup │
                  │ • Store raw     │
                  │ • Return task_id│
                  └────────┬────────┘
                           │ XADD rag:stream:ingest:pending
                           ▼
              ┌────────────────────────┐
              │  Ingestion Agent :2061 │
              │                        │
              │ • Preflight validation │
              │ • Docling parsing      │
              │   (PDF/DOCX/PPTX/HTML/ │
              │    images/LaTeX/code)  │
              │ • OCR for images       │
              └───────────┬────────────┘
                          │ XADD rag:stream:ingest:parsed
                          ▼
              ┌────────────────────────┐
              │  Chunking Agent :2062  │
              │                        │
              │ • Strategy selection:  │
              │   PDF → Hierarchical   │
              │   HTML → Hybrid        │
              │   Code → SyntaxAware   │
              │   CSV → Table          │
              │ • LLM normalization    │
              │   (OCR artifact fix)   │
              │ • Doc summarization    │
              └───────────┬────────────┘
                          │ XADD rag:stream:ingest:chunked
                          ▼
              ┌────────────────────────┐
              │  Embedding Agent :2063 │
              │                        │
              │ • Dense: LiteLLM       │
              │   (text-embedding-3-   │
              │    large, 3072 dims)   │
              │ • Sparse: BM25         │
              │   (client-side encoder)│
              │ • Batch size: 100      │
              └───────────┬────────────┘
                          │ XADD rag:stream:ingest:embedded
                          │
                 ┌────────┴────────┐  ← Fan-out (different consumer groups)
                 ▼                 ▼
    ┌──────────────────┐  ┌──────────────────┐
    │ Retrieval :2064  │  │   Graph :2065    │
    │                  │  │                  │
    │ • Qdrant upsert  │  │ • LLM entity     │
    │ • Named vectors: │  │   extraction     │
    │   dense + sparse │  │ • Neo4j MERGE:   │
    │ • Payload: text, │  │   Document →     │
    │   metadata       │  │   Chunk →        │
    └──────────────────┘  │   Entity         │
                          │ • Relationships  │
                          └──────────────────┘
                                  │
                    XADD rag:stream:ingest:stored ✓
```

## Query Pipeline

```
                    ┌──────────────┐
                    │   Client     │
                    │ (gRPC call)  │
                    └──────┬───────┘
                           │ Query(session_id, query_text)
                           ▼
                  ┌─────────────────────┐
                  │   Router :2060      │
                  │                     │
                  │ • Intent classify:  │
                  │   FACTUAL |         │
                  │   ANALYTICAL |      │
                  │   COMPARATIVE |     │
                  │   PROCEDURAL |      │
                  │   EXPLORATORY       │
                  └────────┬────────────┘
                           │ XADD rag:stream:query:classified
                           │
                  ┌────────┴────────┐  ← Fan-out (parallel retrieval)
                  ▼                 ▼
     ┌───────────────────┐ ┌───────────────────┐
     │ Retrieval :2064   │ │   Graph :2065     │
     │ (search path)     │ │ (query path)      │
     │                   │ │                   │
     │ • Embed query:    │ │ • LLM entity      │
     │   dense + BM25    │ │   extraction      │
     │ • Qdrant hybrid   │ │ • Neo4j BFS       │
     │   search          │ │   expansion       │
     │ • RRF/Convex      │ │   (depth 1-3)     │
     │   fusion          │ │ • Related chunks  │
     └────────┬──────────┘ └────────┬──────────┘
              │                     │
              │ rag:results:        │ rag:results:
              │ {task}:retrieved    │ {task}:graph_context
              │                     │
              └─────────┬───────────┘
                        │ ← Fan-in (merge point)
                        ▼
           ┌────────────────────────┐
           │  Reranking Agent :2066 │
           │                        │
           │ 1. Merge + deduplicate │
           │ 2. Jina Reranker v2    │
           │    (cross-encoder)     │
           │ 3. LLM relevance      │
           │    trimming            │
           │ 4. Reference           │
           │    resolution (BFS)    │
           └───────────┬────────────┘
                       │ XADD rag:stream:query:reranked
                       ▼
           ┌────────────────────────┐
           │  Synthesis Agent :2067 │
           │                        │
           │ • Build context:       │
           │   [1] chunk text...    │
           │   [2] chunk text...    │
           │ • LLM generation       │
           │   (GPT-4o)             │
           │ • Citation extraction  │
           │   ([N] → source map)   │
           │ • Streaming support    │
           └───────────┬────────────┘
                       │ XADD rag:stream:query:synthesized
                       ▼
           ┌────────────────────────┐
           │ Validation Agent :2068 │
           │                        │
           │ • Hallucination check: │
           │   claim → grounding    │
           │ • Attribution check:   │
           │   [N] → source exists? │
           │ • Quality score:       │
           │   0.6*ground +         │
           │   0.4*attribution      │
           └───────────┬────────────┘
                       │
              ┌────────┴────────┐
              ▼                 ▼
     quality ≥ 0.7?      quality < 0.7?
     ┌──────────┐        ┌──────────────────┐
     │  ACCEPT  │        │ RETRY (max 1)    │
     │          │        │ → feedback to     │
     │ → Router │        │   Synthesis       │
     │ → Client │        │ → re-synthesize   │
     └──────────┘        └──────────────────┘
```

## Swarm Protocol — Agent Lifecycle

```
┌──────────────────────────────────────────────────────────────────┐
│                    Agent Lifecycle (per agent)                     │
├──────────────────────────────────────────────────────────────────┤
│                                                                    │
│  START                                                             │
│    │                                                               │
│    ▼                                                               │
│  ┌─────────────┐     HSET rag:agents:{id}                        │
│  │  REGISTER   │────► SADD rag:agents:by_role:{role}             │
│  └──────┬──────┘                                                  │
│         │                                                          │
│         ▼                                                          │
│  ┌─────────────┐     Check Redis, Qdrant, Neo4j, LLM             │
│  │   VERIFY    │                                                  │
│  └──────┬──────┘                                                  │
│         │                                                          │
│         ▼                                                          │
│  ┌─────────────┐     Start gRPC server                            │
│  │   SERVE     │     Join consumer group(s)                       │
│  └──────┬──────┘                                                  │
│         │                                                          │
│    ┌────┴────┐                                                    │
│    ▼         ▼                                                    │
│  ┌─────┐  ┌──────────────────────────────┐                       │
│  │HEART│  │     CONSUMER LOOP            │                       │
│  │BEAT │  │                              │                       │
│  │every│  │  XREADGROUP ... BLOCK 5000   │                       │
│  │ 5s  │  │         │                    │                       │
│  │     │  │         ▼                    │                       │
│  │HSET │  │  ┌─ deadline expired? ──► SKIP                      │
│  │last_│  │  │                           │                       │
│  │heart│  │  ├─ depth > 10? ──────► DLQ                         │
│  │beat │  │  │                           │                       │
│  │     │  │  ├─ attempts ≥ max? ──► DLQ + POISON                │
│  │EXPIR│  │  │                           │                       │
│  │E 30s│  │  ▼                           │                       │
│  └─────┘  │  CLAIM (Lua atomic CAS)     │                       │
│           │  PROCESS                     │                       │
│           │  STORE result                │                       │
│           │  XADD next stream            │                       │
│           │  XACK                        │                       │
│           │  ─── loop ───                │                       │
│           └──────────────────────────────┘                       │
│                                                                    │
│  SIGTERM                                                           │
│    │                                                               │
│    ▼                                                               │
│  ┌─────────────┐     Stop reading new messages                    │
│  │   DRAIN     │     Finish in-flight tasks (30s grace)           │
│  └──────┬──────┘                                                  │
│         │                                                          │
│         ▼                                                          │
│  ┌─────────────┐     DEL rag:agents:{id}                         │
│  │ DEREGISTER  │     SREM rag:agents:by_role:{role}              │
│  └──────┬──────┘                                                  │
│         │                                                          │
│         ▼                                                          │
│       EXIT                                                         │
└──────────────────────────────────────────────────────────────────┘
```

## Redis Streams Topology

```
                        INGESTION PIPELINE
                        ==================

rag:stream:ingest:pending ──► [ingestion-agents]
        │
        ▼
rag:stream:ingest:parsed ───► [chunking-agents]
        │
        ▼
rag:stream:ingest:chunked ──► [embedding-agents]
        │
        ▼
rag:stream:ingest:embedded ─┬► [retrieval-storage-agents] → Qdrant
                            └► [graph-storage-agents]     → Neo4j
        │
        ▼
rag:stream:ingest:stored ───► (completion notification)


                        QUERY PIPELINE
                        ==============

rag:stream:query:classified ─┬► [retrieval-query-agents] → Qdrant search
                             └► [graph-query-agents]     → Neo4j BFS
        │
        ▼
rag:stream:query:retrieved ──► [reranking-agents]
        │                        (also reads graph_context from state store)
        ▼
rag:stream:query:reranked ───► [synthesis-agents]
        │                        ▲
        ▼                        │ (retry with feedback)
rag:stream:query:synthesized ► [validation-agents] ──────┘
        │
        ▼
rag:stream:query:validated ──► Router assembles response → Client


                        DEAD LETTER QUEUES
                        ==================

rag:stream:ingest:failed       (ingestion DLQ)
rag:stream:chunking:dlq        (chunking DLQ)
rag:stream:retrieval:dlq       (retrieval DLQ)
rag:stream:graph:dlq           (graph DLQ)
rag:stream:graph-query:dlq     (graph query DLQ)
```

## Redis Key Schema

```
┌─────────────────────────────────────────────────────────────┐
│                    Redis Key Layout                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  AGENT REGISTRY                                              │
│  ├── rag:agents:{agent_id}        → Hash (role, addr, hb)  │
│  └── rag:agents:by_role:{role}    → Set  (agent_ids)       │
│                                                              │
│  TASKS                                                       │
│  └── rag:tasks:{task_id}          → Hash (status, agent,   │
│                                      attempt, payload)      │
│                                                              │
│  INTERMEDIATE RESULTS (TTL: 1h)                              │
│  └── rag:results:{task_id}:{stage}→ JSON string             │
│      Stages:                                                 │
│      ├── :raw            (uploaded bytes, base64)            │
│      ├── :meta           (document metadata)                 │
│      ├── :parsed         (Docling output)                    │
│      ├── :chunks         (chunked + summary + topics)        │
│      ├── :embeddings     (dense + sparse vectors)            │
│      ├── :classification (intent + query_text)               │
│      ├── :retrieved      (Qdrant search results)             │
│      ├── :graph_context  (Neo4j expansion results)           │
│      ├── :reranked       (Jina + trimmed + references)       │
│      ├── :answer         (LLM response + citations)          │
│      └── :validation     (quality score + issues)            │
│                                                              │
│  SESSIONS (TTL: 1h)                                          │
│  ├── rag:sessions:{id}           → Hash                     │
│  └── rag:sessions:{id}:history   → List                     │
│                                                              │
│  CIRCUIT BREAKERS                                            │
│  └── rag:circuit:{role}          → Hash (state, failures)   │
│                                                              │
│  QUERY CACHE (Phase 4, opt-in via CACHE_ENABLED=true)       │
│  ├── rag:query_cache:index       → List (cache keys, LRU)  │
│  ├── rag:query_cache:{key}:embedding → JSON (query vector)  │
│  └── rag:query_cache:{key}:result    → JSON (cached answer) │
│      TTL: 7 days (Cognee pattern)                           │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
rag-swarm/                          158 source files
├── proto/                          14 proto files
│   ├── common/                     common.proto, health.proto, agent.proto
│   └── services/                   9 service protos (one per agent)
│
├── lib/rag_common/                 Shared library (32 files)
│   ├── config.py                   8 Pydantic Settings classes + @lru_cache getters
│   ├── errors.py                   Error hierarchy
│   ├── retry.py                    CircuitBreaker + grpc_retry
│   ├── models/                     Domain models (document, task w/ provenance, agent, query)
│   ├── redis/                      Client, Streams, Registry, TaskQueue, State, BaseStreamWorker, QueryCache
│   ├── grpc_utils/                 BaseGrpcServer, interceptors, client factory, run_agent helper
│   ├── observability/              OTel tracing, Prometheus metrics, structlog
│   └── llm/                        LiteLLM client (128s retry, rate limiter), parse_llm_json, utils
│
├── agents/                         9 agent microservices
│   ├── router/        :2060        Public gateway, intent classification
│   ├── ingestion/     :2061        Docling parsing, preflight validation
│   ├── chunking/      :2062        4 strategies, LLM normalizer, summarizer
│   ├── embedding/     :2063        Dense (LiteLLM) + BM25 sparse
│   ├── retrieval/     :2064        Qdrant hybrid search + RRF, storage
│   ├── graph/         :2065        Neo4j entity extraction + BFS expansion
│   ├── reranking/     :2066        Jina v2, LLM trimmer, reference resolver
│   ├── synthesis/     :2067        LLM generation, citations, streaming
│   └── validation/    :2068        Hallucination check, attribution, retry
│
├── deploy/
│   ├── docker-compose.yml          Full stack (9 agents + 4 infra)
│   ├── docker-compose.infra.yml    Infra only
│   └── k8s/                        Kubernetes manifests (placeholder)
│
├── tests/                          Integration + E2E (placeholder)
├── scripts/                        Proto gen, health check, seed data
├── .env                            API keys (OpenAI, Jina from Lawren)
├── Makefile                        dev, test, lint, up, down
└── pyproject.toml                  uv workspace root
```

## Technology Stack

```
┌──────────────┬────────────────────────────────────────────┐
│ Layer        │ Technology                                  │
├──────────────┼────────────────────────────────────────────┤
│ Language     │ Python 3.12+                               │
│ API          │ gRPC (grpcio + protobuf)                   │
│ Package Mgr  │ uv (workspace mode)                        │
│ Messaging    │ Redis Streams (consumer groups)             │
│ Shared State │ Redis Hash/String (TTL-managed)             │
│ Vector DB    │ Qdrant (dense + BM25 sparse, RRF fusion)   │
│ Graph DB     │ Neo4j (Cypher, APOC)                       │
│ Doc Parsing  │ Docling (PDF/DOCX/PPTX/HTML/images/LaTeX)  │
│ LLM          │ LiteLLM (OpenAI, Anthropic, Cohere, etc.)  │
│ Reranker     │ Jina Reranker v2 (multilingual)             │
│ Embeddings   │ text-embedding-3-large (3072d) + BM25      │
│ Tracing      │ OpenTelemetry → Jaeger                      │
│ Metrics      │ Prometheus (per-agent /metrics)             │
│ Logging      │ structlog (JSON)                            │
│ Containers   │ Docker Compose / Kubernetes                 │
└──────────────┴────────────────────────────────────────────┘
```

## Production Hardening (Phase 4)

### Resilience Features

| Feature | Implementation | Source |
|---------|---------------|--------|
| **Idempotency** | BaseStreamWorker checks `rag:results:{task_id}:{stage}` before processing | Cognee dedup pattern |
| **Rate limiting** | MovingWindowRateLimiter on LLM calls (configurable req/interval) | Cognee rate_limiter.py |
| **128s retry budget** | `stop_after_delay(128)` replaces fixed 3 attempts; skips NotFoundError | Cognee LiteLLMEmbeddingEngine |
| **Schema versioning** | `version: int = 1` in stream envelope for rolling upgrades | Architecture review feedback |
| **Failed task retention** | 24h TTL on failed/poison tasks (vs 1h normal) for debugging | Cognee 7-day cache pattern |
| **90s drain grace** | Graceful shutdown allows in-flight LLM calls to complete | GPT-4o latency profile |
| **Graph dedup** | Entity dedup by (name, type), edge dedup by (source, target, rel) | Cognee deduplicate_nodes_and_edges |
| **Provenance** | `source_pipeline` + `source_agent` stamps on Task model | Cognee DataPoint pattern |
| **Query cache** | Redis embedding-similarity cache (cosine > 0.95, 7-day TTL, opt-in) | Cognee + Lawren cache config |
| **Atomic task claim** | Lua CAS script prevents double-claiming | Redis best practice |
| **Pipeline depth limit** | Max 10 hops, enforced in BaseStreamWorker | Swarm loop prevention |
| **Cypher sanitization** | `_sanitize_cypher_identifier()` on all dynamic labels | Security review |
| **Deterministic hashing** | SHA-256 for BM25 sparse indices (not Python `hash()`) | Cross-process consistency |
| **Circuit breaker** | Half-open allows exactly one probe; closes on success | TOCTOU fix |

### Query Cache Flow

```
Query → Router
  ├─ CACHE_ENABLED=true?
  │   ├─ Embed query
  │   ├─ QueryCache.lookup(embedding)
  │   │   ├─ cosine > 0.95 → Return cached result (skip pipeline)
  │   │   └─ miss → Continue to pipeline
  │   └─ After pipeline: QueryCache.store(key, embedding, result)
  └─ CACHE_ENABLED=false → Pipeline directly
```
