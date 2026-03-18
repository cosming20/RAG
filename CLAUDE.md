# RAG Swarm — Project Rules

## Architecture Doc Mandate

**Every time code is added, modified, or removed — update `docs/architecture.md`.**
This is non-negotiable. The architecture doc is the single source of truth for how the system works. If the code changes but the doc doesn't, the doc is a lie.

Update the relevant sections:
- New agent? Update the agent table, data flow diagrams, stream topology, port map.
- New Redis key? Update the key schema section.
- New stream topic? Update the streams topology.
- Config change? Update the technology stack or config sections.
- New shared library module? Update the project structure tree.
- New dependency? Note it in the tech stack.

## Project Structure

This is a Python 3.12+ uv workspace monorepo with 9 gRPC microservices.

```
rag-swarm/
├── proto/                   # Protobuf definitions (buf for compilation)
├── lib/rag_common/          # Shared library (config, models, redis, grpc, llm, observability)
├── agents/{name}/           # Each agent is a separate uv workspace package
│   └── {name}_agent/        # Unique package name (NOT 'src/' — namespace collision)
├── deploy/                  # Docker Compose, K8s manifests
├── tests/                   # Integration + E2E tests
├── scripts/                 # Proto gen, health check, seed data
└── docs/                    # Architecture diagrams, specs
```

## Coding Standards

### Package Naming
- Each agent's code lives in `agents/{name}/{name}_agent/` (e.g., `agents/router/router_agent/`)
- NEVER use `src/` as a package name — all agents share the venv, so `src` collides
- pyproject.toml: `packages = ["{name}_agent"]`

### Configuration
- All config via Pydantic Settings with env prefix (e.g., `REDIS_HOST`, `AGENT_GRPC_PORT`)
- NO hardcoded values. Every number must come from config or be a named constant with a comment citing its source
- Config values must cite their source: Cognee, Lawren API, Qdrant docs, or a standard (BM25 k1=1.2)
- Use `@lru_cache` getters for config singletons (pattern: `get_redis_config()`)
- Agent-specific config must NOT override `AgentConfig.grpc_port` — that comes from `AGENT_GRPC_PORT` env var

### Secrets
- All secrets (API keys, passwords) use `pydantic.SecretStr`, never plain `str`
- `.env` file is gitignored. `.env.example` has empty placeholders
- Never log secret values

### Workers
- ALL stream workers MUST extend `BaseStreamWorker` from `rag_common.redis.worker`
- BaseStreamWorker provides: deadline checks, pipeline depth enforcement, max attempts / DLQ, idempotency checks, atomic claim (Lua CAS), envelope forwarding
- Workers only implement `_process_message(msg: StreamMessage) -> Any`
- Override `_publish_success()` only when routing differs (e.g., validation retry loop)
- Set `output_stage` in constructor for idempotency checks

### Error Handling
- LLM calls: 128s retry budget with exponential backoff, skip `NotFoundError`
- Rate limiting: `MovingWindowRateLimiter` (disabled by default, enable via `LLM_RATE_LIMIT_ENABLED=true`)
- gRPC calls: retry only `UNAVAILABLE`, `DEADLINE_EXCEEDED`, `RESOURCE_EXHAUSTED`, `ABORTED`
- Redis Streams: `BUSYGROUP` is expected on `ensure_group()`, don't catch other errors
- Task claim: Lua atomic CAS script, not pipeline HSET (race condition)
- Circuit breaker: half-open allows exactly one probe, not unlimited

### Security
- Cypher queries: ALL labels and relationship types MUST go through `_sanitize_cypher_identifier()`
- Never interpolate user/LLM-controlled strings into Cypher — use `$`-params for values
- BM25 encoder: use `hashlib.sha256` for deterministic hashing, NOT `hash()` (non-deterministic across processes)

### Data Flow Contracts
- Each agent's output key names must match the next agent's expected input key names
- Document the key names in the service docstring
- When changing output format, grep for all consumers of that stage

### Testing
- **DO NOT write unit tests, integration tests, or test files in the codebase**
- Tests will only be effective tests run manually or via scripts — not pytest suites
- No `tests/` directories in the repo
- Smoke test: each agent must start, connect to Redis, register, and shut down cleanly
- Smoke test command: `AGENT_GRPC_PORT={port} uv run --package rag-agent-{name} python -m {name}_agent.main`
- All 9 agents must pass smoke test before any PR
- E2E testing is done by actually running the system and sending real documents/queries

### Docker / Deployment
- Infrastructure ports: 2050-2057 (Redis, Qdrant, Neo4j, Jaeger)
- Agent gRPC ports: 2060-2068
- Agent metrics ports: 9090-9098
- Dockerfile CMD: `uv run --package rag-agent-{name} python -m {name}_agent.main`
- `docker-compose.infra.yml` for infrastructure only
- `docker-compose.yml` for full stack

## Port Map

| Service | Port |
|---------|------|
| Redis | 2050 |
| Qdrant REST | 2051 |
| Qdrant gRPC | 2052 |
| Neo4j Browser | 2053 |
| Neo4j Bolt | 2054 |
| Jaeger OTLP | 2055 |
| Jaeger HTTP | 2056 |
| Jaeger UI | 2057 |
| Router Agent | 2060 |
| Ingestion Agent | 2061 |
| Chunking Agent | 2062 |
| Embedding Agent | 2063 |
| Retrieval Agent | 2064 |
| Graph Agent | 2065 |
| Reranking Agent | 2066 |
| Synthesis Agent | 2067 |
| Validation Agent | 2068 |

## Key Design Decisions

### Why Redis Streams (not Kafka, not NATS)
- Lightweight, already needed for shared state
- Consumer groups provide automatic load balancing
- XCLAIM handles orphan recovery
- Good enough for our scale; swap later if needed

### Why Autonomous Swarm (not DAG pipeline)
- Agents self-register, self-coordinate via streams
- Scale any bottleneck independently
- No central orchestrator = no single point of failure
- Fan-out/fan-in via consumer groups

### Why BM25 (not BM42, not SPLADE)
- BM25 is built into Qdrant (Modifier.IDF on sparse index)
- BM42 was deprecated by Qdrant
- Client-side BM25 for query encoding only; Qdrant handles corpus-level IDF

### Why Jina Reranker (not Cohere, not cross-encoder)
- jina-reranker-v2-base-multilingual: 8K context, multilingual
- API-based = no GPU needed
- Graceful fallback to original ordering if API is down

### Why 90s drain grace period
- GPT-4o generation can take 30-60s
- Kubernetes default `terminationGracePeriodSeconds` = 30s is too short
- Match this in K8s manifests when deploying

## Reference Codebases

Patterns were drawn from these repos (in /Users/cosmingagea/workspace/rag/):
- **Cognee**: chunk_size=1500, batch_size=100, 128s retry, dedup, rate limiting, @lru_cache config
- **Lawren API**: search_ef=128, graph expansion BFS, staged search, preflight validation, hybrid search
- **Qdrant**: gRPC proto patterns, named vectors, sparse vector support
- **Docling**: document parsing, format detection, chunking strategies
