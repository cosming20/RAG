.PHONY: proto install test lint fmt dev infra up down clean health

# ── Proto Compilation ─────────────────────────────────────
proto:
	./scripts/generate_proto.sh

# ── Dependencies ──────────────────────────────────────────
install:
	uv sync --all-packages

# ── Testing ───────────────────────────────────────────────
test:
	uv run pytest tests/ agents/*/tests/ -v

test-unit:
	uv run pytest agents/*/tests/ -v -m "not integration"

test-integration:
	uv run pytest tests/integration/ -v

test-e2e:
	uv run pytest tests/e2e/ -v

# ── Code Quality ──────────────────────────────────────────
lint:
	uv run ruff check .
	uv run ruff format --check .

fmt:
	uv run ruff check --fix .
	uv run ruff format .

# ── Infrastructure ────────────────────────────────────────
infra:
	docker compose -f deploy/docker-compose.infra.yml up -d

infra-down:
	docker compose -f deploy/docker-compose.infra.yml down

# ── Full Stack ────────────────────────────────────────────
up:
	docker compose -f deploy/docker-compose.yml up -d --build

down:
	docker compose -f deploy/docker-compose.yml down

# ── Development ───────────────────────────────────────────
dev: infra install proto
	@echo "Development environment ready."
	@echo "  Redis:  localhost:6379"
	@echo "  Qdrant: localhost:6333 (REST) / 6334 (gRPC)"
	@echo "  Neo4j:  localhost:7687 (Bolt) / 7474 (Browser)"

# ── Health Check ──────────────────────────────────────────
health:
	uv run python scripts/health_check.py

# ── Cleanup ───────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf lib/rag_common/generated/
