FROM python:3.12-slim AS base

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy workspace root and shared library
COPY pyproject.toml uv.lock* ./
COPY lib/rag_common ./lib/rag_common

# Copy this agent
COPY agents/ingestion ./agents/ingestion

# Install dependencies (includes docling + docling-core)
RUN uv sync --frozen --package rag-agent-ingestion

# Run
ENV AGENT_GRPC_PORT=50051
EXPOSE 50051
EXPOSE 9090

CMD ["uv", "run", "--package", "rag-agent-ingestion", "python", "-m", "ingestion_agent.main"]
