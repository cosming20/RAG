"""Shared test fixtures for integration and e2e tests."""

from __future__ import annotations

import asyncio
from typing import AsyncGenerator

import pytest
import pytest_asyncio


@pytest.fixture(scope="session")
def event_loop():
    """Create a session-scoped event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# Testcontainer fixtures will be added in Phase 4
# when we write integration tests with real Redis, Qdrant, Neo4j.
