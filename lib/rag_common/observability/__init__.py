"""Observability setup — tracing, metrics, structured logging."""

from rag_common.observability.logging import setup_logging
from rag_common.observability.metrics import setup_metrics
from rag_common.observability.tracing import setup_tracing

__all__ = ["setup_logging", "setup_metrics", "setup_tracing"]
