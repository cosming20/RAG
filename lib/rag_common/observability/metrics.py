"""Prometheus metrics setup."""

from __future__ import annotations

from prometheus_client import start_http_server

from rag_common.config import ObservabilityConfig


def setup_metrics(config: ObservabilityConfig | None = None) -> None:
    config = config or ObservabilityConfig()
    if config.enable_metrics:
        start_http_server(config.metrics_port)
