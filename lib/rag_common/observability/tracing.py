"""OpenTelemetry distributed tracing setup."""

from __future__ import annotations

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from rag_common.config import ObservabilityConfig


def setup_tracing(config: ObservabilityConfig | None = None) -> trace.Tracer:
    config = config or ObservabilityConfig()
    if not config.enable_tracing:
        return trace.get_tracer(config.service_name)

    resource = Resource.create({"service.name": config.service_name})
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=config.exporter_endpoint, insecure=True)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    return trace.get_tracer(config.service_name)
