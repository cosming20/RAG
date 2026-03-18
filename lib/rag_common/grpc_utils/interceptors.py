"""gRPC server interceptors for logging, metrics, and tracing."""

from __future__ import annotations

import functools
import logging
import time
from typing import Any

import grpc
from prometheus_client import Counter, Histogram

logger = logging.getLogger(__name__)

GRPC_REQUEST_COUNT = Counter(
    "grpc_requests_total",
    "Total gRPC requests",
    ["method", "status"],
)
GRPC_REQUEST_DURATION = Histogram(
    "grpc_request_duration_seconds",
    "gRPC request duration in seconds",
    ["method"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)


def _wrap_rpc_handler(handler: grpc.RpcMethodHandler, method: str) -> grpc.RpcMethodHandler:
    """Wrap an RPC handler to measure actual request processing time."""
    if handler is None:
        return handler

    if handler.unary_unary:
        original = handler.unary_unary

        @functools.wraps(original)
        async def timed_unary_unary(request: Any, context: grpc.aio.ServicerContext) -> Any:
            start = time.monotonic()
            try:
                result = await original(request, context)
                GRPC_REQUEST_COUNT.labels(method=method, status="ok").inc()
                return result
            except Exception:
                GRPC_REQUEST_COUNT.labels(method=method, status="error").inc()
                raise
            finally:
                duration = time.monotonic() - start
                GRPC_REQUEST_DURATION.labels(method=method).observe(duration)
                logger.info("gRPC %s completed in %.3fs", method, duration)

        return handler._replace(unary_unary=timed_unary_unary)

    if handler.unary_stream:
        original = handler.unary_stream

        @functools.wraps(original)
        async def timed_unary_stream(request: Any, context: grpc.aio.ServicerContext) -> Any:
            start = time.monotonic()
            try:
                async for item in original(request, context):
                    yield item
                GRPC_REQUEST_COUNT.labels(method=method, status="ok").inc()
            except Exception:
                GRPC_REQUEST_COUNT.labels(method=method, status="error").inc()
                raise
            finally:
                duration = time.monotonic() - start
                GRPC_REQUEST_DURATION.labels(method=method).observe(duration)

        return handler._replace(unary_stream=timed_unary_stream)

    # For stream_unary and stream_stream, return unwrapped for now
    return handler


class LoggingInterceptor(grpc.aio.ServerInterceptor):
    """Logs gRPC request start."""

    async def intercept_service(
        self,
        continuation: Any,
        handler_call_details: grpc.HandlerCallDetails,
    ) -> Any:
        method = handler_call_details.method
        logger.info("gRPC request: %s", method)
        return await continuation(handler_call_details)


class MetricsInterceptor(grpc.aio.ServerInterceptor):
    """Wraps RPC handlers to measure actual request processing duration."""

    async def intercept_service(
        self,
        continuation: Any,
        handler_call_details: grpc.HandlerCallDetails,
    ) -> Any:
        method = handler_call_details.method
        handler = await continuation(handler_call_details)
        return _wrap_rpc_handler(handler, method)
