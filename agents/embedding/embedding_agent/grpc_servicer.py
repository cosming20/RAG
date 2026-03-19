"""Embedding Agent gRPC servicer -- wraps EmbeddingService for gRPC transport.

Maps gRPC RPC methods to :class:`EmbeddingService` calls.  When proto
generation is complete, this class should inherit from the generated
``EmbeddingAgentServicer`` base class.

Proto service: ``rag.services.embedding.EmbeddingAgent``
RPCs: Embed, EmbedBatch, EmbedStream, EmbedHybrid
"""

from __future__ import annotations

import logging
import time
from typing import Any

import grpc

from .service import EmbeddingService

logger = logging.getLogger(__name__)


class EmbeddingServicer:
    """gRPC servicer for the Embedding Agent.

    Delegates embedding operations to :class:`EmbeddingService`.

    Once proto stubs are generated, update this class to inherit from
    ``embedding_pb2_grpc.EmbeddingAgentServicer`` and return proper
    proto message objects.
    """

    def __init__(self, service: EmbeddingService) -> None:
        self._service = service

    async def Embed(
        self, request: Any, context: grpc.aio.ServicerContext,
    ) -> Any:
        """Generate dense embedding for a single text.

        Proto: EmbedRequest -> EmbedResponse
        """
        start = time.monotonic()
        try:
            text: str = request.text if hasattr(request, "text") else ""
            if not text:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("text field is required")
                return None

            # Use the dense embedder for a single text
            dense_vectors = await self._service._dense.embed_texts([text])

            elapsed = (time.monotonic() - start) * 1000
            logger.info("Embed completed in %.1fms", elapsed)

            return {
                "success": True,
                "vector": {
                    "values": dense_vectors[0],
                    "model": self._service._config.dense_model,
                    "dimensions": len(dense_vectors[0]),
                },
            }

        except Exception as exc:
            logger.exception("Embed failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return None

    async def EmbedBatch(
        self, request: Any, context: grpc.aio.ServicerContext,
    ) -> Any:
        """Generate dense embeddings for multiple texts.

        Proto: EmbedBatchRequest -> EmbedBatchResponse
        """
        start = time.monotonic()
        try:
            texts: list[str] = list(request.texts) if hasattr(request, "texts") else []
            if not texts:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("texts field is required and must not be empty")
                return None

            dense_vectors = await self._service._dense.embed_texts(texts)

            elapsed = (time.monotonic() - start) * 1000
            logger.info("EmbedBatch completed in %.1fms (%d texts)", elapsed, len(texts))

            vectors = [
                {
                    "values": vec,
                    "model": self._service._config.dense_model,
                    "dimensions": len(vec),
                }
                for vec in dense_vectors
            ]

            return {"success": True, "vectors": vectors}

        except Exception as exc:
            logger.exception("EmbedBatch failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return None

    async def EmbedHybrid(
        self, request: Any, context: grpc.aio.ServicerContext,
    ) -> Any:
        """Generate dense + sparse (hybrid) embeddings for multiple texts.

        Proto: EmbedHybridRequest -> EmbedHybridResponse
        """
        start = time.monotonic()
        try:
            texts: list[str] = list(request.texts) if hasattr(request, "texts") else []
            if not texts:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("texts field is required and must not be empty")
                return None

            # Generate both dense and sparse vectors
            dense_vectors = await self._service._dense.embed_texts(texts)
            sparse_vectors = self._service._sparse.encode(texts)

            elapsed = (time.monotonic() - start) * 1000
            logger.info("EmbedHybrid completed in %.1fms (%d texts)", elapsed, len(texts))

            hybrid_vectors = []
            for dense_vec, sparse_vec in zip(dense_vectors, sparse_vectors):
                hybrid_vectors.append({
                    "dense": {
                        "values": dense_vec,
                        "model": self._service._config.dense_model,
                        "dimensions": len(dense_vec),
                    },
                    "sparse": {
                        "indices": sparse_vec["indices"],
                        "values": sparse_vec["values"],
                    },
                })

            return {"success": True, "vectors": hybrid_vectors}

        except Exception as exc:
            logger.exception("EmbedHybrid failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return None
