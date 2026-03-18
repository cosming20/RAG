"""Preflight validation for incoming documents before processing."""

from __future__ import annotations

import logging

from rag_common.models.document import DocumentFormat

from .config import IngestionConfig

logger = logging.getLogger(__name__)


class PreflightValidator:
    """Validates documents before ingestion processing.

    Performs lightweight checks (size, format, encoding) to reject
    obviously invalid documents early and avoid wasting compute.
    """

    def __init__(self, config: IngestionConfig) -> None:
        self._config = config

    def validate(
        self,
        content: bytes,
        filename: str,
        fmt: DocumentFormat,
    ) -> tuple[bool, list[str]]:
        """Run all preflight checks on a document.

        Returns:
            A tuple of (is_valid, errors). If is_valid is True, errors
            will be empty.
        """
        errors: list[str] = []

        self._check_not_empty(content, errors)
        self._check_size_limit(content, filename, errors)
        self._check_format_supported(fmt, errors)
        self._check_encoding(content, fmt, errors)

        is_valid = len(errors) == 0
        if not is_valid:
            logger.warning(
                "Preflight validation failed for %s: %s",
                filename,
                "; ".join(errors),
            )
        return is_valid, errors

    def _check_not_empty(self, content: bytes, errors: list[str]) -> None:
        if not content or len(content) == 0:
            errors.append("Document content is empty")

    def _check_size_limit(
        self, content: bytes, filename: str, errors: list[str]
    ) -> None:
        max_size = self._config.max_document_size_bytes
        if len(content) > max_size:
            errors.append(
                f"Document '{filename}' size {len(content)} bytes "
                f"exceeds limit of {max_size} bytes"
            )

    def _check_format_supported(
        self, fmt: DocumentFormat, errors: list[str]
    ) -> None:
        if fmt.value not in self._config.supported_formats:
            errors.append(
                f"Format '{fmt.value}' is not in the supported formats list"
            )

    def _check_encoding(
        self,
        content: bytes,
        fmt: DocumentFormat,
        errors: list[str],
    ) -> None:
        """Basic encoding validation for text-based formats."""
        text_formats = {
            DocumentFormat.HTML,
            DocumentFormat.MARKDOWN,
            DocumentFormat.CSV,
            DocumentFormat.CODE,
            DocumentFormat.TXT,
            DocumentFormat.LATEX,
        }
        if fmt not in text_formats:
            return

        try:
            content.decode("utf-8")
        except UnicodeDecodeError:
            # Try latin-1 as a fallback; it accepts any byte sequence
            try:
                content.decode("latin-1")
            except UnicodeDecodeError:
                errors.append(
                    "Document content could not be decoded as UTF-8 or Latin-1"
                )
