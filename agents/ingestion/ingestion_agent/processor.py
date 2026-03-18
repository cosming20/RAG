"""Document parsing via Docling with graceful fallback."""

from __future__ import annotations

import io
import logging
from typing import Any

from rag_common.models.document import DocumentFormat

logger = logging.getLogger(__name__)

# Conditionally import Docling -- it may not be installed in all environments.
try:
    from docling.document_converter import DocumentConverter  # type: ignore[import-untyped]

    _DOCLING_AVAILABLE = True
except ImportError:
    _DOCLING_AVAILABLE = False
    logger.info("Docling not installed; falling back to basic text extraction")


class DocumentProcessor:
    """Parses raw document bytes into structured content.

    Uses Docling's DocumentConverter when available.  Falls back to
    basic text extraction otherwise.
    """

    def __init__(self, *, ocr_enabled: bool = False) -> None:
        self._ocr_enabled = ocr_enabled
        self._converter: Any | None = None
        if _DOCLING_AVAILABLE:
            self._converter = DocumentConverter()

    def process(
        self,
        content: bytes,
        filename: str,
        fmt: DocumentFormat,
    ) -> dict[str, Any]:
        """Parse a document and return structured content.

        Returns:
            A dict with keys: text, tables, images, metadata.
        """
        if self._converter is not None:
            try:
                return self._process_with_docling(content, filename, fmt)
            except Exception:
                logger.warning(
                    "Docling parsing failed for %s; falling back to basic extraction",
                    filename,
                    exc_info=True,
                )

        return self._basic_text_extraction(content, filename, fmt)

    # ------------------------------------------------------------------ #
    #  Docling-based parsing                                              #
    # ------------------------------------------------------------------ #

    def _process_with_docling(
        self,
        content: bytes,
        filename: str,
        fmt: DocumentFormat,
    ) -> dict[str, Any]:
        """Parse using Docling's DocumentConverter."""
        source = io.BytesIO(content)
        source.name = filename  # Docling uses .name for format detection

        result = self._converter.convert(source)  # type: ignore[union-attr]
        doc = result.document

        text_parts: list[str] = []
        tables: list[dict[str, Any]] = []
        images: list[dict[str, Any]] = []

        for element in doc.iterate_items():
            item = element if not isinstance(element, tuple) else element[1]
            item_type = type(item).__name__

            if item_type == "TableItem":
                table_data = self._extract_table(item)
                tables.append(table_data)
                # Also add a text representation for downstream chunking
                text_parts.append(table_data.get("text_repr", ""))
            elif item_type == "PictureItem":
                images.append({"caption": getattr(item, "caption", "")})
            else:
                text_content = getattr(item, "text", None)
                if text_content:
                    text_parts.append(text_content)

        return {
            "text": "\n\n".join(text_parts),
            "tables": tables,
            "images": images,
            "metadata": {
                "parser": "docling",
                "filename": filename,
                "format": fmt.value,
                "num_tables": str(len(tables)),
                "num_images": str(len(images)),
            },
        }

    @staticmethod
    def _extract_table(table_item: Any) -> dict[str, Any]:
        """Extract structured data from a Docling TableItem."""
        text_repr = getattr(table_item, "text", "")
        csv_repr = ""
        if hasattr(table_item, "export_to_dataframe"):
            try:
                df = table_item.export_to_dataframe()
                csv_repr = df.to_csv(index=False)
            except Exception:
                logger.debug("Could not export table to CSV", exc_info=True)

        return {
            "text_repr": text_repr,
            "csv": csv_repr,
        }

    # ------------------------------------------------------------------ #
    #  Fallback: basic text extraction                                    #
    # ------------------------------------------------------------------ #

    def _basic_text_extraction(
        self,
        content: bytes,
        filename: str,
        fmt: DocumentFormat,
    ) -> dict[str, Any]:
        """Extract text from document bytes using simple decoding."""
        text = self._decode_content(content)

        return {
            "text": text,
            "tables": [],
            "images": [],
            "metadata": {
                "parser": "basic",
                "filename": filename,
                "format": fmt.value,
            },
        }

    @staticmethod
    def _decode_content(content: bytes) -> str:
        """Try common encodings and return decoded text."""
        for encoding in ("utf-8", "latin-1", "cp1252"):
            try:
                return content.decode(encoding)
            except UnicodeDecodeError:
                continue
        # Last resort: decode with replacement characters
        return content.decode("utf-8", errors="replace")
