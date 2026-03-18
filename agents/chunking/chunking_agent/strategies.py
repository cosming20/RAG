"""Chunking strategies -- split documents into semantically meaningful chunks."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any

from rag_common.models.document import DocumentFormat

from .config import ChunkingConfig


class ChunkingStrategy(ABC):
    """Protocol for all chunking strategies."""

    @abstractmethod
    def chunk(self, text: str, metadata: dict[str, Any]) -> list[dict[str, Any]]:
        """Split text into chunks.

        Each chunk dict contains:
            text, sequence_index, start_offset, end_offset, metadata
        """


class HierarchicalChunker(ChunkingStrategy):
    """Splits by headings/sections, then by paragraphs.

    Respects document structure: first tries to split on markdown-style
    headings, then falls back to paragraph boundaries within sections.
    """

    def __init__(self, config: ChunkingConfig) -> None:
        self._max_size = config.default_chunk_size
        self._min_size = config.min_chunk_size

    def chunk(self, text: str, metadata: dict[str, Any]) -> list[dict[str, Any]]:
        sections = self._split_by_headings(text)
        chunks: list[dict[str, Any]] = []
        offset = 0

        for section in sections:
            if len(section) <= self._max_size:
                if len(section.strip()) >= self._min_size:
                    chunks.append(self._make_chunk(
                        section.strip(), len(chunks), offset, metadata,
                    ))
                offset += len(section)
                continue

            # Section too large -- split by paragraphs
            paragraphs = self._split_by_paragraphs(section)
            current_block: list[str] = []
            current_len = 0

            for para in paragraphs:
                para_len = len(para)
                if current_len + para_len > self._max_size and current_block:
                    merged = "\n\n".join(current_block).strip()
                    if len(merged) >= self._min_size:
                        chunks.append(self._make_chunk(
                            merged, len(chunks), offset, metadata,
                        ))
                    offset += len(merged) + 2  # account for join separator
                    current_block = []
                    current_len = 0
                current_block.append(para)
                current_len += para_len + 2

            if current_block:
                merged = "\n\n".join(current_block).strip()
                if len(merged) >= self._min_size:
                    chunks.append(self._make_chunk(
                        merged, len(chunks), offset, metadata,
                    ))
                offset += len(merged)

        return chunks

    @staticmethod
    def _split_by_headings(text: str) -> list[str]:
        """Split text on markdown headings (# ... or ======)."""
        pattern = r"(?=^#{1,6}\s|\n#{1,6}\s)"
        parts = re.split(pattern, text)
        return [p for p in parts if p.strip()]

    @staticmethod
    def _split_by_paragraphs(text: str) -> list[str]:
        parts = re.split(r"\n\s*\n", text)
        return [p for p in parts if p.strip()]

    @staticmethod
    def _make_chunk(
        text: str,
        index: int,
        offset: int,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "text": text,
            "sequence_index": index,
            "start_offset": offset,
            "end_offset": offset + len(text),
            "metadata": {**metadata, "strategy": "hierarchical"},
        }


class HybridChunker(ChunkingStrategy):
    """Sliding window with overlap, respecting sentence boundaries."""

    def __init__(self, config: ChunkingConfig) -> None:
        self._max_size = config.default_chunk_size
        self._min_size = config.min_chunk_size
        self._overlap = config.chunk_overlap

    def chunk(self, text: str, metadata: dict[str, Any]) -> list[dict[str, Any]]:
        sentences = self._split_sentences(text)
        chunks: list[dict[str, Any]] = []
        current_block: list[str] = []
        current_len = 0
        offset = 0

        for sentence in sentences:
            sent_len = len(sentence)
            if current_len + sent_len > self._max_size and current_block:
                merged = " ".join(current_block).strip()
                if len(merged) >= self._min_size:
                    chunks.append({
                        "text": merged,
                        "sequence_index": len(chunks),
                        "start_offset": offset,
                        "end_offset": offset + len(merged),
                        "metadata": {**metadata, "strategy": "hybrid"},
                    })

                # Overlap: keep trailing sentences whose total length <= overlap
                overlap_block: list[str] = []
                overlap_len = 0
                for s in reversed(current_block):
                    if overlap_len + len(s) > self._overlap:
                        break
                    overlap_block.insert(0, s)
                    overlap_len += len(s) + 1

                offset += current_len - overlap_len
                current_block = overlap_block
                current_len = overlap_len

            current_block.append(sentence)
            current_len += sent_len + 1  # +1 for space join

        if current_block:
            merged = " ".join(current_block).strip()
            if len(merged) >= self._min_size:
                chunks.append({
                    "text": merged,
                    "sequence_index": len(chunks),
                    "start_offset": offset,
                    "end_offset": offset + len(merged),
                    "metadata": {**metadata, "strategy": "hybrid"},
                })

        return chunks

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split on sentence boundaries (period/question/exclamation followed by space)."""
        parts = re.split(r"(?<=[.!?])\s+", text)
        return [p for p in parts if p.strip()]


class SyntaxAwareChunker(ChunkingStrategy):
    """Splits code files by function/class definitions."""

    def __init__(self, config: ChunkingConfig) -> None:
        self._max_size = config.default_chunk_size
        self._min_size = config.min_chunk_size

    def chunk(self, text: str, metadata: dict[str, Any]) -> list[dict[str, Any]]:
        # Match common definition patterns across languages
        pattern = r"(?=^(?:def |class |fn |func |function |pub fn |pub struct |impl |async fn |export ))"
        blocks = re.split(pattern, text, flags=re.MULTILINE)
        chunks: list[dict[str, Any]] = []
        offset = 0

        for block in blocks:
            stripped = block.strip()
            if not stripped:
                offset += len(block)
                continue

            if len(stripped) <= self._max_size:
                if len(stripped) >= self._min_size:
                    chunks.append({
                        "text": stripped,
                        "sequence_index": len(chunks),
                        "start_offset": offset,
                        "end_offset": offset + len(stripped),
                        "metadata": {**metadata, "strategy": "syntax_aware"},
                    })
            else:
                # Large block: fall back to line-based splitting
                lines = stripped.split("\n")
                current_lines: list[str] = []
                current_len = 0
                for line in lines:
                    if current_len + len(line) > self._max_size and current_lines:
                        merged = "\n".join(current_lines)
                        if len(merged) >= self._min_size:
                            chunks.append({
                                "text": merged,
                                "sequence_index": len(chunks),
                                "start_offset": offset,
                                "end_offset": offset + len(merged),
                                "metadata": {**metadata, "strategy": "syntax_aware"},
                            })
                        offset += len(merged) + 1
                        current_lines = []
                        current_len = 0
                    current_lines.append(line)
                    current_len += len(line) + 1

                if current_lines:
                    merged = "\n".join(current_lines)
                    if len(merged) >= self._min_size:
                        chunks.append({
                            "text": merged,
                            "sequence_index": len(chunks),
                            "start_offset": offset,
                            "end_offset": offset + len(merged),
                            "metadata": {**metadata, "strategy": "syntax_aware"},
                        })

            offset += len(block)

        return chunks


class TableChunker(ChunkingStrategy):
    """Chunks table content -- one chunk per table or row group."""

    def __init__(self, config: ChunkingConfig) -> None:
        self._max_size = config.default_chunk_size
        self._min_size = config.min_chunk_size

    def chunk(self, text: str, metadata: dict[str, Any]) -> list[dict[str, Any]]:
        """Split table text into chunks.

        Expects table text to be rows separated by newlines.  Groups
        rows into chunks that fit within max_size.
        """
        rows = text.strip().split("\n")
        chunks: list[dict[str, Any]] = []
        current_rows: list[str] = []
        current_len = 0
        offset = 0

        # Keep header row for each chunk if present
        header = rows[0] if rows else ""
        data_rows = rows[1:] if len(rows) > 1 else rows

        for row in data_rows:
            row_len = len(row) + 1  # +1 for newline
            if current_len + row_len > self._max_size and current_rows:
                merged = "\n".join(current_rows)
                if len(merged) >= self._min_size:
                    chunks.append({
                        "text": merged,
                        "sequence_index": len(chunks),
                        "start_offset": offset,
                        "end_offset": offset + len(merged),
                        "metadata": {**metadata, "strategy": "table"},
                    })
                offset += len(merged) + 1
                current_rows = [header] if header else []
                current_len = len(header) + 1 if header else 0

            current_rows.append(row)
            current_len += row_len

        if current_rows:
            merged = "\n".join(current_rows)
            if len(merged) >= self._min_size:
                chunks.append({
                    "text": merged,
                    "sequence_index": len(chunks),
                    "start_offset": offset,
                    "end_offset": offset + len(merged),
                    "metadata": {**metadata, "strategy": "table"},
                })

        return chunks


def select_strategy(
    fmt: DocumentFormat,
    config: ChunkingConfig,
) -> ChunkingStrategy:
    """Factory: choose the best chunking strategy for a document format."""
    if fmt == DocumentFormat.CODE:
        return SyntaxAwareChunker(config)

    if fmt in {DocumentFormat.CSV, DocumentFormat.XLSX}:
        return TableChunker(config)

    if fmt in {DocumentFormat.MARKDOWN, DocumentFormat.HTML, DocumentFormat.LATEX}:
        return HierarchicalChunker(config)

    # Default: hybrid sliding-window for PDF, DOCX, PPTX, TXT, IMAGE
    return HybridChunker(config)
