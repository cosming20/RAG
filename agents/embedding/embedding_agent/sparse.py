"""Sparse embedding generation using BM25 (built-in to Qdrant).

Qdrant natively supports BM25 sparse vectors via its built-in tokenizer.
Instead of computing sparse vectors client-side, we store the raw text
in Qdrant payloads and configure a sparse vector index with BM25 modifier.

This module provides a lightweight client-side BM25 encoder for cases where
pre-computed sparse vectors are needed (e.g., hybrid search with explicit
sparse queries), and helpers to configure Qdrant's built-in BM25.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from collections import Counter
from typing import TypedDict

logger = logging.getLogger(__name__)

_TOKENIZE_PATTERN = re.compile(r"[a-z0-9]+")

# BM25 standard parameters (Robertson & Walker, Okapi BM25, 1994)
# k1 controls term frequency saturation: higher = more weight to repeated terms
# b controls document length normalization: 0 = no normalization, 1 = full
# These are the TREC/Lucene/Elasticsearch/Qdrant defaults used universally
BM25_K1 = 1.2
BM25_B = 0.75


class SparseEntry(TypedDict):
    """A single sparse vector representation."""

    indices: list[int]
    values: list[float]


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split into tokens."""
    return _TOKENIZE_PATTERN.findall(text.lower())


class BM25Encoder:
    """Client-side BM25 sparse encoder.

    Computes BM25 term weights for a batch of documents. Used to generate
    sparse query vectors for Qdrant hybrid search. For document indexing,
    prefer Qdrant's built-in BM25 modifier on the sparse vector index.

    Note: Qdrant's built-in BM25 is preferred for indexing because it
    maintains corpus-level IDF statistics. This encoder uses batch-level
    IDF which is an approximation suitable for query-time encoding.
    """

    def __init__(self, vocab_size: int = 30_000) -> None:
        self._vocab_size = vocab_size

    def encode(self, texts: list[str]) -> list[SparseEntry]:
        """Encode texts into BM25 sparse vectors."""
        if not texts:
            return []

        tokenized = [_tokenize(t) for t in texts]

        # Compute corpus statistics for the batch
        doc_count = len(tokenized)
        doc_freq: Counter[str] = Counter()
        doc_lengths = []
        for tokens in tokenized:
            doc_freq.update(set(tokens))
            doc_lengths.append(len(tokens))

        avg_dl = sum(doc_lengths) / max(doc_count, 1)

        results: list[SparseEntry] = []
        for i, tokens in enumerate(tokenized):
            if not tokens:
                results.append(SparseEntry(indices=[], values=[]))
                continue

            tf = Counter(tokens)
            dl = doc_lengths[i]

            index_value: dict[int, float] = {}
            for token, count in tf.items():
                # BM25 term frequency component
                tf_norm = (count * (BM25_K1 + 1)) / (
                    count + BM25_K1 * (1 - BM25_B + BM25_B * dl / avg_dl)
                )
                # IDF component
                idf = math.log(
                    (doc_count - doc_freq[token] + 0.5) / (doc_freq[token] + 0.5) + 1.0
                )
                score = tf_norm * idf

                if score > 0:
                    idx = int.from_bytes(hashlib.sha256(token.encode()).digest()[:4], "little") % self._vocab_size
                    index_value[idx] = index_value.get(idx, 0.0) + score

            sorted_pairs = sorted(index_value.items())
            results.append(SparseEntry(
                indices=[p[0] for p in sorted_pairs],
                values=[p[1] for p in sorted_pairs],
            ))

        return results

    def encode_query(self, query: str) -> SparseEntry:
        """Encode a single query into a BM25 sparse vector.

        For queries, we use simple term frequency without length normalization
        since queries are short.
        """
        tokens = _tokenize(query)
        if not tokens:
            return SparseEntry(indices=[], values=[])

        tf = Counter(tokens)
        index_value: dict[int, float] = {}
        for token, count in tf.items():
            idx = int.from_bytes(hashlib.sha256(token.encode()).digest()[:4], "little") % self._vocab_size
            index_value[idx] = index_value.get(idx, 0.0) + float(count)

        sorted_pairs = sorted(index_value.items())
        return SparseEntry(
            indices=[p[0] for p in sorted_pairs],
            values=[p[1] for p in sorted_pairs],
        )
