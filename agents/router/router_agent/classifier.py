"""Intent classification for incoming queries."""

from __future__ import annotations

import logging

from rag_common.llm.client import LLMClient
from rag_common.models.query import QueryIntent

logger = logging.getLogger(__name__)

CLASSIFICATION_PROMPT = """Classify the user's query intent into exactly one of these categories:
- FACTUAL: Simple fact lookup, specific information retrieval
- ANALYTICAL: Requires reasoning, analysis, or interpretation
- COMPARATIVE: Comparing two or more entities, concepts, or documents
- PROCEDURAL: How-to questions, step-by-step instructions
- EXPLORATORY: Open-ended research, broad topic exploration

Respond with ONLY the category name (e.g., FACTUAL).

Query: {query}"""

INTENT_MAP = {
    "FACTUAL": QueryIntent.FACTUAL,
    "ANALYTICAL": QueryIntent.ANALYTICAL,
    "COMPARATIVE": QueryIntent.COMPARATIVE,
    "PROCEDURAL": QueryIntent.PROCEDURAL,
    "EXPLORATORY": QueryIntent.EXPLORATORY,
}


class IntentClassifier:
    """Classifies query intent using an LLM or falls back to keyword heuristics."""

    def __init__(self, llm_client: LLMClient, model: str | None = None) -> None:
        self._llm = llm_client
        self._model = model

    async def classify(self, query_text: str) -> tuple[QueryIntent, float]:
        """Classify query intent. Returns (intent, confidence)."""
        try:
            response = await self._llm.completion(
                messages=[
                    {"role": "system", "content": "You are an intent classifier. Respond with only the category name."},
                    {"role": "user", "content": CLASSIFICATION_PROMPT.format(query=query_text)},
                ],
                model=self._model,
                max_tokens=20,
                temperature=0.0,
            )
            raw = response.choices[0].message.content.strip().upper()
            intent = INTENT_MAP.get(raw, QueryIntent.FACTUAL)
            logger.info("Classified query as %s (raw=%s)", intent, raw)
            return intent, 0.9
        except Exception:
            logger.exception("Intent classification failed, defaulting to FACTUAL")
            return QueryIntent.FACTUAL, 0.5

    def classify_heuristic(self, query_text: str) -> QueryIntent:
        """Fallback keyword-based classification."""
        lower = query_text.lower()
        if any(w in lower for w in ["compare", "versus", "vs", "difference between"]):
            return QueryIntent.COMPARATIVE
        if any(w in lower for w in ["how to", "steps", "guide", "tutorial"]):
            return QueryIntent.PROCEDURAL
        if any(w in lower for w in ["analyze", "explain why", "reason", "impact"]):
            return QueryIntent.ANALYTICAL
        if any(w in lower for w in ["explore", "overview", "tell me about", "what is"]):
            return QueryIntent.EXPLORATORY
        return QueryIntent.FACTUAL
