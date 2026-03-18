"""Hallucination detection via LLM-based claim extraction and grounding.

Extracts factual claims from the generated answer, then checks each claim
against the source chunks for grounding support.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from rag_common.llm.client import LLMClient

logger = logging.getLogger(__name__)

# Number of claims to check per LLM batch call
CLAIM_BATCH_SIZE = 3

CLAIM_EXTRACTION_PROMPT = (
    "Extract factual claims from this answer. Return JSON: "
    '{{"claims": ["claim1", "claim2", ...]}}\n\n'
    "Answer:\n{answer_text}"
)

GROUNDING_CHECK_PROMPT = (
    "Are these claims supported by the provided context? "
    "For each claim, determine if it is grounded in the context.\n\n"
    "Claims:\n{claims_text}\n\n"
    "Context:\n{context_text}\n\n"
    "Return JSON: {{"
    '"results": [{{"claim": "...", "supported": true/false, '
    '"confidence": 0.9, "explanation": "..."}}]'
    "}}"
)


class HallucinationChecker:
    """Detects hallucinated claims by verifying grounding against source chunks."""

    def __init__(self, llm_client: LLMClient, model: str) -> None:
        self._llm = llm_client
        self._model = model

    async def check(
        self,
        answer_text: str,
        source_chunks: list[dict],
    ) -> dict[str, Any]:
        """Extract claims from the answer and verify each against sources.

        Args:
            answer_text: The generated answer to validate.
            source_chunks: The context chunks used during generation.

        Returns:
            Dict with ``claims`` (list of claim results) and
            ``grounding_score`` (fraction of grounded claims).
        """
        # 1. Extract claims
        claims = await self._extract_claims(answer_text)
        if not claims:
            logger.info("No factual claims extracted from answer")
            return {"claims": [], "grounding_score": 1.0}

        # 2. Build combined context from source chunks
        context_text = self._build_context_text(source_chunks)

        # 3. Check grounding in batches
        claim_results: list[dict[str, Any]] = []
        for i in range(0, len(claims), CLAIM_BATCH_SIZE):
            batch = claims[i : i + CLAIM_BATCH_SIZE]
            batch_results = await self._check_grounding_batch(batch, context_text)
            claim_results.extend(batch_results)

        # 4. Compute grounding score
        grounded_count = sum(1 for c in claim_results if c.get("is_grounded", False))
        total = len(claim_results)
        grounding_score = grounded_count / total if total > 0 else 1.0

        logger.info(
            "Hallucination check: %d/%d claims grounded (score=%.2f)",
            grounded_count,
            total,
            grounding_score,
        )

        return {
            "claims": claim_results,
            "grounding_score": grounding_score,
        }

    async def _extract_claims(self, answer_text: str) -> list[str]:
        """Use LLM to extract factual claims from the answer text."""
        prompt = CLAIM_EXTRACTION_PROMPT.format(answer_text=answer_text)
        messages = [{"role": "user", "content": prompt}]

        try:
            response = await self._llm.completion(
                messages,
                model=self._model,
                temperature=0.0,
                max_tokens=2048,
            )
            content = response.choices[0].message.content or ""
            parsed = self._parse_json_response(content)
            claims = parsed.get("claims", [])
            if isinstance(claims, list):
                return [str(c) for c in claims]
        except Exception:
            logger.exception("Failed to extract claims from answer")

        return []

    async def _check_grounding_batch(
        self,
        claims: list[str],
        context_text: str,
    ) -> list[dict[str, Any]]:
        """Check a batch of claims against the context."""
        claims_text = "\n".join(f"- {c}" for c in claims)
        prompt = GROUNDING_CHECK_PROMPT.format(
            claims_text=claims_text,
            context_text=context_text,
        )
        messages = [{"role": "user", "content": prompt}]

        try:
            response = await self._llm.completion(
                messages,
                model=self._model,
                temperature=0.0,
                max_tokens=2048,
            )
            content = response.choices[0].message.content or ""
            parsed = self._parse_json_response(content)
            raw_results = parsed.get("results", [])

            results: list[dict[str, Any]] = []
            for idx, item in enumerate(raw_results):
                claim_text = item.get("claim", claims[idx] if idx < len(claims) else "")
                results.append({
                    "text": claim_text,
                    "is_grounded": bool(item.get("supported", False)),
                    "confidence": float(item.get("confidence", 0.0)),
                    "explanation": item.get("explanation", ""),
                })
            return results

        except Exception:
            logger.exception("Failed to check grounding for claim batch")
            # On failure, mark all claims as ungrounded (conservative)
            return [
                {
                    "text": c,
                    "is_grounded": False,
                    "confidence": 0.0,
                    "explanation": "Grounding check failed",
                }
                for c in claims
            ]

    @staticmethod
    def _build_context_text(source_chunks: list[dict]) -> str:
        """Concatenate source chunk texts for grounding verification."""
        parts: list[str] = []
        for idx, chunk in enumerate(source_chunks, start=1):
            text = chunk.get("text", "")
            source = chunk.get("filename", "unknown")
            parts.append(f"[{idx}] ({source}): {text}")
        return "\n\n".join(parts)

    @staticmethod
    def _parse_json_response(content: str) -> dict[str, Any]:
        """Parse JSON from LLM response, handling markdown code fences."""
        text = content.strip()

        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last fence lines
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON from LLM response: %s...", text[:200])
            return {}
