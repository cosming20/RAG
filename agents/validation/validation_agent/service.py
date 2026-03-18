"""Validation service -- orchestrates hallucination and attribution checks."""

from __future__ import annotations

import logging
from typing import Any

from rag_common.llm.client import LLMClient
from rag_common.redis.client import RedisClient
from rag_common.redis.state import StateStore

from agents.validation.validation_agent.attribution import AttributionChecker
from agents.validation.validation_agent.config import ValidationConfig
from agents.validation.validation_agent.hallucination import HallucinationChecker

logger = logging.getLogger(__name__)

# Weights for composite quality score
GROUNDING_WEIGHT = 0.6
ATTRIBUTION_WEIGHT = 0.4


class ValidationService:
    """Runs hallucination and attribution checks on a synthesized answer.

    Computes a composite quality score and determines whether the answer
    meets the acceptable threshold or needs revision.
    """

    def __init__(
        self,
        redis_client: RedisClient,
        llm_client: LLMClient,
        config: ValidationConfig,
    ) -> None:
        self._state = StateStore(redis_client)
        self._config = config
        self._hallucination_checker = HallucinationChecker(
            llm_client, config.hallucination_model,
        )
        self._attribution_checker = AttributionChecker()

    async def validate(
        self,
        task_id: str,
        answer_data: dict[str, Any],
        source_chunks: list[dict],
        query_text: str,
    ) -> dict[str, Any]:
        """Run validation checks on a synthesized answer.

        Args:
            task_id: Unique task identifier.
            answer_data: The answer dict with ``text``, ``citations``, ``model``.
            source_chunks: The context chunks used for generation.
            query_text: The original user query.

        Returns:
            Dict with ``output_ref``, ``quality_score``, ``is_acceptable``,
            and ``revision_feedback`` (if not acceptable).
        """
        answer_text = answer_data.get("text", "")
        citations = answer_data.get("citations", [])

        # 1. Hallucination check
        hallucination_result = await self._hallucination_checker.check(
            answer_text, source_chunks,
        )
        grounding_score = hallucination_result.get("grounding_score", 0.0)

        # 2. Attribution check (if enabled)
        attribution_score = 1.0
        attribution_result: dict[str, Any] = {"citations": [], "attribution_score": 1.0}
        if self._config.attribution_check_enabled:
            attribution_result = self._attribution_checker.check(
                answer_text, citations, source_chunks,
            )
            attribution_score = attribution_result.get("attribution_score", 0.0)

        # 3. Compute composite quality score
        quality_score = (
            GROUNDING_WEIGHT * grounding_score
            + ATTRIBUTION_WEIGHT * attribution_score
        )

        # 4. Determine acceptability
        is_acceptable = quality_score >= self._config.quality_threshold

        # 5. Generate revision feedback if not acceptable
        revision_feedback: str | None = None
        if not is_acceptable:
            revision_feedback = self._build_revision_feedback(
                hallucination_result, attribution_result,
            )

        # 6. Store validation result
        validation_data = {
            "quality_score": quality_score,
            "grounding_score": grounding_score,
            "attribution_score": attribution_score,
            "is_acceptable": is_acceptable,
            "hallucination_claims": hallucination_result.get("claims", []),
            "attribution_citations": attribution_result.get("citations", []),
            "revision_feedback": revision_feedback,
            "query_text": query_text,
        }
        output_ref = await self._state.store_result(
            task_id, "validation", validation_data,
        )

        logger.info(
            "task_id=%s: validation complete — quality=%.2f, grounding=%.2f, "
            "attribution=%.2f, acceptable=%s",
            task_id,
            quality_score,
            grounding_score,
            attribution_score,
            is_acceptable,
        )

        return {
            "output_ref": output_ref,
            "quality_score": quality_score,
            "is_acceptable": is_acceptable,
            "revision_feedback": revision_feedback,
        }

    @staticmethod
    def _build_revision_feedback(
        hallucination_result: dict[str, Any],
        attribution_result: dict[str, Any],
    ) -> str:
        """Build actionable feedback string for synthesis retry.

        Summarizes ungrounded claims and invalid citations so the
        synthesis agent can correct the answer.
        """
        parts: list[str] = []

        # Ungrounded claims
        ungrounded = [
            c["text"]
            for c in hallucination_result.get("claims", [])
            if not c.get("is_grounded", False)
        ]
        if ungrounded:
            claims_str = "; ".join(ungrounded[:5])  # Limit to 5 for prompt size
            parts.append(f"Claims not grounded: {claims_str}")

        # Invalid citations
        invalid = [
            c
            for c in attribution_result.get("citations", [])
            if not c.get("is_valid", False)
        ]
        if invalid:
            indices = [str(c.get("citation_index", "?")) for c in invalid]
            parts.append(f"Citations invalid: [{', '.join(indices)}]")

        return ". ".join(parts) if parts else "Quality score below threshold"
