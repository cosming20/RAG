"""LLM output parsing utilities."""

from __future__ import annotations

import json
import re
from typing import Any


def parse_llm_json(text: str, default: Any = None) -> Any:
    """Parse JSON from LLM output, stripping markdown code fences if present."""
    cleaned = text.strip()
    # Strip markdown code fences
    cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
    cleaned = re.sub(r"\n?```\s*$", "", cleaned)
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        if default is not None:
            return default
        raise
