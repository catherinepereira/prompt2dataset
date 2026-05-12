"""Resolves a dataset description into a list of queryable subjects"""

from __future__ import annotations

import json
import logging
import os
import re

import anthropic

log = logging.getLogger(__name__)

_CLIENT = None


def _client() -> anthropic.Anthropic:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    return _CLIENT


def _parse_json_array(raw: str) -> list:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        result = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Expected JSON array, got: {raw!r}") from exc
    if not isinstance(result, list):
        raise ValueError(f"Expected a JSON array, got {type(result).__name__}")
    return result


_SUBJECT_SYSTEM = """\
You are a dataset subject extractor. Read a dataset description and return a
JSON array of specific, searchable subject names that data collection tools
can query against external databases.

Rules:
- Return ONLY a JSON array of strings, no commentary or markdown
- Each string must be a concrete, queryable name (scientific or common is fine)
- Prefer specificity: "Turdus migratorius" or "American Robin" over just "robin"
- Remove duplicates; keep ordering logical (taxonomic, alphabetical, etc.)
"""


def resolve_subjects(prompt: str, model: str = "claude-sonnet-4-6") -> list[str]:
    resp = _client().messages.create(
        model=model,
        max_tokens=8096,
        temperature=0,
        system=_SUBJECT_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    subjects = _parse_json_array(resp.content[0].text)
    return [s.strip() for s in subjects if isinstance(s, str) and s.strip()]
