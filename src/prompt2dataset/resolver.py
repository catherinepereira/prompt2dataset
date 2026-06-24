"""Resolves a dataset description into a list of queryable subjects"""

from __future__ import annotations

import json
import logging
import os
import re

import httpx

log = logging.getLogger(__name__)

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("P2D_MODEL", "qwen2.5:3b-instruct")


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
- Remove duplicates. Keep ordering logical (taxonomic, alphabetical, etc.)
"""


def resolve_subjects(prompt: str, model: str = DEFAULT_MODEL) -> list[str]:
    resp = httpx.post(
        f"{OLLAMA_HOST}/api/chat",
        json={
            "model": model,
            "stream": False,
            "options": {"temperature": 0},
            "messages": [
                {"role": "system", "content": _SUBJECT_SYSTEM},
                {"role": "user", "content": prompt},
            ],
        },
        timeout=120,
    )
    resp.raise_for_status()
    subjects = _parse_json_array(resp.json()["message"]["content"])
    return [s.strip() for s in subjects if isinstance(s, str) and s.strip()]
