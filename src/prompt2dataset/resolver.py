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
JSON array of subject names that image search engines can find lots of photos for.

Rules:
- Return ONLY a JSON array of strings, no commentary or markdown
- Use the common name people would type into an image search, not a scientific or
  taxonomic one: "American Robin" not "Turdus migratorius", "Golden Retriever" not
  "Canis lupus familiaris"
- Make each name specific enough to be one class, but no more formal than a search
  needs: "American Robin" over both "robin" and "Turdus migratorius"
- Remove duplicates. Keep the ordering logical (alphabetical or by theme)
"""


def _count_hint(count: int | None) -> str:
    if count and count > 0:
        return f"\nReturn about {count} subjects."
    return ""


def resolve_subjects(
    prompt: str,
    model: str = DEFAULT_MODEL,
    count: int | None = None,
    exclude: list[str] | None = None,
) -> list[str]:
    """Resolve a description into searchable subject names.

    count hints how many to return (the caller still caps the final list). exclude
    lists subjects already chosen, so a follow-up call returns only new ones.
    """
    user = prompt + _count_hint(count)
    if exclude:
        joined = ", ".join(exclude)
        user += f"\nDo not include any of these, they are already chosen: {joined}."

    resp = httpx.post(
        f"{OLLAMA_HOST}/api/chat",
        json={
            "model": model,
            "stream": False,
            "options": {"temperature": 0},
            "messages": [
                {"role": "system", "content": _SUBJECT_SYSTEM},
                {"role": "user", "content": user},
            ],
        },
        timeout=120,
    )
    resp.raise_for_status()
    subjects = _parse_json_array(resp.json()["message"]["content"])
    cleaned = [s.strip() for s in subjects if isinstance(s, str) and s.strip()]

    if exclude:
        seen = {e.strip().lower() for e in exclude}
        cleaned = [s for s in cleaned if s.lower() not in seen]
    return cleaned
