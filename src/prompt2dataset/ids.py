"""Turn a subject or label into a filesystem-safe slug.

Strips punctuation and path separators, so a subject like "American Robin" or a hostile
"../etc" becomes a flat, safe folder name. The slug is the label used for folders,
filenames, and class names.
"""

from __future__ import annotations

import re


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    return text[:80]
