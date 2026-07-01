"""Turn a subject or label into a filesystem-safe slug.

Strips punctuation and path separators, so a subject like "American Robin" or a hostile
"../etc" becomes a flat, safe folder name. The slug is the label used for folders,
filenames, and class names.
"""

from __future__ import annotations

import re
import unicodedata

FALLBACK_SLUG = "unlabeled"

# reserved on Windows as device names, a folder named for one can't be created
_WINDOWS_RESERVED = {
    "con", "prn", "aux", "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}


def slugify(text: str) -> str:
    # fold accents to ASCII so slugs stay portable across filesystems
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text).strip("-")
    text = text[:80]
    if not text or text in _WINDOWS_RESERVED:
        return FALLBACK_SLUG
    return text
