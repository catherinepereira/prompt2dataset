"""Decode downloaded images without letting a hostile file exhaust memory.

Images come from the web, so a small file can decode to a gigapixel bitmap and OOM the
process. PIL only warns at its default threshold. MAX_PIXELS caps the pixel count and
open_rgb raises DecompressionBombError past it, which callers catch alongside the usual
unreadable-file errors.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

# raise past this many pixels, ~50MP covers any photo with room to spare
MAX_PIXELS = 50_000_000
Image.MAX_IMAGE_PIXELS = MAX_PIXELS

# the errors a decode can raise on untrusted input, for callers to catch as one
DecodeError = (Image.UnidentifiedImageError, Image.DecompressionBombError, OSError)


def open_rgb(path: Path) -> Image.Image:
    """Open path as RGB, raising DecodeError on an unreadable or oversized image."""
    return Image.open(path).convert("RGB")
