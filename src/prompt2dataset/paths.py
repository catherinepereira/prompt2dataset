"""Dataset on-disk layout.

A dataset is a directory holding <label>/ image folders and a metadata dir. The
metadata dir name is the single source of truth here so it can change in one place.
"""

from __future__ import annotations

from pathlib import Path

MANIFEST_DIR = ".p2d"


def meta_dir(dataset_root: Path) -> Path:
    d = dataset_root / MANIFEST_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def manifest_path(dataset_root: Path) -> Path:
    return meta_dir(dataset_root) / "manifest.json"
