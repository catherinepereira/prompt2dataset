from __future__ import annotations

from pathlib import Path


def meta_dir(dataset_root: Path) -> Path:
    d = dataset_root / ".p2d"
    d.mkdir(exist_ok=True)
    return d
