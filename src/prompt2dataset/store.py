"""Read and write a dataset's manifest and labels file.

The manifest (manifest.json) is the authoritative record. labels.csv is derived from it
for external tools. Neither touches the terminal, so any consumer can persist a dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from prompt2dataset.models import Dataset, DatasetItem
from prompt2dataset.paths import manifest_path, meta_dir


def load_dataset(dataset_root: Path) -> Dataset:
    path = manifest_path(dataset_root)
    if not path.exists():
        raise FileNotFoundError(f"No manifest found at {path}")
    return Dataset.model_validate_json(path.read_text(encoding="utf-8"))


def save_dataset(ds: Dataset, dataset_root: Path) -> None:
    dataset_root.mkdir(parents=True, exist_ok=True)
    md = meta_dir(dataset_root)
    (md / "manifest.json").write_text(ds.model_dump_json(indent=2), encoding="utf-8")
    _write_labels(ds, md)


def _write_labels(ds: Dataset, md: Path) -> None:
    lines = ["filename,label,subject,source"]
    for item in ds.items:
        source = item.meta.get("source", "unknown")
        subject = item.subject or item.label
        lines.append(f"{item.local_path},{item.label},{subject},{source}")
    (md / "labels.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def prune_missing(
    ds: Dataset,
    dataset_root: Path,
    keep: Optional[Callable[[DatasetItem], bool]] = None,
) -> int:
    """Drop items whose image is not on disk, returning how many were removed.

    A source hands back more URLs than download cleanly (dead links, hotlink 403s), and
    the manifest records an item per URL before the download runs. Without this, failed
    downloads linger as items pointing at files that were never written.

    keep is an optional predicate for items to retain even when their file is absent,
    for a caller that stores some files outside local_path (e.g. a recycle bin).
    """
    before = len(ds.items)
    ds.items = [
        i
        for i in ds.items
        if (keep is not None and keep(i)) or (dataset_root / i.local_path).exists()
    ]
    removed = before - len(ds.items)
    if removed:
        ds.touch()
    return removed
