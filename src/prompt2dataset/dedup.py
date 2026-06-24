"""
Image filters that clean a dataset by content rather than URL.

Two passes, both grouped by label:
- exact-duplicate removal by hashing decoded pixels (no torch)
- outlier removal via CNN embeddings and DBSCAN (needs the [train] extra)
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import click
import numpy as np
from PIL import Image, UnidentifiedImageError
from rich.console import Console
from rich.table import Table

from prompt2dataset.models import Dataset, DatasetItem, ReviewStatus

log = logging.getLogger(__name__)
console = Console()

# DBSCAN neighborhood radius in cosine-distance space, and the minimum
# neighbors to form a cluster. Points outside any cluster are the outliers.
DEFAULT_OUTLIER_EPS = 0.25
OUTLIER_MIN_SAMPLES = 3

EMBED_IMG_SIZE = 224


def _pixel_hash(path: Path) -> str | None:
    """SHA-256 of the decoded RGB pixels, or None if the file can't be read.

    Hashing decoded pixels rather than file bytes makes the same image collide
    even when re-encoded or saved under a different format or filename.
    """
    try:
        img = Image.open(path).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        log.warning("Could not read %s: %s", path, exc)
        return None
    return hashlib.sha256(img.tobytes()).hexdigest()


def _group_by_label(items: list[DatasetItem]) -> dict[str, list[DatasetItem]]:
    groups: dict[str, list[DatasetItem]] = {}
    for item in items:
        groups.setdefault(item.label, []).append(item)
    return groups


def _find_exact_duplicates(
    items: list[DatasetItem],
    dataset_root: Path,
) -> list[DatasetItem]:
    """Within each label, keep the first of any pixel-identical set, flag the rest."""
    flagged: list[DatasetItem] = []
    for label, group in _group_by_label(items).items():
        seen: set[str] = set()
        for item in group:
            h = _pixel_hash(dataset_root / item.local_path)
            if h is None:
                continue
            if h in seen:
                flagged.append(item)
            else:
                seen.add(h)
    return flagged


def _load_embedder():
    """Build the MobileNetV2 feature extractor and its preprocessing transform.

    Imports torch lazily so the rest of the module works without the extra.
    """
    import torch
    import torchvision.models as models
    import torchvision.transforms as T

    weights = models.MobileNet_V2_Weights.DEFAULT
    net = models.mobilenet_v2(weights=weights)
    net.classifier = torch.nn.Identity()
    net.eval()

    tf = T.Compose([
        T.Resize(EMBED_IMG_SIZE),
        T.CenterCrop(EMBED_IMG_SIZE),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return net, tf


def _embed(paths: list[Path], net, tf) -> np.ndarray:
    """L2-normalized feature vectors, one row per path.

    Unreadable images get a zero row so the output stays aligned with paths.
    """
    import torch

    batch = []
    for path in paths:
        try:
            img = Image.open(path).convert("RGB")
            batch.append(tf(img))
        except (UnidentifiedImageError, OSError) as exc:
            log.warning("Could not embed %s: %s", path, exc)
            batch.append(torch.zeros(3, EMBED_IMG_SIZE, EMBED_IMG_SIZE))

    with torch.no_grad():
        feats = net(torch.stack(batch))
    feats = torch.nn.functional.normalize(feats, dim=1)
    return feats.numpy()


def _find_outliers(
    items: list[DatasetItem],
    dataset_root: Path,
    eps: float,
) -> list[DatasetItem]:
    """Within each label, flag images DBSCAN can't place in any dense cluster.

    Embeddings are compared by cosine distance. An image whose neighborhood is
    too sparse to join a cluster is labeled noise, which is the outlier. Groups
    of three or fewer are skipped, since a cluster needs more support than that.
    """
    from sklearn.cluster import DBSCAN

    groups = {
        label: group
        for label, group in _group_by_label(items).items()
        if len(group) > OUTLIER_MIN_SAMPLES
    }
    if not groups:
        return []

    net, tf = _load_embedder()
    flagged: list[DatasetItem] = []
    for group in groups.values():
        paths = [dataset_root / i.local_path for i in group]
        feats = _embed(paths, net, tf)
        labels = DBSCAN(eps=eps, min_samples=OUTLIER_MIN_SAMPLES, metric="cosine").fit_predict(feats)
        for cluster, item in zip(labels, group):
            if cluster == -1:
                flagged.append(item)
    return flagged


def _apply(
    flagged: list[DatasetItem],
    ds: Dataset,
    dataset_root: Path,
    delete: bool,
) -> None:
    """Mark flagged items invalid, or remove them from disk and manifest if delete."""
    flagged_ids = {item.item_id for item in flagged}
    if delete:
        for item in flagged:
            path = dataset_root / item.local_path
            if path.exists():
                path.unlink()
        ds.items = [i for i in ds.items if i.item_id not in flagged_ids]
    else:
        for item in ds.items:
            if item.item_id in flagged_ids:
                item.review_status = ReviewStatus.invalid
    ds.touch()


def _load_image_dataset(dataset_root: Path):
    """Load the dataset and return it with the candidate items to check.

    Candidates are images that exist on disk and aren't already invalid.
    """
    from prompt2dataset.ingest import load_dataset

    ds = load_dataset(dataset_root)
    candidates = [
        i for i in ds.items
        if i.review_status != ReviewStatus.invalid
        and (dataset_root / i.local_path).exists()
    ]
    return ds, candidates


def _report(flagged: list[DatasetItem], ds: Dataset, dataset_root: Path, delete: bool, noun: str) -> None:
    """Apply the flag, save, and print a one-line result."""
    from prompt2dataset.ingest import save_dataset

    _apply(flagged, ds, dataset_root, delete)
    save_dataset(ds, dataset_root)
    if delete:
        console.print(f"[green]v[/] Deleted {len(flagged)} {noun}")
    else:
        console.print(
            f"[green]v[/] Flagged {len(flagged)} {noun} as invalid. "
            "Run [cyan]p2d review[/] to inspect."
        )


@click.command(name="dedup")
@click.option("--delete", is_flag=True, help="Delete flagged files instead of marking them invalid.")
def dedup_cmd(delete: bool) -> None:
    """Remove exact-duplicate images from the current dataset.

    Duplicates are found by hashing decoded pixels, so the same image under a
    different filename or format is caught. By default flagged images are marked
    invalid in the manifest so you can review them, pass --delete to remove the
    files.
    """
    dataset_root = Path.cwd()
    ds, candidates = _load_image_dataset(dataset_root)
    if not candidates:
        console.print("[yellow]No images on disk to check.[/]")
        return

    with console.status("[bold]Hashing pixels...[/]", spinner="dots"):
        dupes = _find_exact_duplicates(candidates, dataset_root)

    if not dupes:
        console.print("[green]No duplicates found.[/]")
        return

    _report(dupes, ds, dataset_root, delete, "duplicates")


@click.command(name="outliers")
@click.option(
    "--eps",
    default=DEFAULT_OUTLIER_EPS,
    show_default=True,
    help="DBSCAN cosine-distance radius. Lower flags more images.",
)
@click.option("--delete", is_flag=True, help="Delete flagged files instead of marking them invalid.")
def outliers_cmd(eps: float, delete: bool) -> None:
    """Remove outlier images that don't fit the rest of their subject.

    Each image is embedded with a pretrained CNN, then DBSCAN flags those that
    don't cluster with the others (scraping junk like charts or text-on-white).
    Needs the [train] extra. By default flagged images are marked invalid in the
    manifest so you can review them, pass --delete to remove the files.
    """
    dataset_root = Path.cwd()
    ds, candidates = _load_image_dataset(dataset_root)
    if not candidates:
        console.print("[yellow]No images on disk to check.[/]")
        return

    try:
        with console.status("[bold]Embedding images...[/]", spinner="dots"):
            outliers = _find_outliers(candidates, dataset_root, eps)
    except ImportError as exc:
        raise click.ClickException(
            f"The outlier pass needs PyTorch and scikit-learn ({exc.name} missing). "
            'Install them with: pip install "prompt2dataset[train]".'
        ) from exc

    if not outliers:
        console.print("[green]No outliers found.[/]")
        return

    _report(outliers, ds, dataset_root, delete, "outliers")
