"""Content-based cleaning: find duplicate and outlier images, grouped by label.

find_exact_duplicates hashes decoded pixels, so the same image re-encoded or renamed is
caught. find_outliers embeds each image with a pretrained CNN and flags those DBSCAN
can't cluster with the rest of their label (scraping junk like charts or text). Both
return flagged items, apply_flags marks or removes them. Torch is imported lazily so the
duplicate pass works without it.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from prompt2dataset.models import Dataset, DatasetItem, ReviewStatus

log = logging.getLogger(__name__)

# DBSCAN neighborhood radius in cosine-distance space, and the minimum neighbors to
# form a cluster. Points outside any cluster are the outliers
DEFAULT_OUTLIER_EPS = 0.25
OUTLIER_MIN_SAMPLES = 3
EMBED_IMG_SIZE = 224


def _pixel_hash(path: Path) -> str | None:
    """SHA-256 of the decoded RGB pixels, or None if the file can't be read."""
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


def find_exact_duplicates(items: list[DatasetItem], dataset_root: Path) -> list[DatasetItem]:
    """Within each label, keep the first of any pixel-identical set, flag the rest."""
    flagged: list[DatasetItem] = []
    for group in _group_by_label(items).values():
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
    """MobileNetV2 feature extractor and its preprocessing transform (needs torch)."""
    import torch
    import torchvision.models as models
    import torchvision.transforms as T

    net = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
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
    """L2-normalized feature vectors, one row per path. Unreadable images get a zero row."""
    import torch

    batch = []
    for path in paths:
        try:
            batch.append(tf(Image.open(path).convert("RGB")))
        except (UnidentifiedImageError, OSError) as exc:
            log.warning("Could not embed %s: %s", path, exc)
            batch.append(torch.zeros(3, EMBED_IMG_SIZE, EMBED_IMG_SIZE))

    with torch.no_grad():
        feats = net(torch.stack(batch))
    return torch.nn.functional.normalize(feats, dim=1).numpy()


def find_outliers(
    items: list[DatasetItem],
    dataset_root: Path,
    eps: float = DEFAULT_OUTLIER_EPS,
) -> list[DatasetItem]:
    """Within each label, flag images DBSCAN can't place in any dense cluster.

    Groups of three or fewer are skipped, since a cluster needs more support than that.
    Needs torch and scikit-learn.
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
        flagged.extend(item for cluster, item in zip(labels, group) if cluster == -1)
    return flagged


def apply_flags(
    flagged: list[DatasetItem],
    ds: Dataset,
    dataset_root: Path,
    delete: bool = False,
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
