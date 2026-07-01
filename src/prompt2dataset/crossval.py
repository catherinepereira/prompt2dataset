"""Find likely mislabeled images by out-of-fold cross-validation.

`p2d train` only checks its held-out validation split, so images in the training split
never get a label check. Cross-validation closes that gap: the dataset is split into k
folds, and each image is predicted by a model trained on the other k-1 folds, so every
image is judged by a model that never saw it. The disagreements are written to
misclassified.json, the same file `p2d review --misclassified` reads.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.rule import Rule

from prompt2dataset.train import (
    BATCH_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    _build_model,
    _find_lr,
)
from prompt2dataset.utils import meta_dir

log = logging.getLogger(__name__)
console = Console()


def _collect_samples(dataset_root: Path, items: list) -> list[tuple[Path, str]]:
    samples = [
        (dataset_root / i.local_path, i.label)
        for i in items
        if (dataset_root / i.local_path).exists()
    ]
    if not samples:
        raise click.ClickException("No images found on disk. Run `p2d add` first.")
    return samples


def _fold_indices(n: int, k: int) -> list[list[int]]:
    """Split indices 0..n-1 into k contiguous folds of near-equal size."""
    folds: list[list[int]] = []
    start = 0
    for f in range(k):
        size = n // k + (1 if f < n % k else 0)
        folds.append(list(range(start, start + size)))
        start += size
    return folds


def _build_transforms(img_size: int):
    import torchvision.transforms as T

    normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    train_tf = T.Compose([
        T.RandomResizedCrop(img_size),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToTensor(),
        normalize,
    ])
    eval_tf = T.Compose([
        T.Resize(int(img_size * 1.14)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        normalize,
    ])
    return train_tf, eval_tf


def _make_dataset(samples, class_to_idx, transform, img_size):
    from torch.utils.data import Dataset as TorchDataset
    from PIL import Image, UnidentifiedImageError

    class _ImageDataset(TorchDataset):
        def __len__(self):
            return len(samples)

        def __getitem__(self, idx):
            path, label = samples[idx]
            try:
                img = Image.open(path).convert("RGB")
            except (UnidentifiedImageError, OSError):
                img = Image.new("RGB", (img_size, img_size))
            return transform(img), class_to_idx[label]

    return _ImageDataset()


def _train_one_fold(train_samples, class_to_idx, epochs, img_size, device, lr):
    """Train a fresh model on one fold's training split at the given lr."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    train_tf, _ = _build_transforms(img_size)
    loader = DataLoader(
        _make_dataset(train_samples, class_to_idx, train_tf, img_size),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True,
    )
    model = _build_model("mobilenet_v2", len(class_to_idx)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model.train()
    for _ in range(epochs):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
    return model


def _find_lr_once(samples, class_to_idx, img_size, device) -> float:
    """Find a learning rate once on the full dataset, reused for every fold.

    The LR-finder range test is expensive, and a per-fold sweep would pay it k times
    for a value that barely moves between folds.
    """
    import torch.nn as nn
    from torch.utils.data import DataLoader

    train_tf, _ = _build_transforms(img_size)
    loader = DataLoader(
        _make_dataset(samples, class_to_idx, train_tf, img_size),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True,
    )
    model = _build_model("mobilenet_v2", len(class_to_idx)).to(device)
    return _find_lr(model, loader, device, nn.CrossEntropyLoss())


def _predict_fold(model, val_samples, class_to_idx, img_size, device) -> list[int]:
    """Predicted class index for each held-out sample."""
    import torch
    from torch.utils.data import DataLoader

    _, eval_tf = _build_transforms(img_size)
    loader = DataLoader(
        _make_dataset(val_samples, class_to_idx, eval_tf, img_size),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True,
    )
    model.eval()
    preds: list[int] = []
    with torch.no_grad():
        for images, _labels in loader:
            preds.extend(model(images.to(device)).argmax(dim=1).cpu().tolist())
    return preds


def _crossval(
    dataset_root: Path,
    items: list,
    folds: int,
    epochs: int,
    img_size: int,
    seed: int | None = None,
) -> list[dict]:
    import random
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[dim]Device:[/] {device.type.upper()}")

    samples = _collect_samples(dataset_root, items)
    if len(samples) < folds:
        raise click.ClickException(
            f"Need at least {folds} images for {folds}-fold cross-validation, "
            f"found {len(samples)}. Lower --folds or add more images."
        )

    labels = sorted({label for _, label in samples})
    class_to_idx = {label: i for i, label in enumerate(labels)}
    idx_to_class = {i: label for label, i in class_to_idx.items()}

    rng = random.Random(seed)
    rng.shuffle(samples)
    fold_idx = _fold_indices(len(samples), folds)
    console.print(f"[dim]Classes:[/] {len(labels)}   [dim]Images:[/] {len(samples)}   [dim]Folds:[/] {folds}\n")

    with console.status("[bold]Finding learning rate...[/]", spinner="dots"):
        lr = _find_lr_once(samples, class_to_idx, img_size, device)
    console.print(f"[green]v[/] Learning rate: [bold]{lr:.2e}[/]\n")

    misclassified: list[dict] = []
    # A class can land entirely in one fold, leaving that fold's training split with no
    # examples of it. The model can't predict a class it never trained on, so those
    # held-out images would all read as mislabeled. Skip them rather than emit false
    # positives, and report how many were skipped
    skipped = 0
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[cyan]{task.completed}[/]/[cyan]{task.total}[/] folds"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("  Cross-validating", total=folds)
        for f in range(folds):
            val_ids = set(fold_idx[f])
            val_samples = [samples[i] for i in fold_idx[f]]
            train_samples = [s for i, s in enumerate(samples) if i not in val_ids]
            train_classes = {label for _, label in train_samples}
            model = _train_one_fold(train_samples, class_to_idx, epochs, img_size, device, lr)
            preds = _predict_fold(model, val_samples, class_to_idx, img_size, device)
            for (path, true_label), pred in zip(val_samples, preds):
                if true_label not in train_classes:
                    skipped += 1
                    continue
                pred_label = idx_to_class[pred]
                if pred_label != true_label:
                    misclassified.append({
                        "path": str(path.resolve()),
                        "true_label": true_label,
                        "predicted": pred_label,
                    })
            progress.advance(task)

    if skipped:
        console.print(
            f"[yellow]![/] Skipped {skipped} images whose class was absent from a fold's "
            f"training split. Add more images per class or lower --folds to check them.\n"
        )

    (meta_dir(dataset_root) / "misclassified.json").write_text(
        json.dumps(misclassified, indent=2), encoding="utf-8"
    )
    return misclassified


@click.command("crossval")
@click.option("--folds", default=5, show_default=True, help="Number of cross-validation folds")
@click.option("--epochs", default=5, show_default=True, help="Training epochs per fold")
@click.option("--img-size", default=224, show_default=True, help="Input image size (square)")
@click.option("--seed", default=None, type=int, help="Seed the fold shuffle for reproducible runs")
def crossval_cmd(folds: int, epochs: int, img_size: int, seed: int | None) -> None:
    """Find likely mislabeled images by out-of-fold cross-validation.

    Trains one model per fold on the other folds and predicts the held-out one, so every
    image gets a prediction from a model that never saw it. Writes the disagreements to
    misclassified.json, which `p2d review --misclassified` steps through.
    """
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
    except ImportError:
        raise click.ClickException(
            "torch and torchvision are required. Install with:\n"
            '  pip install "prompt2dataset[train]"'
        )

    from prompt2dataset.models import Dataset

    dataset_root = Path.cwd()
    manifest = meta_dir(dataset_root) / "manifest.json"
    if not manifest.exists():
        raise click.ClickException("No manifest found. Run `p2d add` first.")
    ds = Dataset.model_validate_json(manifest.read_text(encoding="utf-8"))
    if not ds.items:
        raise click.ClickException("No items in dataset. Run `p2d add` first.")

    console.print()
    console.print(Rule(f"[bold violet]p2d crossval[/] [dim]{dataset_root.name}[/]"))
    console.print(f"\n[dim]Folds:[/] {folds}   [dim]Epochs/fold:[/] {epochs}\n")

    misclassified = _crossval(dataset_root, ds.items, folds, epochs, img_size, seed)

    console.print()
    console.print(Panel(
        f"[bold]{len(misclassified)}[/] likely mislabeled images across the whole dataset.\n"
        f"[dim]misclassified.json[/] written to [dim]{meta_dir(dataset_root)}/[/]\n"
        f"Run [bold]p2d review --misclassified[/] to step through them.",
        title="[green]Cross-validation complete[/]",
        border_style="green",
    ))
