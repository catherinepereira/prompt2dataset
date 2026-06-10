"""Fine-tune a pretrained image classifier on a prompt2dataset dataset"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.rule import Rule
from rich.table import Table

from prompt2dataset.utils import meta_dir

log = logging.getLogger(__name__)
console = Console()

SUPPORTED_MODELS = ["mobilenet_v2", "resnet18", "resnet50"]
BATCH_SIZE = 32
LR_FINDER_ITERS = 100
LR_FINDER_EDGE_SKIP = 5
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _load_split(
    dataset_root: Path,
    items: list,
    val_split: float,
    img_size: int,
) -> tuple:
    import torch
    from torch.utils.data import DataLoader, Dataset as TorchDataset
    import torchvision.transforms as T
    from PIL import Image, UnidentifiedImageError

    samples: list[tuple[Path, str]] = []
    for item in items:
        path = dataset_root / item.local_path
        if path.exists():
            samples.append((path, item.label))

    if not samples:
        raise click.ClickException("No images found on disk. Run `p2d add` first.")

    labels = sorted({label for _, label in samples})
    class_to_idx = {label: i for i, label in enumerate(labels)}

    random.shuffle(samples)
    n_val = max(1, int(len(samples) * val_split))
    val_samples = samples[:n_val]
    train_samples = samples[n_val:]

    if not train_samples:
        raise click.ClickException(
            f"Not enough images to split ({len(samples)} total). "
            "Lower --val-split or add more images."
        )

    normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    train_tf = T.Compose([
        T.RandomResizedCrop(img_size),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToTensor(),
        normalize,
    ])
    val_tf = T.Compose([
        T.Resize(int(img_size * 1.14)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        normalize,
    ])

    class _ImageDataset(TorchDataset):
        def __init__(self, samples, transform):
            self.samples = samples
            self.transform = transform

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            path, label = self.samples[idx]
            try:
                img = Image.open(path).convert("RGB")
            except (UnidentifiedImageError, Exception):
                img = Image.new("RGB", (img_size, img_size))
            return self.transform(img), class_to_idx[label]

    train_loader = DataLoader(
        _ImageDataset(train_samples, train_tf),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        _ImageDataset(val_samples, val_tf),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True,
    )

    return train_loader, val_loader, val_samples, class_to_idx, len(train_samples), len(val_samples)


def _build_model(model_name: str, num_classes: int):
    import torch.nn as nn
    import torchvision.models as models

    if model_name == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.DEFAULT
        model = models.mobilenet_v2(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    elif model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise click.ClickException(f"Unknown model {model_name!r}. Choose from: {SUPPORTED_MODELS}")

    return model


def _find_lr(model, train_loader, device, criterion) -> float:
    import io
    import logging
    import torch
    import numpy as np
    from torch_lr_finder import LRFinder

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-7)
    finder = LRFinder(model, optimizer, criterion, device=device)

    # suppress tqdm/print from the finder without affecting Rich's spinner
    import sys
    devnull = open(os.devnull, "w")
    old_stdout, old_stderr = sys.stdout, sys.stderr
    logging.disable(logging.CRITICAL)
    sys.stdout = sys.stderr = devnull
    try:
        finder.range_test(train_loader, end_lr=1.0, num_iter=LR_FINDER_ITERS, step_mode="exp")
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr
        devnull.close()
        logging.disable(logging.NOTSET)

    lrs = finder.history["lr"]
    losses = finder.history["loss"]
    finder.reset()

    lrs, losses = lrs[LR_FINDER_EDGE_SKIP:-LR_FINDER_EDGE_SKIP], losses[LR_FINDER_EDGE_SKIP:-LR_FINDER_EDGE_SKIP]
    if not lrs:
        return 1e-4
    grads = np.gradient(losses)
    return float(lrs[int(np.argmin(grads))])


def _train(
    dataset_root: Path,
    items: list,
    model_name: str,
    epochs: int,
    val_split: float,
    img_size: int,
) -> dict:
    import torch
    import torch.nn as nn

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[dim]Device:[/] {device.type.upper()}")

    train_loader, val_loader, val_samples, class_to_idx, n_train, n_val = _load_split(
        dataset_root, items, val_split, img_size
    )
    num_classes = len(class_to_idx)
    console.print(f"[dim]Classes:[/] {num_classes}   [dim]Train:[/] {n_train}   [dim]Val:[/] {n_val}\n")

    model = _build_model(model_name, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()

    with console.status("[bold]Finding learning rate...[/]", spinner="dots"):
        lr = _find_lr(model, train_loader, device, criterion)
    console.print(f"[green]v[/] Learning rate: [bold]{lr:.2e}[/]\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history: list[dict] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[cyan]{task.completed}[/]/[cyan]{task.total}[/] epochs"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("  Training", total=epochs)

        for epoch in range(1, epochs + 1):
            model.train()
            train_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                loss = criterion(model(images), labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(images)
            train_loss /= n_train

            model.eval()
            correct = 0
            all_preds: list[int] = []
            all_targets: list[int] = []
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    preds = model(images).argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    all_preds.extend(preds.cpu().tolist())
                    all_targets.extend(labels.cpu().tolist())

            val_acc = correct / n_val
            scheduler.step()
            history.append({"epoch": epoch, "train_loss": round(train_loss, 4), "val_acc": round(val_acc, 4)})
            progress.advance(task)

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    per_class: dict[str, dict] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    misclassified = []
    for (path, _), pred, target in zip(val_samples, all_preds, all_targets):
        cls = idx_to_class[target]
        pred_cls = idx_to_class[pred]
        if pred == target:
            per_class[cls]["tp"] += 1
        else:
            per_class[pred_cls]["fp"] += 1
            per_class[cls]["fn"] += 1
            misclassified.append({
                "path": str(path.resolve()),
                "true_label": cls,
                "predicted": pred_cls,
            })

    class_report = {}
    for cls, counts in per_class.items():
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        class_report[cls] = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
        }

    overall_acc = sum(p == t for p, t in zip(all_preds, all_targets)) / len(all_targets)

    report = {
        "model": model_name,
        "epochs": epochs,
        "lr": lr,
        "val_split": val_split,
        "n_train": n_train,
        "n_val": n_val,
        "overall_accuracy": round(overall_acc, 4),
        "per_class": class_report,
        "history": history,
        "trained_at": time.time(),
    }

    model.eval()
    scripted = torch.jit.script(model)
    md = meta_dir(dataset_root)
    scripted.save(str(md / "model.pt"))

    (md / "labels.json").write_text(
        json.dumps([idx_to_class[i] for i in range(num_classes)], indent=2),
        encoding="utf-8",
    )
    (md / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (md / "misclassified.json").write_text(
        json.dumps(misclassified, indent=2), encoding="utf-8"
    )

    return report, misclassified


@click.command("train")
@click.option("--epochs", default=5, show_default=True, help="Number of training epochs")
@click.option("--val-split", default=0.2, show_default=True, help="Fraction of images held out for validation")
@click.option("--img-size", default=224, show_default=True, help="Input image size (square)")
@click.option("--model", "model_name", default="mobilenet_v2", show_default=True,
              type=click.Choice(SUPPORTED_MODELS), help="Pretrained backbone")
def train_cmd(epochs: int, val_split: float, img_size: int, model_name: str) -> None:
    """Fine-tune a pretrained image classifier on the current dataset.

    Trains on all downloaded images across every subject and writes
    model.pt, labels.json, and report.json into the .fieldwork directory.
    Learning rate is found automatically via the Leslie Smith range test.
    """
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
    except ImportError:
        raise click.ClickException(
            "torch and torchvision are required for training.\n"
            "Install the CUDA build with:\n"
            "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126"
        )

    from prompt2dataset.models import Dataset, MediaType

    dataset_root = Path.cwd()
    md = meta_dir(dataset_root)
    manifest = md / "manifest.json"
    if not manifest.exists():
        raise click.ClickException("No manifest found. Run `p2d add` first.")

    ds = Dataset.model_validate_json(manifest.read_text(encoding="utf-8"))

    if ds.media_type == MediaType.video:
        raise click.ClickException("p2d train supports image datasets only.")

    if not ds.items:
        raise click.ClickException("No items in dataset. Run `p2d add` first.")

    console.print()
    console.print(Rule(f"[bold violet]p2d train[/] [dim]{dataset_root.name}[/]"))
    console.print(f"\n[dim]Backbone:[/] {model_name}   [dim]Epochs:[/] {epochs}\n")

    report, misclassified = _train(dataset_root, ds.items, model_name, epochs, val_split, img_size)

    console.print()
    table = Table(title="Per-class results", show_lines=True)
    table.add_column("Subject", style="cyan")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("F1", justify="right")
    for cls, metrics in sorted(report["per_class"].items()):
        f1 = metrics["f1"]
        style = "red" if f1 < 0.5 else ("yellow" if f1 < 0.75 else "")
        table.add_row(cls, str(metrics["precision"]), str(metrics["recall"]), str(f1), style=style)
    console.print(table)

    console.print()
    console.print(Panel(
        f"[bold]Overall accuracy:[/] {report['overall_accuracy']:.1%}   "
        f"[bold]Val images:[/] {report['n_val']}   "
        f"[bold]Misclassified:[/] {len(misclassified)}\n"
        f"[dim]model.pt  labels.json  report.json  misclassified.json[/] written to [dim]{md}/[/]\n"
        f"Run [bold]fieldwork review --misclassified[/] to step through misclassified images.",
        title="[green]Training complete[/]",
        border_style="green",
    ))
