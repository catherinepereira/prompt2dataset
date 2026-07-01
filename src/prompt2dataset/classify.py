"""Train an image classifier and use it to find likely mislabeled images.

Three entry points, all torch-lazy so importing this module needs no torch:
- train: fine-tune a pretrained backbone on the dataset, save a TorchScript model
- infer: predict every image with the saved model, return predicted != label
- crossval: out-of-fold prediction, so every image is judged by a model that never
  trained on it, which checks the whole dataset rather than only a held-out split

train and crossval write misclassified.json, which review consumers read
"""

from __future__ import annotations

import json
import logging
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from prompt2dataset.models import DatasetItem
from prompt2dataset.paths import meta_dir
from prompt2dataset.progress import OnProgress, Reporter

log = logging.getLogger(__name__)

SUPPORTED_MODELS = ["mobilenet_v2", "resnet18", "resnet50"]
BATCH_SIZE = 32
LR_FINDER_ITERS = 100
LR_FINDER_EDGE_SKIP = 5
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass(frozen=True)
class Prediction:
    item_id: str
    label: str
    subject: str
    predicted: str
    local_path: str


def torch_available() -> bool:
    from importlib.util import find_spec

    return find_spec("torch") is not None and find_spec("torchvision") is not None


def model_exists(dataset_root: Path) -> bool:
    return (meta_dir(dataset_root) / "model.pt").exists()


# ----- shared transforms + data -----

def _transforms(img_size: int):
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
    from PIL import Image

    from prompt2dataset.images import DecodeError, open_rgb

    class _ImageDataset(TorchDataset):
        def __len__(self):
            return len(samples)

        def __getitem__(self, idx):
            path, label = samples[idx]
            try:
                img = open_rgb(path)
            except DecodeError:
                img = Image.new("RGB", (img_size, img_size))
            return transform(img), class_to_idx[label]

    return _ImageDataset()


def _build_model(model_name: str, num_classes: int):
    import torch.nn as nn
    import torchvision.models as models

    if model_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unknown model {model_name!r}. Choose from {SUPPORTED_MODELS}")
    return model


def _find_lr(model, train_loader, device, criterion) -> float:
    import logging as _logging
    import os
    import sys

    import numpy as np
    import torch
    from torch_lr_finder import LRFinder

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-7)
    finder = LRFinder(model, optimizer, criterion, device=device)

    devnull = open(os.devnull, "w")
    old_stdout, old_stderr = sys.stdout, sys.stderr
    _logging.disable(_logging.CRITICAL)
    sys.stdout = sys.stderr = devnull
    try:
        finder.range_test(train_loader, end_lr=1.0, num_iter=LR_FINDER_ITERS, step_mode="exp")
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr
        devnull.close()
        _logging.disable(_logging.NOTSET)

    lrs = finder.history["lr"][LR_FINDER_EDGE_SKIP:-LR_FINDER_EDGE_SKIP]
    losses = finder.history["loss"][LR_FINDER_EDGE_SKIP:-LR_FINDER_EDGE_SKIP]
    finder.reset()
    if not lrs:
        return 1e-4
    return float(lrs[int(np.argmin(np.gradient(losses)))])


def _samples_on_disk(dataset_root: Path, items: list[DatasetItem]) -> list[tuple[Path, str]]:
    return [
        (dataset_root / i.local_path, i.label)
        for i in items
        if (dataset_root / i.local_path).exists()
    ]


# ----- train -----

def train(
    dataset_root: Path,
    items: list[DatasetItem],
    *,
    model: str = "mobilenet_v2",
    epochs: int = 5,
    val_split: float = 0.2,
    img_size: int = 224,
    on_progress: OnProgress = None,
) -> dict:
    """Fine-tune a backbone on the dataset and save model.pt, return a report."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    reporter = Reporter(on_progress)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples = _samples_on_disk(dataset_root, items)
    if not samples:
        raise ValueError("No images found on disk.")
    labels = sorted({label for _, label in samples})
    class_to_idx = {label: i for i, label in enumerate(labels)}
    idx_to_class = {i: label for label, i in class_to_idx.items()}

    random.shuffle(samples)
    n_val = max(1, int(len(samples) * val_split))
    val_samples, train_samples = samples[:n_val], samples[n_val:]
    if not train_samples:
        raise ValueError(f"Not enough images to split ({len(samples)} total).")

    train_tf, eval_tf = _transforms(img_size)
    train_loader = DataLoader(
        _make_dataset(train_samples, class_to_idx, train_tf, img_size),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        _make_dataset(val_samples, class_to_idx, eval_tf, img_size),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True,
    )

    net = _build_model(model, len(labels)).to(device)
    criterion = nn.CrossEntropyLoss()
    reporter.start(epochs, "Finding learning rate")
    lr = _find_lr(net, train_loader, device, criterion)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    all_preds: list[int] = []
    all_targets: list[int] = []
    for epoch in range(1, epochs + 1):
        net.train()
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            criterion(net(images), targets).backward()
            optimizer.step()
        scheduler.step()
        reporter.advance(f"Epoch {epoch}/{epochs}")

    net.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for images, targets in val_loader:
            preds = net(images.to(device)).argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(targets.tolist())

    report = _report_and_save(
        net, model, epochs, lr, val_split, len(train_samples), len(val_samples),
        val_samples, all_preds, all_targets, idx_to_class, dataset_root,
    )
    return report


def _report_and_save(net, model_name, epochs, lr, val_split, n_train, n_val,
                     val_samples, all_preds, all_targets, idx_to_class, dataset_root):
    import torch

    per_class: dict[str, dict] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    misclassified = []
    for (path, _), pred, target in zip(val_samples, all_preds, all_targets):
        cls, pred_cls = idx_to_class[target], idx_to_class[pred]
        if pred == target:
            per_class[cls]["tp"] += 1
        else:
            per_class[pred_cls]["fp"] += 1
            per_class[cls]["fn"] += 1
            misclassified.append({"path": str(path.resolve()), "true_label": cls, "predicted": pred_cls})

    class_report = {}
    for cls, c in per_class.items():
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        class_report[cls] = {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3)}

    overall = sum(p == t for p, t in zip(all_preds, all_targets)) / len(all_targets)
    report = {
        "model": model_name, "epochs": epochs, "lr": lr, "val_split": val_split,
        "n_train": n_train, "n_val": n_val, "overall_accuracy": round(overall, 4),
        "per_class": class_report, "trained_at": time.time(),
    }

    md = meta_dir(dataset_root)
    net.eval()
    torch.jit.script(net).save(str(md / "model.pt"))
    num_classes = len(idx_to_class)
    (md / "labels.json").write_text(
        json.dumps([idx_to_class[i] for i in range(num_classes)], indent=2), encoding="utf-8"
    )
    (md / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (md / "misclassified.json").write_text(json.dumps(misclassified, indent=2), encoding="utf-8")
    return report


# ----- full-dataset inference -----

def find_mismatches(items: list[DatasetItem], predictions: dict[str, str]) -> list[Prediction]:
    """Items whose predicted class differs from their label. Torch-free, stub-testable."""
    out: list[Prediction] = []
    for item in items:
        predicted = predictions.get(item.item_id)
        if predicted is None or predicted == item.label:
            continue
        out.append(Prediction(
            item_id=item.item_id, label=item.label,
            subject=item.subject or item.label, predicted=predicted,
            local_path=item.local_path,
        ))
    return out


def infer(
    dataset_root: Path,
    items: list[DatasetItem],
    *,
    on_progress: OnProgress = None,
) -> list[Prediction]:
    """Predict every image with the saved model, return predicted != label."""
    import torch

    from prompt2dataset.images import DecodeError, open_rgb

    reporter = Reporter(on_progress)
    md = meta_dir(dataset_root)
    labels: list[str] = json.loads((md / "labels.json").read_text(encoding="utf-8"))
    # jit.load runs the model's code, so it trusts this local model.pt, written by train
    net = torch.jit.load(str(md / "model.pt"))
    net.eval()

    _, eval_tf = _transforms(224)
    on_disk = [i for i in items if (dataset_root / i.local_path).exists()]
    predictions: dict[str, str] = {}
    reporter.start(max(len(on_disk), 1), "Running classifier")
    with torch.no_grad():
        for start in range(0, len(on_disk), BATCH_SIZE):
            chunk = on_disk[start:start + BATCH_SIZE]
            tensors, ids = [], []
            for item in chunk:
                try:
                    img = open_rgb(dataset_root / item.local_path)
                except DecodeError:
                    continue
                tensors.append(eval_tf(img))
                ids.append(item.item_id)
            if tensors:
                preds = net(torch.stack(tensors)).argmax(dim=1).tolist()
                for item_id, idx in zip(ids, preds):
                    if 0 <= idx < len(labels):
                        predictions[item_id] = labels[idx]
            reporter.advance(step=len(chunk), message=f"Classified {min(start + BATCH_SIZE, len(on_disk))}/{len(on_disk)}")
    return find_mismatches(on_disk, predictions)


# ----- cross-validation -----

def _fold_indices(n: int, k: int) -> list[list[int]]:
    """Split 0..n-1 into k contiguous folds of near-equal size."""
    folds, start = [], 0
    for f in range(k):
        size = n // k + (1 if f < n % k else 0)
        folds.append(list(range(start, start + size)))
        start += size
    return folds


def _find_lr_once(samples, class_to_idx, img_size, device) -> float:
    import torch.nn as nn
    from torch.utils.data import DataLoader

    train_tf, _ = _transforms(img_size)
    loader = DataLoader(
        _make_dataset(samples, class_to_idx, train_tf, img_size),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True,
    )
    model = _build_model("mobilenet_v2", len(class_to_idx)).to(device)
    return _find_lr(model, loader, device, nn.CrossEntropyLoss())


def _train_fold(train_samples, class_to_idx, epochs, img_size, device, lr):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    train_tf, _ = _transforms(img_size)
    loader = DataLoader(
        _make_dataset(train_samples, class_to_idx, train_tf, img_size),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True,
    )
    model = _build_model("mobilenet_v2", len(class_to_idx)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            criterion(model(images), targets).backward()
            optimizer.step()
        scheduler.step()
    return model


def _predict_fold(model, val_samples, class_to_idx, img_size, device) -> list[int]:
    import torch
    from torch.utils.data import DataLoader

    _, eval_tf = _transforms(img_size)
    loader = DataLoader(
        _make_dataset(val_samples, class_to_idx, eval_tf, img_size),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True,
    )
    model.eval()
    preds: list[int] = []
    with torch.no_grad():
        for images, _t in loader:
            preds.extend(model(images.to(device)).argmax(dim=1).cpu().tolist())
    return preds


def crossval(
    dataset_root: Path,
    items: list[DatasetItem],
    *,
    folds: int = 5,
    epochs: int = 5,
    img_size: int = 224,
    seed: int | None = None,
    on_progress: OnProgress = None,
) -> list[dict]:
    """Out-of-fold cross-validation. Writes and returns misclassified entries.

    A class small relative to folds can land entirely in one fold, leaving that fold's
    training split without it. Those images can't be predicted, so they're skipped
    rather than counted as mislabeled.
    """
    import torch

    reporter = Reporter(on_progress)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples = _samples_on_disk(dataset_root, items)
    if len(samples) < folds:
        raise ValueError(f"Need at least {folds} images for {folds}-fold, found {len(samples)}.")

    labels = sorted({label for _, label in samples})
    class_to_idx = {label: i for i, label in enumerate(labels)}
    idx_to_class = {i: label for label, i in class_to_idx.items()}

    random.Random(seed).shuffle(samples)
    fold_idx = _fold_indices(len(samples), folds)
    lr = _find_lr_once(samples, class_to_idx, img_size, device)

    misclassified: list[dict] = []
    skipped = 0
    reporter.start(folds, "Cross-validating")
    for f in range(folds):
        val_ids = set(fold_idx[f])
        val_samples = [samples[i] for i in fold_idx[f]]
        train_samples = [s for i, s in enumerate(samples) if i not in val_ids]
        train_classes = {label for _, label in train_samples}
        model = _train_fold(train_samples, class_to_idx, epochs, img_size, device, lr)
        preds = _predict_fold(model, val_samples, class_to_idx, img_size, device)
        for (path, true_label), pred in zip(val_samples, preds):
            if true_label not in train_classes:
                skipped += 1
                continue
            pred_label = idx_to_class[pred]
            if pred_label != true_label:
                misclassified.append({"path": str(path.resolve()), "true_label": true_label, "predicted": pred_label})
        reporter.advance(f"Fold {f + 1}/{folds}")

    if skipped:
        reporter.set_message(f"Skipped {skipped} images whose class was absent from a fold's training split")

    (meta_dir(dataset_root) / "misclassified.json").write_text(
        json.dumps(misclassified, indent=2), encoding="utf-8"
    )
    return misclassified
