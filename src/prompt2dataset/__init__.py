"""prompt2dataset: build labeled image datasets from a plain-English prompt.

Package-first API. Import what you need directly from prompt2dataset:

    from prompt2dataset import (
        load_dataset, save_dataset, generate, resolve_subjects,
        find_exact_duplicates, find_outliers, train, infer, crossval,
    )

The command-line interface (prompt2dataset.cli) is a thin layer over these.
"""

from __future__ import annotations

__version__ = "2.0.0"

from prompt2dataset.classify import (
    Prediction,
    crossval,
    find_mismatches,
    infer,
    model_exists,
    torch_available,
    train,
)
from prompt2dataset.clean import (
    apply_flags,
    find_exact_duplicates,
    find_outliers,
)
from prompt2dataset.download import download_file, extension_for, host_is_public
from prompt2dataset.ids import slugify
from prompt2dataset.models import Dataset, DatasetItem, ReviewStatus
from prompt2dataset.paths import MANIFEST_DIR, manifest_path, meta_dir
from prompt2dataset.pipeline import GenerateResult, generate, records_to_items
from prompt2dataset.progress import OnProgress, Progress
from prompt2dataset.resolver import resolve_subjects
from prompt2dataset.sources import (
    REGISTRY,
    SourceAdapter,
    fetch_all,
    register_source,
    source_names,
)
from prompt2dataset.store import load_dataset, prune_missing, save_dataset

__all__ = [
    "Dataset", "DatasetItem", "ReviewStatus",
    "load_dataset", "save_dataset", "prune_missing",
    "slugify", "meta_dir", "manifest_path", "MANIFEST_DIR",
    "resolve_subjects",
    "REGISTRY", "SourceAdapter", "register_source", "source_names", "fetch_all",
    "download_file", "extension_for", "host_is_public",
    "records_to_items", "generate", "GenerateResult",
    "find_exact_duplicates", "find_outliers", "apply_flags",
    "train", "infer", "crossval", "find_mismatches", "Prediction",
    "model_exists", "torch_available",
    "Progress", "OnProgress",
    "__version__",
]
