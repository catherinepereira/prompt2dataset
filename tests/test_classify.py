from __future__ import annotations

import pytest

from prompt2dataset.classify import Prediction, _fold_indices, find_mismatches
from prompt2dataset.models import DatasetItem


@pytest.mark.parametrize("n,k", [(10, 5), (11, 5), (13, 5), (6, 5), (7, 3), (5, 5), (5, 1), (2, 2)])
def test_fold_indices_partition(n, k):
    folds = _fold_indices(n, k)
    assert len(folds) == k
    flat = sorted(i for f in folds for i in f)
    assert flat == list(range(n))
    sizes = [len(f) for f in folds]
    assert max(sizes) - min(sizes) <= 1


def _item(item_id, label, subject=""):
    return DatasetItem(item_id=item_id, label=label, subject=subject,
                       source_url="u", local_path=f"{label}/{item_id}.png")


def test_find_mismatches_flags_disagreements():
    items = [_item("a", "robin", "American Robin"), _item("b", "sparrow")]
    preds = {"a": "sparrow", "b": "sparrow"}
    out = find_mismatches(items, preds)
    assert len(out) == 1
    assert out[0].item_id == "a" and out[0].predicted == "sparrow"
    assert out[0].subject == "American Robin"


def test_find_mismatches_falls_back_to_label_for_subject():
    items = [_item("a", "robin")]  # no subject
    out = find_mismatches(items, {"a": "sparrow"})
    assert out[0].subject == "robin"


def test_find_mismatches_skips_missing_and_correct():
    items = [_item("a", "robin"), _item("b", "sparrow")]
    assert find_mismatches(items, {}) == []
    assert find_mismatches(items, {"a": "robin"}) == []
