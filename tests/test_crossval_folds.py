from __future__ import annotations

import pytest

from prompt2dataset.crossval import _fold_indices


@pytest.mark.parametrize(
    "n,k",
    [(10, 5), (11, 5), (13, 5), (6, 5), (9, 4), (7, 3), (100, 7), (5, 5), (5, 1), (2, 2)],
)
def test_fold_indices_partition(n, k):
    folds = _fold_indices(n, k)
    assert len(folds) == k
    flat = [i for fold in folds for i in fold]
    # exact partition: every index once, no gaps or overlaps
    assert sorted(flat) == list(range(n))
    # near-equal sizes
    sizes = [len(f) for f in folds]
    assert max(sizes) - min(sizes) <= 1
