from __future__ import annotations

from pathlib import Path

from PIL import Image

from prompt2dataset.clean import apply_flags, find_exact_duplicates
from prompt2dataset.models import Dataset, DatasetItem, ReviewStatus


def _item(label: str, name: str) -> DatasetItem:
    return DatasetItem(
        item_id=name, label=label, source_url=f"https://x/{name}",
        local_path=f"{label}/{name}.png",
    )


def _write(root: Path, item: DatasetItem, color) -> None:
    p = root / item.local_path
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color).save(p)


def test_find_exact_duplicates(tmp_path: Path):
    root = tmp_path / "ds"
    a = _item("robin", "a")
    b = _item("robin", "b")  # identical pixels to a
    c = _item("robin", "c")  # different
    _write(root, a, (10, 20, 30))
    _write(root, b, (10, 20, 30))
    _write(root, c, (200, 0, 0))

    dupes = find_exact_duplicates([a, b, c], root)
    assert [d.item_id for d in dupes] == ["b"]


def test_oversized_image_is_skipped_not_crashed(tmp_path: Path, monkeypatch):
    # a decode past the pixel cap raises DecompressionBombError, which the clean pass
    # must treat as an unreadable file, not let it propagate and kill the run
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 16)  # 8x8 = 64 px exceeds this
    root = tmp_path / "ds"
    big = _item("robin", "big")
    ok = _item("robin", "ok")
    _write(root, big, (10, 20, 30))
    _write(root, ok, (200, 0, 0))

    # does not raise, and the oversized image is simply not hashed/flagged
    dupes = find_exact_duplicates([big, ok], root)
    assert dupes == []


def test_apply_flags_marks_invalid(tmp_path: Path):
    root = tmp_path / "ds"
    a = _item("robin", "a")
    _write(root, a, (1, 1, 1))
    ds = Dataset(dataset_id="ds", prompt="", subjects=["robin"], sources=[], items=[a])

    apply_flags([a], ds, root, delete=False)
    assert ds.items[0].review_status == ReviewStatus.invalid
    assert (root / a.local_path).exists()


def test_apply_flags_delete_removes(tmp_path: Path):
    root = tmp_path / "ds"
    a = _item("robin", "a")
    _write(root, a, (1, 1, 1))
    ds = Dataset(dataset_id="ds", prompt="", subjects=["robin"], sources=[], items=[a])

    apply_flags([a], ds, root, delete=True)
    assert ds.items == []
    assert not (root / a.local_path).exists()
