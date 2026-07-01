from __future__ import annotations

from pathlib import Path

from PIL import Image

from prompt2dataset.ingest import prune_missing
from prompt2dataset.models import Dataset, DatasetItem


def _item(label: str, n: int) -> DatasetItem:
    url = f"https://example.test/{label}/{n}.png"
    item_id = DatasetItem.make_id(url)
    return DatasetItem(
        item_id=item_id,
        label=label,
        source_url=url,
        local_path=str(Path(label) / f"{label}_{item_id}.png"),
    )


def _build(tmp_path: Path) -> tuple[Dataset, Path, DatasetItem, DatasetItem]:
    root = tmp_path / "ds"
    on_disk = _item("robin", 0)
    p = root / on_disk.local_path
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), (0, 0, 0)).save(p)
    missing = _item("robin", 1)  # no file written
    ds = Dataset(dataset_id="ds", prompt="", subjects=["robin"], sources=[], items=[on_disk, missing])
    return ds, root, on_disk, missing


def test_prune_drops_items_with_no_file(tmp_path: Path):
    ds, root, on_disk, missing = _build(tmp_path)
    removed = prune_missing(ds, root)
    assert removed == 1
    assert [i.item_id for i in ds.items] == [on_disk.item_id]


def test_prune_keeps_predicate_matches(tmp_path: Path):
    ds, root, on_disk, missing = _build(tmp_path)
    # keep the missing item anyway (as a recycle-bin item would be kept)
    removed = prune_missing(ds, root, keep=lambda i: i.item_id == missing.item_id)
    assert removed == 0
    assert len(ds.items) == 2


def test_prune_noop_when_all_present(tmp_path: Path):
    ds, root, on_disk, _ = _build(tmp_path)
    ds.items = [on_disk]
    assert prune_missing(ds, root) == 0
