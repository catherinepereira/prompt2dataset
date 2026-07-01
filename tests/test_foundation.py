from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from prompt2dataset.ids import FALLBACK_SLUG, slugify
from prompt2dataset.models import Dataset, DatasetItem
from prompt2dataset.paths import MANIFEST_DIR, meta_dir
from prompt2dataset.progress import Reporter
from prompt2dataset.store import load_dataset, prune_missing, save_dataset


def test_slugify():
    assert slugify("American Robin") == "american-robin"
    assert slugify("../etc/passwd") == "etcpasswd"
    assert slugify("A/B:C") == "abc"
    assert slugify("  Blue  Jay  ") == "blue-jay"


def test_slugify_never_empty_or_traversal():
    # inputs that strip to nothing fall back rather than collapsing into the root
    for hostile in ["", "...", "..", ".", "/", "///", "@#$%"]:
        assert slugify(hostile) == FALLBACK_SLUG
    # a slug can't contain a path separator, so it can't escape its parent dir
    for hostile in ["../../etc", "..\\..\\x", "foo/../bar", "C:\\Windows"]:
        assert "/" not in slugify(hostile) and "\\" not in slugify(hostile)


def test_slugify_avoids_windows_reserved_names():
    # con/nul/etc. can't be folder names on Windows, so they fall back
    for reserved in ["con", "NUL", "aux", "COM1", "lpt9"]:
        assert slugify(reserved) == FALLBACK_SLUG


def test_slugify_folds_unicode_to_ascii():
    assert slugify("Café Crème") == "cafe-creme"
    # a slug of only non-ascii strips empty and falls back
    assert slugify("日本語") == FALLBACK_SLUG


def test_meta_dir_creates_with_parents(tmp_path: Path):
    root = tmp_path / "new" / "deep" / "ds"
    md = meta_dir(root)
    assert md.exists() and md.name == MANIFEST_DIR


def test_save_and_load_roundtrip(tmp_path: Path):
    root = tmp_path / "ds"
    ds = Dataset(dataset_id="ds", prompt="birds", subjects=["robin"], sources=["web"])
    ds.items.append(
        DatasetItem(item_id="a", label="robin", subject="American Robin",
                    source_url="u", local_path="robin/robin_a.png",
                    meta={"source": "web"})
    )
    save_dataset(ds, root)
    reloaded = load_dataset(root)
    assert reloaded.items[0].subject == "American Robin"
    # labels.csv carries label and subject columns
    csv = (root / MANIFEST_DIR / "labels.csv").read_text()
    assert csv.splitlines()[0] == "filename,label,subject,source"
    assert "robin,American Robin,web" in csv


def test_load_missing_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "nope")


def test_labels_csv_escapes_hostile_fields(tmp_path: Path):
    root = tmp_path / "ds"
    ds = Dataset(dataset_id="ds", prompt="", subjects=["x"], sources=[])
    # a subject with a comma, and one starting with a formula char
    ds.items.append(DatasetItem(item_id="a", label="birds", subject="robin, jay",
                                source_url="u", local_path="birds/birds_a.png",
                                meta={"source": "web"}))
    ds.items.append(DatasetItem(item_id="b", label="=cmd", subject="ok",
                                source_url="u", local_path="birds/birds_b.png",
                                meta={"source": "web"}))
    save_dataset(ds, root)
    rows = (root / MANIFEST_DIR / "labels.csv").read_text().splitlines()
    # the embedded comma is quoted, not spilled into a new column
    assert '"robin, jay"' in rows[1]
    # the formula-looking label is neutralized with a leading quote
    assert "'=cmd" in rows[2]


def test_prune_missing_drops_and_keeps(tmp_path: Path):
    root = tmp_path / "ds"
    present = DatasetItem(item_id="p", label="robin", source_url="u",
                          local_path="robin/robin_p.png")
    (root / present.local_path).parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), (0, 0, 0)).save(root / present.local_path)
    absent = DatasetItem(item_id="a", label="robin", source_url="u",
                         local_path="robin/robin_a.png")
    ds = Dataset(dataset_id="ds", prompt="", subjects=["robin"], sources=[],
                 items=[present, absent])

    assert prune_missing(ds, root) == 1
    assert [i.item_id for i in ds.items] == ["p"]


def test_reporter_forwards_updates():
    seen = []
    r = Reporter(lambda p: seen.append((p.done, p.total, p.message)))
    r.start(3, "go")
    r.advance("one")
    assert seen[0] == (0, 3, "go")
    assert seen[1] == (1, 3, "one")
