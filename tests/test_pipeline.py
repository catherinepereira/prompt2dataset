from __future__ import annotations

from pathlib import Path
from unittest import mock

from prompt2dataset import pipeline
from prompt2dataset.models import Dataset
from prompt2dataset.store import load_dataset


def _fresh(tmp_path: Path) -> tuple[Dataset, Path]:
    root = tmp_path / "ds"
    return Dataset(dataset_id="ds", prompt="", subjects=[], sources=[]), root


async def _fake_fetch(subjects, sources, limit):
    return {
        s: {sources[0]: [{"source": sources[0], "url": f"https://x/{s}.jpg"}]}
        for s in subjects
    }


def _ok_download(url, dest, client=None):
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b"x")
    return True


def test_records_to_items_stores_subject_and_slug():
    raw = {"American Robin": {"web": [{"source": "web", "url": "https://x/1.jpg"}]}}
    items = pipeline.records_to_items(raw)
    assert items[0].label == "american-robin"
    assert items[0].subject == "American Robin"


def test_generate_downloads_and_saves(tmp_path: Path):
    ds, root = _fresh(tmp_path)
    with mock.patch.object(pipeline, "fetch_all", _fake_fetch), mock.patch.object(
        pipeline, "download_file", _ok_download
    ):
        result = pipeline.generate(ds, root, ["Otter", "Seal"], ["duckduckgo"], 3)
    assert result.saved == 2 and result.dropped == 0
    assert load_dataset(root).subjects == ["Otter", "Seal"]


def test_generate_prunes_failed_downloads(tmp_path: Path):
    ds, root = _fresh(tmp_path)
    with mock.patch.object(pipeline, "fetch_all", _fake_fetch), mock.patch.object(
        pipeline, "download_file", lambda u, d, client=None: False
    ):
        result = pipeline.generate(ds, root, ["Otter"], ["duckduckgo"], 3)
    assert result.failed == 1 and result.dropped == 1
    assert load_dataset(root).items == []


def test_generate_skips_known_subjects(tmp_path: Path):
    ds, root = _fresh(tmp_path)
    with mock.patch.object(pipeline, "fetch_all", _fake_fetch), mock.patch.object(
        pipeline, "download_file", _ok_download
    ):
        pipeline.generate(ds, root, ["Otter"], ["duckduckgo"], 3)
        result = pipeline.generate(ds, root, ["Otter"], ["duckduckgo"], 3)
    assert result == pipeline.GenerateResult(0, 0, 0, 0, 0)


def test_generate_reports_progress(tmp_path: Path):
    ds, root = _fresh(tmp_path)
    seen = []
    with mock.patch.object(pipeline, "fetch_all", _fake_fetch), mock.patch.object(
        pipeline, "download_file", _ok_download
    ):
        pipeline.generate(
            ds, root, ["Otter"], ["duckduckgo"], 2, on_progress=lambda p: seen.append(p.message)
        )
    assert any("Searching" in m for m in seen)
    assert any("Saved" in m for m in seen)
