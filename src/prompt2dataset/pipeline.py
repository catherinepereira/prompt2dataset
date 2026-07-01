"""The headless generate pipeline: subjects to downloaded, manifested images.

generate() runs fetch, records-to-items, download, and prune with no terminal I/O,
reporting through an optional progress callback. The CLI and any other consumer call it.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from prompt2dataset.download import DOWNLOAD_TIMEOUT, download_file, extension_for
from prompt2dataset.ids import slugify
from prompt2dataset.models import Dataset, DatasetItem
from prompt2dataset.progress import OnProgress, Reporter
from prompt2dataset.sources import fetch_all
from prompt2dataset.store import prune_missing, save_dataset

# Seconds between downloads, a courtesy to the hosts we fetch from
DOWNLOAD_RATE_LIMIT = 0.1


@dataclass
class GenerateResult:
    records: int
    added: int
    saved: int
    failed: int
    dropped: int


def records_to_items(raw_results: dict) -> list[DatasetItem]:
    """Flatten fetch_all's nested results into dataset items."""
    items: list[DatasetItem] = []
    for subject, source_map in raw_results.items():
        label = slugify(subject)
        for records in source_map.values():
            for rec in records:
                url = rec.get("url", "")
                if not url:
                    continue
                item_id = DatasetItem.make_id(url)
                ext = extension_for(url)
                items.append(DatasetItem(
                    item_id=item_id,
                    label=label,
                    subject=subject,
                    source_url=url,
                    local_path=str(Path(label) / f"{label}_{item_id}{ext}"),
                    meta={k: v for k, v in rec.items() if k != "url"},
                ))
    return items


def generate(
    ds: Dataset,
    dataset_root: Path,
    subjects: list[str],
    sources: list[str],
    limit: int,
    *,
    on_progress: OnProgress = None,
    keep_on_prune=None,
) -> GenerateResult:
    """Fetch and download images for subjects into ds, merging into its manifest.

    Only subjects not already in ds are fetched. Items whose download fails are pruned
    so the manifest lists only images on disk. keep_on_prune retains items whose file is
    intentionally elsewhere (a recycle bin). Saves the dataset before returning.
    """
    reporter = Reporter(on_progress)

    new_subjects = [s for s in subjects if s not in ds.subjects]
    ds.subjects += new_subjects
    for s in sources:
        if s not in ds.sources:
            ds.sources.append(s)
    for subject in new_subjects:
        (dataset_root / slugify(subject)).mkdir(parents=True, exist_ok=True)

    if not new_subjects:
        return GenerateResult(0, 0, 0, 0, 0)

    # fetch every subject in one event loop, so fetch_all fans them out concurrently
    reporter.start(1, "Searching sources")
    raw_results = asyncio.run(fetch_all(new_subjects, sources, limit))
    reporter.advance(f"Searched {len(new_subjects)} subjects")

    records = sum(
        len(recs) for src_map in raw_results.values() for recs in src_map.values()
    )
    new_items = records_to_items(raw_results)
    added_ids = set(ds.add_items(new_items))
    added = len(added_ids)

    # only the newly added items need downloading, existing ones are already on disk
    pending = [i for i in ds.items if i.item_id in added_ids]
    reporter.start(max(len(pending), 1), "Downloading images")
    saved = failed = 0
    with httpx.Client(timeout=DOWNLOAD_TIMEOUT) as client:
        for item in pending:
            if download_file(item.source_url, dataset_root / item.local_path, client=client):
                saved += 1
            else:
                failed += 1
            reporter.advance(f"Saved {saved}/{len(pending)}")
            time.sleep(DOWNLOAD_RATE_LIMIT)

    dropped = prune_missing(ds, dataset_root, keep=keep_on_prune)
    save_dataset(ds, dataset_root)
    return GenerateResult(records, added, saved, failed, dropped)
