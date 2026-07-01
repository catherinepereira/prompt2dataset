from __future__ import annotations

import hashlib
import time
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

# Length of the hex item id derived from a source URL. 12 hex chars = 48 bits,
# enough to make collisions across a single dataset's URLs vanishingly unlikely.
ID_LENGTH = 12


class ReviewStatus(str, Enum):
    pending = "pending"
    valid = "valid"
    invalid = "invalid"


class DatasetItem(BaseModel):
    item_id: str
    label: str
    source_url: str
    local_path: str
    # the original human-readable subject the label was slugified from, for display
    # defaults empty so manifests written before this field loads still validate
    subject: str = ""
    review_status: ReviewStatus = ReviewStatus.pending
    meta: dict[str, Any] = Field(default_factory=dict)
    fetched_at: float = Field(default_factory=time.time)

    @classmethod
    def make_id(cls, source_url: str) -> str:
        return hashlib.sha1(source_url.encode()).hexdigest()[:ID_LENGTH]


class Dataset(BaseModel):
    dataset_id: str
    prompt: str
    subjects: list[str]
    sources: list[str]
    items: list[DatasetItem] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)

    def touch(self) -> None:
        self.updated_at = time.time()

    def add_items(self, new_items: list[DatasetItem]) -> list[str]:
        """Append items not already present by id, returning the ids actually added."""
        existing = {item.item_id for item in self.items}
        added: list[str] = []
        for item in new_items:
            if item.item_id not in existing:
                self.items.append(item)
                existing.add(item.item_id)
                added.append(item.item_id)
        self.touch()
        return added

    def pending_review(self) -> list[DatasetItem]:
        return [i for i in self.items if i.review_status == ReviewStatus.pending]

    def stats(self) -> dict[str, int]:
        counts: dict[str, int] = {"total": len(self.items), "pending": 0, "valid": 0, "invalid": 0}
        for item in self.items:
            counts[item.review_status.value] += 1
        return counts
