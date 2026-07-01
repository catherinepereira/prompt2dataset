from __future__ import annotations

import asyncio

from prompt2dataset.sources import (
    REGISTRY,
    SourceAdapter,
    fetch_all,
    register_source,
    source_names,
)


def test_builtin_sources_present():
    names = source_names()
    assert {"duckduckgo", "bing", "wikimedia_commons", "openverse", "inaturalist"} <= set(names)


def test_register_source_adds_and_replaces():
    async def fake_fetch(subject, limit):
        return [{"source": "fake", "url": f"https://x/{subject}.png"}]

    register_source(SourceAdapter(name="fake", description="test", fetch=fake_fetch))
    assert "fake" in REGISTRY

    result = asyncio.run(fetch_all(["robin"], ["fake"], 5))
    assert result["robin"]["fake"][0]["url"] == "https://x/robin.png"

    del REGISTRY["fake"]


def test_fetch_all_skips_unknown_source():
    # unknown source name is skipped, not an error
    result = asyncio.run(fetch_all(["robin"], ["does_not_exist"], 5))
    assert result == {}
