"""
Image source adapters

To add a new source:
1. Write an async fetch function: (subject, limit) -> list[dict]
2. Add a SourceAdapter entry to REGISTRY at the bottom of this file
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Coroutine

import httpx

from prompt2dataset.models import MediaType

log = logging.getLogger(__name__)

HTTP_TIMEOUT = 30

# Each source caps how many results it returns per request. These are the
# per-page maximums the respective APIs accept. We never ask for more.
INATURALIST_MAX_PER_PAGE = 200
OPENVERSE_MAX_PAGE_SIZE = 20
WIKIMEDIA_MAX_RESULTS = 50

# Bing has no JSON API. Its async endpoint returns a fixed-size page of results.
BING_PAGE_SIZE = 35


def _user_agent() -> str:
    contact = os.environ.get("P2D_CONTACT", "unknown")
    return f"prompt2dataset/0.1 ({contact}) httpx/0.27"

FetchFn = Callable[[str, int], Coroutine[Any, Any, list[dict[str, Any]]]]


@dataclass(frozen=True)
class SourceAdapter:
    name: str
    description: str
    fetch: FetchFn
    media_types: tuple[MediaType, ...] = (MediaType.image,)


async def _fetch_inaturalist(subject: str, limit: int = 20) -> list[dict[str, Any]]:
    params: dict[str, Any] = {
        "q": subject,
        "quality_grade": "research",
        "photos": "true",
        "per_page": min(limit, INATURALIST_MAX_PER_PAGE),
        "order": "votes",
        "order_by": "votes",
    }
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        try:
            resp = await client.get(
                "https://api.inaturalist.org/v1/observations",
                params=params,
                headers={"Accept": "application/json"},
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            log.warning("iNaturalist failed for %r: %s", subject, exc)
            return []

    results = []
    for obs in resp.json().get("results", []):
        photos = obs.get("photos") or []
        if not photos:
            continue
        url = photos[0].get("url", "").replace("/square.", "/medium.")
        if not url:
            continue
        results.append({
            "source": "inaturalist",
            "url": url,
            "taxon": obs.get("taxon", {}).get("name", subject),
            "common_name": obs.get("taxon", {}).get("preferred_common_name", ""),
            "place": obs.get("place_guess", ""),
            "observed_on": obs.get("observed_on", ""),
            "license": photos[0].get("license_code", "unknown"),
        })
        if len(results) >= limit:
            break

    return results


async def _fetch_openverse(subject: str, limit: int = 20) -> list[dict[str, Any]]:
    params = {
        "q": subject,
        "license_type": "commercial,modification",
        "page_size": min(limit, OPENVERSE_MAX_PAGE_SIZE),
    }
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        try:
            resp = await client.get("https://api.openverse.org/v1/images/", params=params)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            log.warning("Openverse failed for %r: %s", subject, exc)
            return []

    results = []
    for img in resp.json().get("results", [])[:limit]:
        url = img.get("url", "")
        if not url:
            continue
        url = _wikimedia_full_to_thumb(url)
        results.append({
            "source": "openverse",
            "url": url,
            "title": img.get("title", ""),
            "creator": img.get("creator", ""),
            "license": img.get("license", "unknown"),
            "source_site": img.get("source", ""),
        })

    return results


def _wikimedia_full_to_thumb(url: str, width: int = 960) -> str:
    """Convert a full-res upload.wikimedia.org URL to a scaled thumbnail URL."""
    m = re.match(
        r"(https://upload\.wikimedia\.org/wikipedia/commons/)([0-9a-f]/[0-9a-f]{2})/(.+)$",
        url,
    )
    if not m:
        return url
    base, hash_path, filename = m.groups()
    return f"{base}thumb/{hash_path}/{filename}/{width}px-{filename}"


async def _query_wikimedia_commons(subject: str, limit: int) -> list[dict[str, Any]]:
    params: dict[str, Any] = {
        "action": "query",
        "generator": "search",
        "gsrsearch": subject,
        "gsrnamespace": 6,
        "gsrlimit": min(limit, WIKIMEDIA_MAX_RESULTS),
        "prop": "imageinfo",
        # iiurlwidth asks Wikimedia for a scaled thumbnail rather than the full file,
        # per their guidance for API consumers to avoid 429s.
        "iiprop": "url|mime",
        "iiurlwidth": 800,
        "format": "json",
    }
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        try:
            resp = await client.get(
                "https://commons.wikimedia.org/w/api.php",
                params=params,
                headers={"User-Agent": _user_agent()},
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            log.warning("Wikimedia Commons failed for %r: %s", subject, exc)
            return []

    return list(resp.json().get("query", {}).get("pages", {}).values())


async def _fetch_wikimedia_commons(subject: str, limit: int = 20) -> list[dict[str, Any]]:
    results = []
    for page in await _query_wikimedia_commons(subject, limit):
        info = (page.get("imageinfo") or [{}])[0]
        url = info.get("thumburl") or info.get("url", "")
        if not url or info.get("mime") == "image/svg+xml":
            continue
        # Strip the UTM params appended by Wikimedia to thumburls that cause CDN 403s
        url = url.split("?")[0]
        results.append({
            "source": "wikimedia_commons",
            "url": url,
            "title": page.get("title", "").removeprefix("File:"),
            "description_url": info.get("descriptionurl", ""),
        })
        if len(results) >= limit:
            break

    return results


async def _fetch_wikimedia_commons_video(subject: str, limit: int = 20) -> list[dict[str, Any]]:
    results = []
    for page in await _query_wikimedia_commons(subject, limit):
        info = (page.get("imageinfo") or [{}])[0]
        if not (info.get("mime") or "").startswith("video/"):
            continue
        # A video thumburl is a still frame, so use the full file URL instead
        url = info.get("url", "")
        if not url:
            continue
        url = url.split("?")[0]
        results.append({
            "source": "wikimedia_commons_video",
            "url": url,
            "title": page.get("title", "").removeprefix("File:"),
            "description_url": info.get("descriptionurl", ""),
            "mime": info.get("mime", ""),
        })
        if len(results) >= limit:
            break

    return results


async def _fetch_duckduckgo(subject: str, limit: int = 20) -> list[dict[str, Any]]:
    # DDG requires a POST first to get a vqd token, then a GET for results
    headers = {
        "User-Agent": _user_agent(),
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Referer": "https://duckduckgo.com/",
    }
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, follow_redirects=True) as client:
        try:
            init = await client.post(
                "https://duckduckgo.com/",
                data={"q": subject},
                headers={**headers, "Content-Type": "application/x-www-form-urlencoded"},
            )
            init.raise_for_status()
            vqd_match = re.search(r'vqd=["\']([\d-]+)["\']', init.text)
            if not vqd_match:
                log.warning("DuckDuckGo: could not find vqd token for %r", subject)
                return []
            vqd = vqd_match.group(1)

            resp = await client.get(
                "https://duckduckgo.com/i.js",
                params={"l": "us-en", "o": "json", "q": subject, "vqd": vqd, "f": ",,,,,", "p": "1"},
                headers=headers,
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            log.warning("DuckDuckGo failed for %r: %s", subject, exc)
            return []

    results = []
    for img in resp.json().get("results", [])[:limit]:
        url = img.get("image", "")
        if not url:
            continue
        results.append({
            "source": "duckduckgo",
            "url": url,
            "title": img.get("title", ""),
            "width": img.get("width"),
            "height": img.get("height"),
            "source_site": img.get("source", ""),
            "thumbnail": img.get("thumbnail", ""),
        })

    return results


async def _fetch_bing(subject: str, limit: int = 20) -> list[dict[str, Any]]:
    # Bing's image results page embeds each result as JSON in an m="..." attribute
    # on the result anchors. murl is the full image URL. Paginate with first=.
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }
    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, follow_redirects=True) as client:
        for first in range(0, limit * 2, BING_PAGE_SIZE):
            if len(results) >= limit:
                break
            try:
                resp = await client.get(
                    "https://www.bing.com/images/async",
                    params={"q": subject, "first": first, "count": BING_PAGE_SIZE, "mmasync": 1},
                    headers=headers,
                )
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                log.warning("Bing failed for %r: %s", subject, exc)
                break

            urls = re.findall(r'murl&quot;:&quot;(.*?)&quot;', resp.text)
            if not urls:
                break
            for raw in urls:
                url = raw.replace("\\u0026", "&").replace("\\/", "/")
                if url in seen or not url.startswith("http"):
                    continue
                seen.add(url)
                results.append({"source": "bing", "url": url})
                if len(results) >= limit:
                    break

    return results


REGISTRY: dict[str, SourceAdapter] = {
    "duckduckgo": SourceAdapter(
        name="duckduckgo",
        description=(
            "General web image search via DuckDuckGo."
        ),
        fetch=_fetch_duckduckgo,
    ),
    "bing": SourceAdapter(
        name="bing",
        description="General web image search via Bing.",
        fetch=_fetch_bing,
    ),
    "inaturalist": SourceAdapter(
        name="inaturalist",
        description=(
            "Research-grade nature observation photos from iNaturalist."
        ),
        fetch=_fetch_inaturalist,
    ),
    "openverse": SourceAdapter(
        name="openverse",
        description=(
            "Openly-licensed images from Openverse (Wikipedia, Flickr, museums, and more)."
        ),
        fetch=_fetch_openverse,
    ),
    "wikimedia_commons": SourceAdapter(
        name="wikimedia_commons",
        description=(
            "Freely licensed photos and diagrams from Wikimedia Commons."
        ),
        fetch=_fetch_wikimedia_commons,
    ),
    "wikimedia_commons_video": SourceAdapter(
        name="wikimedia_commons_video",
        description=(
            "Freely licensed video clips from Wikimedia Commons."
        ),
        fetch=_fetch_wikimedia_commons_video,
        media_types=(MediaType.video,),
    ),
}


async def fetch_all(
    subjects: list[str],
    source_names: list[str],
    limit_per_subject: int = 20,
    media_type: MediaType = MediaType.image,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Fetch from each source for every subject concurrently."""
    tasks: dict[tuple[str, str], asyncio.Task] = {}
    async with asyncio.TaskGroup() as tg:
        for subject in subjects:
            for source_name in source_names:
                adapter = REGISTRY.get(source_name)
                if adapter is None:
                    log.warning("Unknown source %r - skipping", source_name)
                    continue
                if media_type not in adapter.media_types:
                    log.warning(
                        "Source %r does not support %s - skipping",
                        source_name, media_type.value,
                    )
                    continue
                tasks[(subject, source_name)] = tg.create_task(
                    adapter.fetch(subject, limit_per_subject)
                )

    out: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for (subject, source_name), task in tasks.items():
        out.setdefault(subject, {})[source_name] = task.result()

    return out
