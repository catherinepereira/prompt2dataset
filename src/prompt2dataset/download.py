"""Fetch an image URL to disk, refusing internal hosts and oversized bodies.

Sources return URLs we then fetch. Redirects are followed by hand so each hop's host is
checked before we connect, so a URL (or a redirect from one) can't reach localhost, a
cloud metadata endpoint, or an internal address. The body is streamed to a temp file and
capped, so a hostile or oversized response can't exhaust memory or leave a truncated
file the next run treats as done.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import socket
from pathlib import Path

import httpx

log = logging.getLogger(__name__)

MAX_DOWNLOAD_BYTES = 512 * 1024 * 1024
_DOWNLOAD_CHUNK = 1024 * 1024
_MAX_REDIRECTS = 10
_TIMEOUT = 20

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}


def extension_for(url: str) -> str:
    """The image extension implied by a URL, defaulting to .jpg."""
    ext = Path(url.split("?")[0].rstrip("/")).suffix.lower()
    return ext if ext in IMAGE_EXTS else ".jpg"


def _user_agent() -> str:
    contact = os.environ.get("P2D_CONTACT", "unknown")
    return f"prompt2dataset/2.0 ({contact}) httpx"


def host_is_public(host: str) -> bool:
    """True if every address the host resolves to is a routable public IP.

    Resolving the name here catches a public DNS name pointed at a private IP, which a
    literal-only check would miss. A rebinding attacker can still return a different IP
    at connect time, which is out of scope for this tool.
    """
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            return False
    return True


def download_file(url: str, dest: Path) -> bool:
    """Fetch url to dest. Returns True on success, False on any refusal or error."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    ua = _user_agent()
    part = dest.with_name(dest.name + ".part")
    try:
        # httpx speaks only http(s), so file:// and ftp:// are already refused
        with httpx.Client(timeout=_TIMEOUT) as client:
            for _ in range(_MAX_REDIRECTS + 1):
                if not host_is_public(httpx.URL(url).host):
                    log.warning("Refusing non-public host for %s", url)
                    return False
                with client.stream("GET", url, headers={"User-Agent": ua}) as resp:
                    if resp.is_redirect:
                        url = str(resp.next_request.url)
                        continue
                    resp.raise_for_status()
                    length = resp.headers.get("Content-Length")
                    if length and int(length) > MAX_DOWNLOAD_BYTES:
                        log.warning("Skipping %s: %s bytes exceeds cap", url, length)
                        return False
                    written = 0
                    with open(part, "wb") as f:
                        for chunk in resp.iter_bytes(_DOWNLOAD_CHUNK):
                            written += len(chunk)
                            if written > MAX_DOWNLOAD_BYTES:
                                log.warning("Aborting %s: exceeded %d byte cap", url, MAX_DOWNLOAD_BYTES)
                                part.unlink(missing_ok=True)
                                return False
                            f.write(chunk)
                    os.replace(part, dest)
                    return True
            log.warning("Too many redirects for %s", url)
            return False
    except Exception as exc:
        log.warning("Download failed %s: %s", url, exc)
        part.unlink(missing_ok=True)
        return False
