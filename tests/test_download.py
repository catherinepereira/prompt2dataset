from __future__ import annotations

from pathlib import Path

import httpx

from prompt2dataset import download as dl
from prompt2dataset.download import (
    MAX_DOWNLOAD_BYTES,
    download_file,
    extension_for,
    host_is_public,
)


def test_extension_for():
    assert extension_for("https://x/a.png") == ".png"
    assert extension_for("https://x/a.JPG?w=1") == ".jpg"
    assert extension_for("https://x/a.webp") == ".webp"
    # unknown/extensionless defaults to jpg
    assert extension_for("https://x/photo") == ".jpg"
    assert extension_for("https://x/a.svg") == ".jpg"


def test_host_is_public_rejects_internal():
    assert host_is_public("localhost") is False
    assert host_is_public("127.0.0.1") is False
    assert host_is_public("10.0.0.1") is False
    assert host_is_public("192.168.1.1") is False
    # link-local metadata endpoint
    assert host_is_public("169.254.169.254") is False


def test_host_is_public_rejects_unresolvable():
    assert host_is_public("nonexistent.invalid.") is False


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), timeout=5)


def test_download_writes_body(tmp_path: Path, monkeypatch):
    # host_is_public does a real DNS lookup, stub it so the mock transport is reached
    monkeypatch.setattr(dl, "host_is_public", lambda host: True)

    def handler(request):
        return httpx.Response(200, content=b"imagebytes")

    dest = tmp_path / "img.jpg"
    assert download_file("https://cdn.example/a.jpg", dest, client=_client(handler)) is True
    assert dest.read_bytes() == b"imagebytes"
    # the temp .part file is cleaned up
    assert not dest.with_name(dest.name + ".part").exists()


def test_download_rejects_redirect_to_private_host(tmp_path: Path, monkeypatch):
    # a public URL that 302s to the cloud metadata endpoint must be refused on the
    # second hop, not followed
    reached = []

    def fake_public(host):
        reached.append(host)
        return host == "cdn.example"

    monkeypatch.setattr(dl, "host_is_public", fake_public)

    def handler(request):
        return httpx.Response(302, headers={"Location": "http://169.254.169.254/latest/"})

    dest = tmp_path / "img.jpg"
    assert download_file("https://cdn.example/a.jpg", dest, client=_client(handler)) is False
    assert not dest.exists()
    # the guard ran on both the original and the redirect target
    assert reached == ["cdn.example", "169.254.169.254"]


def test_download_aborts_oversized_stream(tmp_path: Path, monkeypatch):
    # a body larger than the cap with no Content-Length must be aborted mid-stream
    monkeypatch.setattr(dl, "host_is_public", lambda host: True)

    def oversized():
        chunk = b"\0" * (1024 * 1024)
        sent = 0
        while sent <= MAX_DOWNLOAD_BYTES:
            sent += len(chunk)
            yield chunk

    def handler(request):
        return httpx.Response(200, content=oversized())

    dest = tmp_path / "big.jpg"
    assert download_file("https://cdn.example/big.jpg", dest, client=_client(handler)) is False
    assert not dest.exists()
    assert not dest.with_name(dest.name + ".part").exists()


def test_download_rejects_oversized_content_length(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(dl, "host_is_public", lambda host: True)

    def handler(request):
        return httpx.Response(
            200, headers={"Content-Length": str(MAX_DOWNLOAD_BYTES + 1)}, content=b"x"
        )

    dest = tmp_path / "big.jpg"
    assert download_file("https://cdn.example/big.jpg", dest, client=_client(handler)) is False
    assert not dest.exists()
