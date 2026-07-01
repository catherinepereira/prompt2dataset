from __future__ import annotations

from prompt2dataset.download import extension_for, host_is_public


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
