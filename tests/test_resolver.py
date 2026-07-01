from __future__ import annotations

from unittest import mock

import httpx

from prompt2dataset import resolver


def _fake_ollama(content: str):
    def post(url, json, timeout):
        return httpx.Response(
            200, json={"message": {"content": content}}, request=httpx.Request("POST", url)
        )

    return post


def test_resolve_parses_and_strips():
    with mock.patch.object(resolver.httpx, "post", _fake_ollama('["American Robin", "Blue Jay"]')):
        assert resolver.resolve_subjects("birds") == ["American Robin", "Blue Jay"]


def test_resolve_excludes_already_chosen():
    with mock.patch.object(
        resolver.httpx, "post", _fake_ollama('["Blue Jay", "Robin", "Crow"]')
    ):
        out = resolver.resolve_subjects("birds", exclude=["blue jay"])
    assert "Blue Jay" not in out and "Robin" in out


def test_count_hint_appended_to_prompt():
    captured = {}

    def post(url, json, timeout):
        captured["user"] = json["messages"][-1]["content"]
        return httpx.Response(200, json={"message": {"content": "[]"}}, request=httpx.Request("POST", url))

    with mock.patch.object(resolver.httpx, "post", post):
        resolver.resolve_subjects("birds", count=5)
    assert "about 5" in captured["user"]
