"""Phase 6.3: verify_citation, with the network stubbed.

What these pin down: id normalisation, and that each way a lookup can end maps to
the right status - especially that a *failed* lookup is ERROR, never NOT
FOUND. Whether arXiv's real API answers a nonexistent id the way the stub
does here is a real-network question (see the 6.3 report).
"""

import urllib.error

import pytest

from research_copilot.tools import citations
from research_copilot.tools.citations import check_arxiv_id, normalize_arxiv_id, parse_check

FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">{entries}</feed>"""

ENTRY = """<entry><id>http://arxiv.org/abs/{id}v1</id><title>{title}</title>
<summary>abstract</summary><published>2023-09-26T00:00:00Z</published>
<author><name>A. Author</name></author></entry>"""

ERROR_ENTRY = """<entry><id>http://arxiv.org/api/errors#incorrect_id_format_for_{id}</id>
<title>Error</title><summary>incorrect id format for {id}</summary></entry>"""


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("2309.15217", "2309.15217"),
        ("2309.15217v2", "2309.15217"),
        ("http://arxiv.org/abs/2309.15217v3", "2309.15217"),
        ("https://arxiv.org/pdf/2309.15217.pdf", "2309.15217"),
        ("arXiv:1706.03762", "1706.03762"),
        ("cs/0112017", "cs/0112017"),
        ("hep-th/9901001v2", "hep-th/9901001"),
        ("attention is all you need", None),
        ("12345", None),
    ],
)
def test_normalize_arxiv_id(raw, expected):
    assert normalize_arxiv_id(raw) == expected


def stub_fetch(monkeypatch, *, body=None, exc=None):
    calls = []

    def fake(arxiv_id):
        calls.append(arxiv_id)
        if exc is not None:
            raise exc
        return body

    monkeypatch.setattr(citations, "_fetch_by_id", fake)
    return calls


def test_found(monkeypatch):
    calls = stub_fetch(monkeypatch, body=FEED.format(entries=ENTRY.format(id="2309.15217", title="RAGAS")))
    result = check_arxiv_id("http://arxiv.org/abs/2309.15217v2")
    assert result == "FOUND: 2309.15217 - RAGAS"
    assert calls == ["2309.15217"]
    assert parse_check(result) == "found"


def test_not_found_is_an_empty_feed(monkeypatch):
    stub_fetch(monkeypatch, body=FEED.format(entries=""))
    result = check_arxiv_id("2401.99999")
    assert result.startswith("NOT FOUND:")
    assert parse_check(result) == "not_found"


def test_invalid_without_a_network_call(monkeypatch):
    calls = stub_fetch(monkeypatch, body="unused")
    result = check_arxiv_id("not an id")
    assert result.startswith("INVALID:")
    assert calls == []
    assert parse_check(result) == "invalid"


def test_arxiv_error_entry_is_invalid(monkeypatch):
    stub_fetch(monkeypatch, body=FEED.format(entries=ERROR_ENTRY.format(id="9999.9999")))
    assert parse_check(check_arxiv_id("9999.9999")) == "invalid"


@pytest.mark.parametrize(
    "exc",
    [
        urllib.error.URLError("connection refused"),
        TimeoutError("timed out"),
        ValueError("Rate exceeded."),
    ],
    ids=["network", "timeout", "rate-limit"],
)
def test_a_failed_lookup_is_error_not_not_found(monkeypatch, exc):
    """The distinction the Critic's prompt depends on. A timeout must never read
    as "this citation is fake"."""
    stub_fetch(monkeypatch, exc=exc)
    result = check_arxiv_id("2309.15217")
    assert result.startswith("ERROR:")
    assert "says nothing about the citation" in result
    assert parse_check(result) == "error"


def test_unknown_tool_output_parses_as_error():
    assert parse_check("Error running verify_citation: boom") == "error"
    assert parse_check("") == "error"
