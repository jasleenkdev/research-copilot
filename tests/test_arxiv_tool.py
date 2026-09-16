import urllib.error

import pytest

from research_copilot.tools import arxiv
from research_copilot.tools.arxiv import (
    build_search_query,
    format_papers,
    parse_arxiv_feed,
    search_arxiv,
)

SAMPLE_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2005.11401v4</id>
    <published>2020-05-22T17:40:40Z</published>
    <title>Retrieval-Augmented Generation for
      Knowledge-Intensive NLP Tasks</title>
    <summary>  Large pre-trained language models have
      a limited ability to access and manipulate knowledge.  </summary>
    <author><name>Patrick Lewis</name></author>
    <author><name>Ethan Perez</name></author>
    <author><name>Aleksandra Piktus</name></author>
    <author><name>Fabio Petroni</name></author>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/1706.03762v7</id>
    <published>2017-06-12T17:57:34Z</published>
    <title>Attention Is All You Need</title>
    <summary>The dominant sequence transduction models...</summary>
    <author><name>Ashish Vaswani</name></author>
  </entry>
</feed>"""

ERROR_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/api/errors#incorrect_id_format</id>
    <title>Error</title>
    <summary>incorrect id format</summary>
  </entry>
</feed>"""


def test_parse_normalizes_whitespace_and_fields():
    papers = parse_arxiv_feed(SAMPLE_FEED)
    assert len(papers) == 2
    first = papers[0]
    assert first.title == "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
    assert first.published == "2020-05-22"
    assert first.url == "http://arxiv.org/abs/2005.11401v4"
    assert first.summary.startswith("Large pre-trained language models have a limited")
    assert len(first.authors) == 4


def test_parse_raises_on_api_error_entry():
    with pytest.raises(ValueError, match="incorrect id format"):
        parse_arxiv_feed(ERROR_FEED)


def test_format_papers_abbreviates_authors_and_truncates():
    text = format_papers(parse_arxiv_feed(SAMPLE_FEED), max_summary_chars=20)
    assert "Patrick Lewis, Ethan Perez, Aleksandra Piktus et al." in text
    assert "[2] Attention Is All You Need" in text
    assert "…" in text


def test_format_papers_empty():
    assert "No arXiv papers matched" in format_papers([])


def test_build_search_query():
    assert build_search_query("rag evaluation") == "all:rag AND all:evaluation"
    assert build_search_query("ti:attention AND cat:cs.CL") == "ti:attention AND cat:cs.CL"


def test_tool_schema_comes_from_signature_and_docstring():
    schema = search_arxiv.tool_call_schema.model_json_schema()
    assert search_arxiv.name == "search_arxiv"
    assert search_arxiv.description.startswith("Search arXiv for academic papers")
    assert schema["required"] == ["query"]
    assert "keywords" in schema["properties"]["query"]["description"]
    assert schema["properties"]["max_results"]["default"] == 5


def test_tool_invokes_fetch_and_formats(monkeypatch):
    calls = []

    def fake_fetch(search_query, max_results):
        calls.append((search_query, max_results))
        return SAMPLE_FEED

    monkeypatch.setattr(arxiv, "_fetch", fake_fetch)
    result = search_arxiv.invoke({"query": "rag", "max_results": 50})
    assert calls == [("all:rag", 10)]  # max_results clamped
    assert "Attention Is All You Need" in result


def test_tool_returns_errors_as_text(monkeypatch):
    def rate_limited(search_query, max_results):
        raise urllib.error.HTTPError(arxiv.ARXIV_API_URL, 429, "Too Many Requests", None, None)

    monkeypatch.setattr(arxiv, "_fetch", rate_limited)
    result = search_arxiv.invoke({"query": "rag"})
    assert "HTTP 429" in result
