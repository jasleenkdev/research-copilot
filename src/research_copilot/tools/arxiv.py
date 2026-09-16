"""arXiv search tool.

CONCEPT: Tools
A tool is a function plus a description the model can read: a name, a
description, and a JSON Schema for its arguments. The model never runs your
code. It asks you to, by emitting a tool call `{name, args, id}`. You run the
function and send the result back (see agent_loop.py).

The @tool decorator builds that schema from the function itself:
  - the function name       -> tool name
  - type hints and defaults -> argument types, required vs. optional
  - the docstring           -> tool description, and with
                               parse_docstring=True the "Args:" section becomes
                               per-argument descriptions
So the docstring is prompt text. Write it for the model. Print
`search_arxiv.tool_call_schema.model_json_schema()` to see exactly what the model
receives.

Why arXiv: the API is free and needs no key. Its terms ask for at most one
request every 3 seconds, and it sometimes answers "Rate exceeded", so this tool
throttles itself and returns failures as text instead of raising.
"""

import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass

from langchain_core.tools import tool

ARXIV_API_URL = "https://export.arxiv.org/api/query"
_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
_MIN_SECONDS_BETWEEN_REQUESTS = 3.0
_REQUEST_TIMEOUT_SECONDS = 20
_USER_AGENT = "research-copilot/0.1 (learning project)"

_rate_lock = threading.Lock()
_last_request_at = 0.0


@dataclass(frozen=True)
class Paper:
    title: str
    authors: list[str]
    published: str  # YYYY-MM-DD
    summary: str
    url: str


def build_search_query(query: str) -> str:
    """Convert free text to arXiv query syntax. Queries already using fields pass through."""
    if ":" in query:  # e.g. "ti:attention AND cat:cs.CL"
        return query
    return " AND ".join(f"all:{term}" for term in query.split())


def _clean(text: str | None) -> str:
    return " ".join((text or "").split())


def parse_arxiv_feed(xml_text: str) -> list[Paper]:
    """Parse the Atom XML the arXiv API returns."""
    root = ET.fromstring(xml_text)
    papers = []
    for entry in root.findall("atom:entry", _ATOM_NS):
        url = _clean(entry.findtext("atom:id", namespaces=_ATOM_NS))
        summary = _clean(entry.findtext("atom:summary", namespaces=_ATOM_NS))
        # For a malformed query, arXiv returns a feed containing one "error" entry.
        if "/api/errors" in url:
            raise ValueError(f"arXiv rejected the query: {summary}")
        papers.append(
            Paper(
                title=_clean(entry.findtext("atom:title", namespaces=_ATOM_NS)),
                authors=[
                    _clean(author.findtext("atom:name", namespaces=_ATOM_NS))
                    for author in entry.findall("atom:author", _ATOM_NS)
                ],
                published=_clean(
                    entry.findtext("atom:published", namespaces=_ATOM_NS)
                )[:10],
                summary=summary,
                url=url,
            )
        )
    return papers


def format_papers(papers: list[Paper], max_summary_chars: int = 600) -> str:
    """Render papers as compact text for the model to read."""
    if not papers:
        return "No arXiv papers matched that query. Try fewer or different keywords."
    blocks = []
    for i, paper in enumerate(papers, start=1):
        authors = ", ".join(paper.authors[:3])
        if len(paper.authors) > 3:
            authors += " et al."
        summary = paper.summary
        if len(summary) > max_summary_chars:
            summary = summary[:max_summary_chars].rsplit(" ", 1)[0] + "…"
        blocks.append(
            f"[{i}] {paper.title}\n"
            f"    Authors: {authors}\n"
            f"    Published: {paper.published}\n"
            f"    URL: {paper.url}\n"
            f"    Abstract: {summary}"
        )
    return "\n\n".join(blocks)


def _fetch(search_query: str, max_results: int) -> str:
    global _last_request_at
    params = urllib.parse.urlencode(
        {
            "search_query": search_query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
        }
    )
    request = urllib.request.Request(
        f"{ARXIV_API_URL}?{params}", headers={"User-Agent": _USER_AGENT}
    )
    with _rate_lock:
        wait = _MIN_SECONDS_BETWEEN_REQUESTS - (time.monotonic() - _last_request_at)
        if wait > 0:
            time.sleep(wait)
        try:
            with urllib.request.urlopen(
                request, timeout=_REQUEST_TIMEOUT_SECONDS
            ) as response:
                body = response.read().decode("utf-8")
        finally:
            _last_request_at = time.monotonic()
    # When rate limiting, arXiv can send a short plain-text body instead of XML.
    if not body.lstrip().startswith("<"):
        raise ValueError(body.strip()[:200] or "empty response")
    return body


@tool(parse_docstring=True)
def search_arxiv(query: str, max_results: int = 5) -> str:
    """Search arXiv for academic papers. Returns title, authors, date, URL, and abstract for each.

    Use for questions about research literature or recent methods in CS, ML,
    physics, math, statistics, and related fields, or when specific papers
    should be cited. Results are ranked by relevance, not recency.

    Args:
        query: A few keywords, e.g. "retrieval augmented generation evaluation".
            Every keyword must match, so keep it short. arXiv field syntax also
            works, e.g. "ti:attention AND cat:cs.CL".
        max_results: How many papers to return, from 1 to 10.
    """
    # CONCEPT: Tool errors are observations, not crashes
    # If the search fails, the model should find out and adapt: retry with other
    # keywords, or answer without sources and say so. Returning the error as the
    # tool result makes that possible. Raising would end the whole agent run.
    max_results = max(1, min(max_results, 10))
    try:
        xml_text = _fetch(build_search_query(query), max_results)
        return format_papers(parse_arxiv_feed(xml_text))
    except urllib.error.HTTPError as exc:
        return (
            f"arXiv search failed (HTTP {exc.code}). The API may be rate limiting. "
            "Try once more, or answer without new sources and say so."
        )
    except (urllib.error.URLError, TimeoutError) as exc:
        return f"arXiv search failed (network error: {exc}). Answer without new sources and say so."
    except (ET.ParseError, ValueError) as exc:
        return f"arXiv search failed: {exc}"
