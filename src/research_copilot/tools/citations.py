"""verify_citation: does an arXiv paper with this id exist? (Phase 6.3)

The Critic's tool, and the upgrade Phase 5 flagged. Phase 5's critic could only
*doubt* a citation ("is 2309.15217 real?"). This lets it *check*.

It is deliberately narrow. It takes one id, returns found / not found /
invalid / error, and on "found" gives the title so the Critic can see whether
the draft attributes the right paper to that id. It does not search. A Critic
that can search is a second Researcher whose findings nobody reviews - the same
reason Phase 5 bound no tools to its critic at all. Verifying a named
reference is judging the draft. Looking for new references is doing the
research.

CONCEPT: a tool result the code can parse, not just the model
Every result starts with a fixed status word:

    FOUND: 2309.15217 - RAGAS: Automated Evaluation of Retrieval Augmented ...
    NOT FOUND: no arXiv paper has the id 2401.99999
    INVALID: '12345' is not an arXiv id (expected e.g. 2309.15217)
    ERROR: lookup failed (network error: ...). This says nothing about the citation.

The model reads the whole line. The Critic subgraph's `compile_verdict`
reads the first word, and lifts the results into `citation_checks` as structured
data (see multi_agent_state.CitationCheck). It is the same move as
`research_outcome`: facts code will branch on should not have to be recovered
from prose.

CONCEPT: a failed lookup is not a missing paper
ERROR is its own status, and the wording says outright that it is not evidence
against the citation. arXiv rate-limits and times out. A Critic that read
"lookup failed" as "citation is fake" would reject correct drafts whenever the
network hiccupped, and every false rejection spends a revision.
"""

import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from langchain_core.tools import tool

# Reuse Phase 1's throttle and parser. arXiv's "one request every 3 seconds" is
# per client, not per tool, so verify_citation and search_arxiv must share one
# rate limiter or together they would exceed it.
from research_copilot.tools import arxiv as _arxiv

# New-style ids (2309.15217, with optional version) and old-style ones
# (cs/0112017, hep-th/9901001). Anything else is INVALID without a network
# call.
_NEW_ID = re.compile(r"\b(\d{4}\.\d{4,5})(v\d+)?\b")
_OLD_ID = re.compile(r"\b([a-z\-]+(?:\.[A-Z]{2})?/\d{7})(v\d+)?\b")


def normalize_arxiv_id(text: str) -> str | None:
    """Pull a bare arXiv id out of whatever the draft cited.

    Accepts the id alone, with a version suffix, or inside a URL
    (arxiv.org/abs/..., arxiv.org/pdf/....pdf). The version is dropped: a paper
    exists if any version does, and the draft's "v2" is not what is in doubt.
    """
    text = (text or "").strip()
    for pattern in (_NEW_ID, _OLD_ID):
        match = pattern.search(text)
        if match:
            return match.group(1)
    return None


def _fetch_by_id(arxiv_id: str) -> str:
    """One `id_list` query, under Phase 1's shared throttle.

    Separate from `_arxiv._fetch` only because that function builds a
    `search_query`, and an exact-id lookup is `id_list`. Everything else - the
    lock, the spacing, the plain-text rate-limit body - is handled the same way.
    """
    params = urllib.parse.urlencode({"id_list": arxiv_id, "max_results": 1})
    request = urllib.request.Request(
        f"{_arxiv.ARXIV_API_URL}?{params}", headers={"User-Agent": _arxiv._USER_AGENT}
    )
    with _arxiv._rate_lock:
        wait = _arxiv._MIN_SECONDS_BETWEEN_REQUESTS - (time.monotonic() - _arxiv._last_request_at)
        if wait > 0:
            time.sleep(wait)
        try:
            with urllib.request.urlopen(request, timeout=_arxiv._REQUEST_TIMEOUT_SECONDS) as response:
                body = response.read().decode("utf-8")
        finally:
            _arxiv._last_request_at = time.monotonic()
    if not body.lstrip().startswith("<"):
        raise ValueError(body.strip()[:200] or "empty response")
    return body


def check_arxiv_id(raw: str) -> str:
    """The tool's logic, as a plain function (tests call it with a stubbed fetch)."""
    arxiv_id = normalize_arxiv_id(raw)
    if arxiv_id is None:
        return f"INVALID: {raw!r} is not an arXiv id (expected e.g. 2309.15217)"
    try:
        papers = _arxiv.parse_arxiv_feed(_fetch_by_id(arxiv_id))
    except ValueError as exc:
        # arXiv answers a malformed id with an error entry, which the Phase 1
        # parser raises as "arXiv rejected the query". The regex already
        # filtered the obvious cases, so this is an id arXiv considers invalid.
        if "rejected" in str(exc):
            return f"INVALID: arXiv rejected {arxiv_id!r} as an id"
        return f"ERROR: lookup failed ({exc}). This says nothing about the citation."
    except (urllib.error.URLError, TimeoutError, ET.ParseError) as exc:
        return f"ERROR: lookup failed ({type(exc).__name__}: {exc}). This says nothing about the citation."

    # A well-formed id that does not exist comes back as an empty feed, or as
    # an entry with no title - both mean "no such paper".
    papers = [p for p in papers if p.title]
    if not papers:
        return f"NOT FOUND: no arXiv paper has the id {arxiv_id}"
    return f"FOUND: {arxiv_id} - {papers[0].title}"


@tool(parse_docstring=True)
def verify_citation(arxiv_id: str) -> str:
    """Check that an arXiv paper with this id exists, and get its title.

    Use it on each arXiv citation in the draft you are reviewing, to confirm the
    paper is real and that the draft attributes the right paper to the id. It
    does not search: pass an id or an arXiv URL, not keywords.

    Args:
        arxiv_id: An arXiv id such as "2309.15217" or an arXiv URL such as
            "http://arxiv.org/abs/2309.15217v2".
    """
    return check_arxiv_id(arxiv_id)


def parse_check(tool_output: str) -> str:
    """Map a verify_citation result to a CitationCheck status."""
    head = (tool_output or "").split(":", 1)[0].strip().upper()
    return {
        "FOUND": "found",
        "NOT FOUND": "not_found",
        "INVALID": "invalid",
    }.get(head, "error")


# Phase 7 A1: citations are now extracted and checked by code before the
# Critic judges (agents/critic.py). These find every arXiv reference a draft
# makes: bare ids, arxiv.org URLs, "arXiv:" prefixes - and malformed
# "arXiv:12345"-style references, which must reach the checker so they come
# back INVALID rather than being silently skipped.
_CITED = re.compile(
    r"arxiv\.org/(?:abs|pdf)/(?P<url>[^\s\]\)\},;]+?)(?:\.pdf)?(?=[\s\]\)\},;]|$)"
    r"|arXiv:\s*(?P<prefixed>[^\s\]\)\},;]+)"
    r"|\b(?P<bare>\d{4}\.\d{4,5}(?:v\d+)?)\b",
    re.IGNORECASE,
)

MAX_CITATIONS_CHECKED = 10


def extract_citations(text: str) -> list[str]:
    """Every arXiv reference in `text`, in order of first appearance, one per
    paper (versions and URL/id duplicates collapse to one). Capped, because each
    lookup waits on arXiv's 3-second throttle."""
    seen: set[str] = set()
    found: list[str] = []
    for match in _CITED.finditer(text or ""):
        raw = (match.group("url") or match.group("prefixed") or match.group("bare") or "").rstrip(".")
        key = normalize_arxiv_id(raw) or raw
        if key and key not in seen:
            seen.add(key)
            found.append(key)
    return found[:MAX_CITATIONS_CHECKED]
