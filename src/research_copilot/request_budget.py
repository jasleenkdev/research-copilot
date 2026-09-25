"""A ceiling on how big one model request may get. (Phase 7, A1)

Found by E2E01's third attempt on Groq: `413 Request too large ... Requested
8849` against a free-tier limit of 8,000 tokens per request. Nothing in the
graph capped the size of a single request. Phase 4 pruned the *conversation*,
and the Supervisor excerpts what it is shown, but an agent's own request - a
Researcher's accumulated search results, a Critic's full notes plus draft plus
lookups - grew as large as the work made it. On Anthropic the ceiling is far
higher, so the same gap would have stayed hidden until one big enough pass
tripped it in production. It is a real gap on every provider, and a small
free-tier limit just exposed it first.

CONCEPT: size the request before sending it, and shrink only what may shrink
Each agent knows which parts of its request are *load-bearing* and which are
*material*:

    load-bearing  instructions, the question, the draft being judged - never cut
    material      search results, research notes - cut from the tail, with a
                  visible marker, when the request would not fit

`fit_text` does the cutting. The agent decides what to pass it. A Critic judging
half a draft is worse than one judging a draft against half the notes, and only
the agent knows which of its inputs that is.

CONCEPT: a trim is an intervention, and interventions are logged
Truncated context changes what the model sees as surely as an overridden route
changes where the run goes. So every trim returns an `Intervention` (kind "trim"), and each agent
writes it to the interventions field it owns (`researcher_interventions`,
`writer_interventions`, `critic_interventions`) - the same field its retries
and fallbacks go to (Part D). The Supervisor is shown them as facts, and the final state
lists them - the same visibility as the Supervisor's overrides, the Critic's
fail-closed parsing, and `fixed_policy`.

CONCEPT: the estimate is approximate, so there is a second line of defence
Sizes are estimated with `count_tokens_approximately` (about 4 characters per
token), not the provider's tokenizer. If the provider still refuses a request
as too large, `resilience.invoke_with_recovery` lets the agent rebuild it at 70%
of the budget and try once more (`on_too_large`). Past that, the agent
degrades as for any failed call.
"""

import os
from typing import TypedDict

from langchain_core.messages import BaseMessage
from langchain_core.messages.utils import count_tokens_approximately

# Per-provider defaults, in estimated tokens.
#   groq        the free tier's per-minute limit is 8,000 tokens, and a single
#               request larger than that can never be served. 6,500 leaves room
#               for estimation error (the 413 counted 8,849).
#   anthropic   far below the 1M context window. A production ceiling, not a
#               model limit: a request this big is a runaway, and it costs
#               money in proportion to its size.
DEFAULT_MAX_REQUEST_TOKENS = {"groq": 6_500, "anthropic": 100_000}

# The shrink factor for the one retry after a provider still says "too large".
RETRY_FACTOR = 0.7

TRIM_MARKER = "\n[... {cut} characters cut to fit the request size limit ...]"


class Intervention(TypedDict, total=False):
    """One intervention on an agent's attempt (Phase 7 Part D).

    One record type for everything that changed what an agent saw or which
    model answered it, so the Supervisor, the CLI and the harness read one
    field per agent, not several. `kind` says which:

      trim             part of the request was cut to fit the size limit
                       (`part`, `tokens_before`, `tokens_after`, `limit`)
      hint_retry       retried once with a hint (e.g. after an invented tool)
      too_large_retry  the provider refused the size; retried once, smaller
      fallback_model   the primary's daily quota was spent; the fallback answered
      gave_up          the retry failed too; the agent degraded

    Until Part D these were `TrimRecord`s in `<agent>_trims` fields, introduced
    in Part A and merged here while no real thread depended on the old names.
    """

    node: str
    kind: str
    part: str
    tokens_before: int
    tokens_after: int
    limit: int
    detail: str


def max_request_tokens(provider: str | None = None) -> int:
    """The per-request ceiling for the current provider. Override with
    RESEARCH_COPILOT_MAX_REQUEST_TOKENS."""
    override = os.getenv("RESEARCH_COPILOT_MAX_REQUEST_TOKENS")
    if override:
        return int(override)
    if provider is None:
        from research_copilot.config import get_settings

        provider = get_settings().provider
    return DEFAULT_MAX_REQUEST_TOKENS.get(provider, DEFAULT_MAX_REQUEST_TOKENS["anthropic"])


def estimate(messages: list[BaseMessage]) -> int:
    return count_tokens_approximately(messages)


def text_tokens(text: str) -> int:
    return count_tokens_approximately([("user", text or "")])


def fit_text(text: str, available_tokens: int) -> tuple[str, int, int]:
    """Cut `text` from the tail so it fits `available_tokens` (estimated).

    Returns (text, tokens_before, tokens_after). Keeps the head, because notes
    and search results put their most relevant material first. Never cuts below
    a few hundred characters: a request that cannot fit even then is left to
    the provider's refusal and the retry path, not silently emptied.
    """
    before = text_tokens(text)
    if before <= available_tokens:
        return text, before, before
    keep = max(400, available_tokens * 4 - len(TRIM_MARKER) - 20)
    if keep >= len(text):
        return text, before, before
    cut = text[:keep] + TRIM_MARKER.format(cut=len(text) - keep)
    return cut, before, text_tokens(cut)


def trim_record(node: str, part: str, before: int, after: int, limit: int) -> Intervention:
    return {"node": node, "kind": "trim", "part": part, "tokens_before": before,
            "tokens_after": after, "limit": limit}


def describe_intervention(i: dict) -> str:
    """One line for any kind of intervention - the only formatter.

    Every reader (the Supervisor's view, the CLI, the live-check report) goes
    through this. Part D found why: the Supervisor's view formatted every
    entry as a trim, a hint-retry entry has no token counts, the KeyError was
    caught by the Supervisor's own error handling, and the run silently fell
    back to fixed_policy. A formatting assumption became a routing change.
    """
    kind = i.get("kind", "trim")
    node = i.get("node", "?")
    if kind == "trim":
        return (f"{node} trimmed {i.get('part', '?')} {i.get('tokens_before', '?')}->"
                f"{i.get('tokens_after', '?')} tokens (limit {i.get('limit', '?')})")
    detail = (i.get("detail") or "")[:120]
    return f"{node} {kind}" + (f": {detail}" if detail else "")
