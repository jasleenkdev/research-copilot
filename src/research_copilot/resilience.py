"""Recovering from a failed model call inside a node. (Phase 7 - Part D, pulled forward)

Phase 7's live pass (A1, Groq gpt-oss-120b) found the first concrete case. The
Researcher's model asked for a tool called `open_file`: gpt-oss is trained
with a built-in browsing tool, and it reached for it. The provider rejected
the request outright (`400 tool_use_failed: attempted to call tool 'open_file'
which was not in request.tools`). Nothing caught that inside the Researcher, so
one bad call ended the whole multi-agent run.

This module is the start of Part D's answer, built for that one case but
shaped for all of them:

    invoke_with_recovery(runnable, request, recoverable=..., note=...)

  1. call the model
  2. if the failure is *recoverable* - a kind the model can plausibly fix
     itself, given a hint - retry ONCE, with `note` added to the instructions
  3. if it fails again, or the failure is not recoverable, return a
     `ModelCallFailure` instead of raising. The node decides what a failed
     call means for its own output (the Researcher hands over what it has;
     the Critic fails closed), because only the node knows what "degraded but
     still useful" looks like for its job

CONCEPT: three layers of "fallback", and how this one relates to the others
Part D will name all of them. This is the middle one:

  API level     Anthropic's server-side refusal fallback (models.py): the
                *provider* re-runs a refused request on another model.
  call level    this module: *our* code retries one model call, with a hint,
                inside the node that made it. Part D adds transient-error
                retries (rate limits, 5xx) and fallback models here.
  routing level the Supervisor's `fixed_policy` (6.2): when the Supervisor's
                *output* is unusable, the *graph* takes the known-good route.
                It is not a model call being retried - a decision being
                replaced.

They do not overlap. A call-level recovery that succeeds is invisible to the
routing level. A Supervisor call that still fails after call-level recovery
is what routing-level fallback exists for.

Only one retry, deliberately. A model that invents a tool twice with an
explicit "you have no such tool" note in front of it will not do better on a
third try, and on a free tier each attempt spends thousands of tokens.
"""

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage

log = logging.getLogger(__name__)


@dataclass
class ModelCallFailure:
    """A model call that did not produce a usable message, even after recovery."""

    error: str
    attempts: int


def is_invalid_tool_call(exc: Exception) -> bool:
    """The model called a tool it was not given, and the provider refused it.

    Groq reports this as a 400 with code `tool_use_failed`. Recognised by the
    message rather than by exception class, so that the check does not import
    every provider's SDK. The Anthropic path does not raise this: an unknown
    tool name there reaches `_execute_tool_call`, which already answers it
    with an error ToolMessage (Phase 1).
    """
    text = str(exc)
    return "tool_use_failed" in text or "not in request.tools" in text


def _with_note(request: Sequence[BaseMessage], note: str) -> list[BaseMessage]:
    """Insert `note` after the leading system messages.

    Not appended at the end: a system message after the conversation reads
    as a new instruction about the next turn on some providers, and is
    rejected on others. Next to the other instructions it is just one more
    instruction.
    """
    request = list(request)
    i = 0
    while i < len(request) and isinstance(request[i], SystemMessage):
        i += 1
    return [*request[:i], SystemMessage(content=note), *request[i:]]


def invoke_with_recovery(
    runnable,
    request: Sequence[BaseMessage],
    *,
    recoverable: Callable[[Exception], bool],
    note: str,
    where: str,
) -> tuple[AIMessage | ModelCallFailure, int]:
    """Call `runnable`; on a recoverable failure retry once with `note`.

    Returns (message or failure, attempts made). Non-recoverable exceptions
    propagate unchanged - a missing API key or a malformed request is a bug to
    surface, not a condition to paper over.
    """
    try:
        return runnable.invoke(list(request)), 1
    except Exception as exc:  # noqa: BLE001 - filtered by `recoverable`
        if not recoverable(exc):
            raise
        log.warning("%s: model call failed (%s); retrying once with a hint", where, exc)
        first = exc
    try:
        return runnable.invoke(_with_note(request, note)), 2
    except Exception as exc:  # noqa: BLE001
        if not recoverable(exc):
            raise
        log.warning("%s: retry failed too (%s); handing back a failure", where, exc)
        return ModelCallFailure(error=f"{type(first).__name__}: {first} | retry: {exc}", attempts=2), 2
