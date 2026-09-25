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


INTERVENTION_EVENT = "intervention"


def emit_intervention(record: dict) -> None:
    """Announce an intervention the moment it happens, as a custom stream event.
    (Phase 7 Part C)

    Interventions happen *inside* a node - a retry, a fallback, a trim - and a
    node's state update is only visible when the node finishes. Without this, a
    live stream would show less than the final state does: "the Critic is
    retrying with a hint" would only appear after the Critic had finished. So
    each one is also dispatched as a custom event (`on_custom_event`, name
    "intervention"), which `astream_events` delivers immediately, from inside
    sync nodes and subgraphs alike (measured in Part C). streaming.py turns it
    into a normalised event. The state write still happens, as before - the
    event is in addition to it, not instead of it.

    Outside any graph run there is no stream to announce to, and LangChain
    raises one specific RuntimeError. That, and only that, is ignored - see the
    README's lesson on handlers broad enough to swallow our own bugs.
    """
    from langchain_core.callbacks.manager import dispatch_custom_event

    try:
        dispatch_custom_event(INTERVENTION_EVENT, dict(record))
    except RuntimeError as exc:
        if "without a parent run id" not in str(exc):
            raise


@dataclass
class ModelCallFailure:
    """A model call that did not produce a usable message, even after recovery."""

    error: str
    attempts: int


# --------------------------------------------------------------------------
# PHASE 7 PART D: one classifier, built from real failures only
# --------------------------------------------------------------------------
# Every rule below was written against a real provider error body, captured in
# Phase 7's live runs and kept in tests/fixtures/provider_errors.json. Nothing
# here handles a failure that has not been observed. Unobserved kinds (5xx,
# timeouts, connection errors) are left to the provider SDK's own retries,
# unchanged - see the ownership table in the README ("Part D").
#
# One trap the real bodies exposed: Groq's 413 ("request too large") carries
# `code: rate_limit_exceeded`, the same code as both 429s. A classifier keyed on
# the error code would treat a request that can never succeed as "slow down and
# retry". So every rule keys on status and message, never on the code alone.

QUOTA_EXHAUSTED = "quota_exhausted"   # 429, a *daily* limit: nothing left until it refills
RATE_LIMITED = "rate_limited"         # 429, a per-minute limit: the SDK's to retry
TOO_LARGE = "too_large"               # 413 / "prompt is too long": shrink, do not wait
INVALID_TOOL = "invalid_tool"         # 400 tool_use_failed: the model's behaviour
MODEL_NOT_FOUND = "model_not_found"   # 404: configuration, not a runtime condition
OTHER = "other"


def classify(exc: BaseException) -> str:
    """Which row of the ownership table an exception belongs to."""
    text = str(exc)
    status = getattr(exc, "status_code", None)
    if status == 429 or "Error code: 429" in text:
        return QUOTA_EXHAUSTED if (" per day (" in text or "(TPD)" in text or "(RPD)" in text) else RATE_LIMITED
    if status == 413 or "Error code: 413" in text or "Request too large" in text or "prompt is too long" in text:
        return TOO_LARGE
    if "tool_use_failed" in text or "not in request.tools" in text:
        return INVALID_TOOL
    if "model_not_found" in text:
        return MODEL_NOT_FOUND
    return OTHER


class QuotaExhausted(RuntimeError):
    """The provider's daily limit is spent, and no fallback model took over.

    Raised, never degraded around (Part D decision). Once the quota is gone,
    every later call in the run fails the same way, so "degrade gracefully"
    would only be a slower, more expensive route to the same total failure.
    The run-level handler (CLI, live-check, and later the API) turns it into
    one clean message. The Supervisor must not swallow it into fixed_policy.
    """


def is_invalid_tool_call(exc: Exception) -> bool:
    """The model called a tool it was not given, and the provider refused it.

    Groq reports this as a 400 with code `tool_use_failed`. The Anthropic path
    does not raise this: an unknown tool name there reaches `_execute_tool_call`,
    which already answers it with an error ToolMessage (Phase 1).
    """
    return classify(exc) == INVALID_TOOL


def is_request_too_large(exc: Exception) -> bool:
    """The provider refused the request for its size (Phase 7). Groq: 413.
    Anthropic: 400 "prompt is too long" (documented; not yet observed)."""
    return classify(exc) == TOO_LARGE


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
    on_too_large: Callable[[], Sequence[BaseMessage]] | None = None,
    fallback: Callable[[], tuple[object, Sequence[BaseMessage]]] | None = None,
    on_intervention: Callable[[dict], None] | None = None,
) -> tuple[AIMessage | ModelCallFailure, int]:
    """Call `runnable`, and recover from the failures this layer owns.

    The call-level row of the ownership table (README, Part D):

      failure            action                                 bound
      -----------------  -------------------------------------  -------------
      invented tool      retry once with `note`                 one retry
      too large          retry once with `on_too_large()`       one retry
      daily quota        `fallback()` model, else QuotaExhausted  no retry of
                                                                the same model
      anything else      propagate (the SDK already retried what
                         it owns; the rest is a bug to surface)

    `fallback` returns (runnable, request) for the fallback model: its own
    binding (tools / structured output) and a request sized to its own limit,
    because both are resolved per model (Part B's note). `on_intervention`
    receives one record per recovery, so the calling agent can log it in state.

    Returns (message or failure, attempts made).
    """
    node = where.split(" ")[0]

    def record(entry: dict) -> None:
        # Part C: announce it live, then hand it to the agent for its state field.
        emit_intervention({"node": node, **entry})
        if on_intervention is not None:
            on_intervention(entry)

    def quota(exc: BaseException, attempts: int):
        resolved = fallback() if fallback is not None else None
        if resolved is None:
            raise QuotaExhausted(f"{where}: {exc}") from exc
        log.warning("%s: daily quota exhausted (%s); switching to the fallback model", where, exc)
        fb_runnable, fb_request = resolved
        record({"kind": "fallback_model", "detail": f"primary model's daily quota exhausted: {str(exc)[:160]}"})
        result, more = invoke_with_recovery(
            fb_runnable, fb_request, recoverable=recoverable, note=note,
            where=f"{where} (fallback)", on_intervention=record,
        )
        return result, attempts + more

    try:
        return runnable.invoke(list(request)), 1
    except Exception as exc:  # noqa: BLE001 - dispatched on classify() below
        kind = classify(exc)
        if kind == QUOTA_EXHAUSTED:
            return quota(exc, 1)
        first = exc
        if on_too_large is not None and kind == TOO_LARGE:
            log.warning("%s: request refused as too large (%s); retrying once, smaller", where, exc)
            retry_request = list(on_too_large())
            record({"kind": "too_large_retry", "detail": str(exc)[:160]})
        elif recoverable(exc):
            log.warning("%s: model call failed (%s); retrying once with a hint", where, exc)
            retry_request = _with_note(request, note)
            record({"kind": "hint_retry", "detail": str(exc)[:160]})
        else:
            raise
    try:
        return runnable.invoke(retry_request), 2
    except Exception as exc:  # noqa: BLE001
        kind = classify(exc)
        if kind == QUOTA_EXHAUSTED:
            return quota(exc, 2)
        if not (recoverable(exc) or kind == TOO_LARGE):
            raise
        log.warning("%s: retry failed too (%s); handing back a failure", where, exc)
        record({"kind": "gave_up", "detail": f"retry failed too: {str(exc)[:160]}"})
        return ModelCallFailure(error=f"{type(first).__name__}: {first} | retry: {exc}", attempts=2), 2


def model_fallback(bind, build, *, override=None):
    """Build the `fallback=` argument for `invoke_with_recovery`, per call site.

    `bind(model)` applies what this call needs - tools, structured output - to
    the fallback model. `build(limit)` sizes the request to *that model's*
    request limit. Both are re-resolved for the fallback because both depend on
    the model (Part B's note: structured-output method, strict mode and size
    limit are per provider). Returns None when no fallback is configured, which
    `invoke_with_recovery` turns into QuotaExhausted.
    """

    def make():
        from research_copilot.models import get_fallback_model, provider_of
        from research_copilot.request_budget import max_request_tokens

        model = override or get_fallback_model()
        if model is None:
            return None
        return bind(model), build(max_request_tokens(provider_of(model)))

    return make
