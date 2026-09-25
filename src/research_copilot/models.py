"""Chat model factory.

CONCEPT: Chat models are Runnables
ChatAnthropic is LangChain's wrapper around the Anthropic Messages API. Like
every LangChain component (prompts, parsers, tools, whole chains), it implements
the `Runnable` interface: `.invoke()`, `.batch()`, `.stream()`, plus async
versions (`.ainvoke()`, ...). That shared interface is what lets
`prompt | model | parser` compose in chains.py.

    input:  a list of messages (a plain string or a PromptValue gets converted)
    output: an AIMessage

The rest of the code depends on "a chat model" (BaseChatModel), not on
Anthropic specifically. That's why tests can swap in a fake model, and why a
provider change would only touch this file.
"""

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel

from research_copilot.config import PROVIDERS, get_settings, require_anthropic_key, require_groq_key

# Anthropic server-side refusal fallback. If Claude's safety classifiers decline
# a request, the API retries it on a fallback model instead of returning
# stop_reason="refusal". Research questions will almost never trigger it. This
# is an Anthropic API feature. LangChain's own `.with_fallbacks()`, for outages
# and errors, comes in Phase 7.
_SERVER_SIDE_FALLBACK_BETA = "server-side-fallback-2026-07-01"


# --------------------------------------------------------------------------
# Phase 7: a second provider
# --------------------------------------------------------------------------
# CONCEPT: provider portability, finally exercised
# Phase 1's module docstring made a promise: "the rest of the code depends on
# 'a chat model' (BaseChatModel), not on Anthropic specifically ... a provider
# change would only touch this file." Every chain, agent, and summarizer in
# the project gets its model from `get_chat_model()`, so switching
# RESEARCH_COPILOT_PROVIDER switches all of them at once. Phase 7 is the first
# time that promise is tested.
#
# It is *mostly* true. Two places need to know the provider, and both are here:
#
#   1. construction - the class, the model id, the key, and provider-only
#      options (Anthropic's server-side refusal fallback does not exist on
#      Groq, so it is simply not sent there)
#   2. structured output - see `structured_output_method` below. That is
#      the one provider difference that reaches into agent code, and the
#      Supervisor asks this module rather than naming a method itself.
#
# What does *not* transfer, and is recorded rather than smoothed over: a
# behaviour verified on Groq is verified *on Groq*. Anthropic-specific plumbing
# (the fallback beta, `output_config.format`, adaptive thinking blocks) is not
# exercised by a Groq run at all. See the live-check report's item 1.

# Groq's own max completion default. Anthropic gets 16000 (Phase 1). Groq's
# free tier limits tokens *per minute* (8K for gpt-oss-120b), so a smaller cap
# keeps one runaway reply from eating a minute's budget. 4096 rather than
# 2048 because gpt-oss is a reasoning model: its hidden reasoning counts as
# completion tokens, and a cap that the reasoning alone can exhaust truncates
# the answer (or the JSON) that follows it.
_GROQ_DEFAULT_MAX_TOKENS = 4096

# Groq models that support `response_format: json_schema` (constrained or
# best-effort), per console.groq.com/docs/structured-outputs as of Sept 2026.
# Llama 3.3 70B is not one of them.
_GROQ_JSON_SCHEMA_MODELS = frozenset(
    {"openai/gpt-oss-20b", "openai/gpt-oss-120b", "openai/gpt-oss-safeguard-20b", "qwen/qwen3.8-27b"}
)


# --------------------------------------------------------------------------
# PHASE 7 PART D: keeping the SDK layer out of the call level's failures
# --------------------------------------------------------------------------
# The Groq SDK retries every 429. It honours retry-after only up to 60 s, and
# past that it backs off 0.5 -> 8 s. A daily-limit 429 says "try again in 48
# minutes", so the SDK's retries against it are guaranteed to fail and each one
# spends a request of the 1,000-per-day limit. The ownership table gives that
# failure to the call level, which falls back or stops the run. So the SDK
# must not touch it.
#
# The SDK honours a server header, `x-should-retry: false`. This transport adds
# it to daily-limit 429 responses before the SDK sees them. The SDK still
# retries per-minute 429s (row A) exactly as before. It is the narrowest way to
# move one failure out of the SDK's hands without taking the others with it.


def _is_daily_limit(response) -> bool:
    if response.status_code != 429:
        return False
    try:
        return " per day (" in response.read().decode("utf-8", "replace")
    except Exception:  # noqa: BLE001 - an unreadable body is not a daily limit
        return False


def _no_retry(response, request):
    """A copy of `response` carrying `x-should-retry: false`.

    `request` is passed in, not read from `response.request`: at the transport
    level httpx has not attached it yet. Reading it raises - and the SDK
    treats *that* as a connection error and retries it, which is the waste
    this transport exists to stop. Found by the fixture test, not by review.
    """
    import httpx

    headers = httpx.Headers(response.headers)
    headers["x-should-retry"] = "false"
    return httpx.Response(response.status_code, headers=headers, content=response.content,
                          request=request, extensions=response.extensions)


class QuotaAwareTransport:
    """httpx transport: mark daily-limit 429s as not-to-be-retried by the SDK."""

    def __init__(self, inner=None):
        import httpx

        self._inner = inner or httpx.HTTPTransport()

    def handle_request(self, request):
        response = self._inner.handle_request(request)
        return _no_retry(response, request) if _is_daily_limit(response) else response

    def close(self):
        self._inner.close()


class AsyncQuotaAwareTransport:
    """The async twin, for `ainvoke` / `astream_events` (Part C)."""

    def __init__(self, inner=None):
        import httpx

        self._inner = inner or httpx.AsyncHTTPTransport()

    async def handle_async_request(self, request):
        response = await self._inner.handle_async_request(request)
        if response.status_code == 429:
            await response.aread()
            if _is_daily_limit(response):
                return _no_retry(response, request)
        return response

    async def aclose(self):
        await self._inner.aclose()


def get_chat_model(
    *,
    max_tokens: int | None = None,
    server_side_fallback: bool = True,
    provider: str | None = None,
    model_name: str | None = None,
) -> BaseChatModel:
    """Build the chat model every chain and agent uses.

    `provider` defaults to the RESEARCH_COPILOT_PROVIDER setting ("anthropic").
    If you set RESEARCH_COPILOT_MODEL to an Anthropic model that rejects the
    `fallbacks` parameter, pass server_side_fallback=False.
    """
    settings = get_settings()
    provider = (provider or settings.provider).lower()
    if provider not in PROVIDERS:
        raise RuntimeError(
            f"Unknown provider {provider!r} (RESEARCH_COPILOT_PROVIDER); "
            f"expected one of {', '.join(PROVIDERS)}."
        )

    if provider == "groq":
        require_groq_key()
        # Imported here so the Anthropic path never needs langchain-groq
        # installed (it is an optional dependency: pip install -e ".[groq]").
        import httpx
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=model_name or settings.groq_model,
            max_tokens=max_tokens or _GROQ_DEFAULT_MAX_TOKENS,
            # Part D: the SDK owns per-minute 429s (row A of the ownership
            # table) and honours their retry-after, which is under 60 s. Its
            # default of 2 retries is enough. The 6 used during Part A were
            # never a deliberate choice, and on a *daily*-limit 429 every one
            # of them was guaranteed to fail (7 requests, 21 s, in the test).
            max_retries=2,
            # ...and the transport stops the SDK retrying daily-limit 429s at
            # all: those belong to the call level (resilience.py).
            http_client=httpx.Client(transport=QuotaAwareTransport()),
            http_async_client=httpx.AsyncClient(transport=AsyncQuotaAwareTransport()),
            # No temperature, for parity with the Anthropic path - the
            # comparison should differ by provider, not by sampling settings.
        )

    require_anthropic_key()
    extra = {}
    if server_side_fallback:
        # `betas` routes the request through the SDK's beta endpoint;
        # `model_kwargs` passes any extra API parameter through unchanged.
        extra = {
            "betas": [_SERVER_SIDE_FALLBACK_BETA],
            "model_kwargs": {"fallbacks": "default"},
        }

    return ChatAnthropic(
        model=model_name or settings.model,
        max_tokens=max_tokens or 16000,
        # No `temperature`: current Claude models reject sampling parameters.
        # They use adaptive thinking by default, deciding how much to reason
        # before answering. The reasoning comes back as "thinking" content
        # blocks on the AIMessage. Parsers skip those blocks, and the tool
        # loop has to send them back unchanged (see agent_loop.py).
        **extra,
    )


def structured_output_kwargs(model) -> dict:
    """The `with_structured_output(...)` keyword arguments for this model.

    Groq's json_schema models get `strict=True`: without it Groq's json_schema
    is "best effort" (asked for, not enforced), and the point of choosing
    json_schema is enforcement - the same class of guarantee Anthropic's
    `output_config.format` gives. Verified live on gpt-oss-120b in Phase 7 A1
    (SupervisorDecision parses in both modes).
    """
    method = structured_output_method(model)
    try:
        from langchain_groq import ChatGroq
    except ImportError:  # pragma: no cover
        ChatGroq = None
    if method == "json_schema" and ChatGroq is not None and isinstance(model, ChatGroq):
        return {"method": method, "strict": True}
    return {"method": method}


def structured_output_method(model) -> str:
    """Which `with_structured_output(method=...)` to use for this model.

    CONCEPT: the same schema, enforced three different ways
      json_schema       the provider constrains generation to the schema.
                        Anthropic (`output_config.format`), and Groq's
                        gpt-oss / Qwen models (with strict=True). Invalid
                        output cannot be produced.
                        NOT guaranteed: field order. The Supervisor puts
                        `rationale` before `next` so the route is written
                        after the reasoning (agents/supervisor.py). Phase 7
                        A1 found gpt-oss-120b emits `next` first even under
                        strict json_schema - a schema's key order is not a
                        generation order. Whether Anthropic honours it is
                        still to be checked with a real key.
      function_calling  the schema is offered as a tool and the model is made
                        to call it. Groq's Llama models (not on the free
                        tier; kept for paid keys). Output is parsed and
                        *validated* afterwards, not constrained, so an
                        invalid `next` is possible: it fails validation and
                        takes the Supervisor's fixed_policy fallback, logged.
                        Argument order is up to the model, so "rationale is
                        generated first" is no longer guaranteed.
      json_mode         valid JSON of any shape. Not used: weaker than both.

    Anthropic's forced tool calling is *not* used, because the Anthropic API
    rejects a forced tool choice when thinking is on (6.2's reason for
    json_schema). On Groq, forcing the tool is exactly what function_calling
    does, and it is fine there.

    Unknown objects (test fakes) get "json_schema", the method the project
    was designed around.
    """
    try:
        from langchain_groq import ChatGroq
    except ImportError:  # pragma: no cover - groq extra not installed
        ChatGroq = None
    if ChatGroq is not None and isinstance(model, ChatGroq):
        return "json_schema" if model.model_name in _GROQ_JSON_SCHEMA_MODELS else "function_calling"
    return "json_schema"


def get_fallback_model(*, max_tokens: int | None = None) -> BaseChatModel | None:
    """The model to switch to when the primary's daily quota is gone (Part D).

    Configured with RESEARCH_COPILOT_FALLBACK_MODEL as "provider:model", e.g.
    "groq:openai/gpt-oss-20b" (a separate daily bucket on Groq's free tier) or
    "anthropic:claude-sonnet-5". Unset means no fallback: a spent quota stops
    the run cleanly (QuotaExhausted).

    Only a spent daily quota triggers it - the one trigger observed. Outages
    (5xx, timeouts) are not observed, so they stay with the SDK's retries.
    """
    import os

    spec = (os.getenv("RESEARCH_COPILOT_FALLBACK_MODEL") or "").strip()
    if not spec:
        return None
    provider, _, name = spec.partition(":")
    if not name:
        raise RuntimeError(
            f"RESEARCH_COPILOT_FALLBACK_MODEL={spec!r}: expected 'provider:model', "
            "e.g. 'groq:openai/gpt-oss-20b'."
        )
    return get_chat_model(provider=provider, model_name=name, max_tokens=max_tokens)


def provider_of(model) -> str:
    """Which provider a model object belongs to - for per-model limits."""
    try:
        from langchain_groq import ChatGroq
    except ImportError:  # pragma: no cover
        ChatGroq = None
    if ChatGroq is not None and isinstance(model, ChatGroq):
        return "groq"
    if isinstance(model, ChatAnthropic):
        return "anthropic"
    return get_settings().provider


class ModelNotAvailable(RuntimeError):
    """The configured model does not exist for this key (row F: fail fast)."""


def check_model_available(provider: str | None = None, model_name: str | None = None) -> None:
    """Fail fast, before a run starts, if the configured model is not available.

    Part A's first live call hit `404 model_not_found` (Llama 3.3 70B is not on
    Groq's free tier) mid-run. This asks the provider's model list up front:
    one free call, no tokens. Never retried, and never papered over by a
    fallback - a wrong model name is a configuration error to fix.
    """
    settings = get_settings()
    provider = (provider or settings.provider).lower()
    if provider == "groq":
        require_groq_key()
        import groq

        name = model_name or settings.groq_model
        available = {m.id for m in groq.Groq(max_retries=1).models.list().data}
    else:
        require_anthropic_key()
        import anthropic

        # Documented API (Models API); not exercised live - no Anthropic key yet.
        name = model_name or settings.model
        available = {m.id for m in anthropic.Anthropic(max_retries=1).models.list()}
    if name not in available:
        raise ModelNotAvailable(
            f"model {name!r} is not available to this {provider} key. "
            f"Available: {', '.join(sorted(available)) or '(none)'}"
        )
