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


def get_chat_model(
    *,
    max_tokens: int | None = None,
    server_side_fallback: bool = True,
    provider: str | None = None,
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
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=settings.groq_model,
            max_tokens=max_tokens or _GROQ_DEFAULT_MAX_TOKENS,
            # The free tier limits tokens per minute. A multi-agent run can
            # cross that inside one turn, so a 429 is expected, not exceptional.
            # The Groq SDK retries 429s itself, honouring retry-after.
            max_retries=6,
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
        model=settings.model,
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

