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
from research_copilot.config import get_settings, require_anthropic_key

# Anthropic server-side refusal fallback. If Claude's safety classifiers decline
# a request, the API retries it on a fallback model instead of returning
# stop_reason="refusal". Research questions will almost never trigger it. This
# is an Anthropic API feature. LangChain's own `.with_fallbacks()`, for outages
# and errors, comes in Phase 7.
_SERVER_SIDE_FALLBACK_BETA = "server-side-fallback-2026-07-01"


def get_chat_model(
    *, max_tokens: int = 16000, server_side_fallback: bool = True
) -> ChatAnthropic:
    """Build the chat model every chain and agent uses.

    If you set RESEARCH_COPILOT_MODEL to a model that rejects the `fallbacks`
    parameter, pass server_side_fallback=False.
    """
    require_anthropic_key()
    settings = get_settings()

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
        max_tokens=max_tokens,
        # No `temperature`: current Claude models reject sampling parameters.
        # They use adaptive thinking by default, deciding how much to reason
        # before answering. The reasoning comes back as "thinking" content
        # blocks on the AIMessage. Parsers skip those blocks, and the tool
        # loop has to send them back unchanged (see agent_loop.py).
        **extra,
    )
