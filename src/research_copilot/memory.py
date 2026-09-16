"""Conversation memory: a multi-turn history that stays inside a budget.

CONCEPT: the model remembers nothing
Every call to a chat model is stateless. A "conversation" exists only because
you resend the whole message list each turn. Phase 1's tool loop already worked
this way: the list *is* the memory. Here that list becomes an object with a
budget and a pruning policy. This is deliberately the same shape LangGraph uses
for state (a dict holding a `messages` list), so Phase 3 can lift it into a
StateGraph with little change.

CONCEPT: why history needs a budget at all
The list grows every turn, and the tokens in it are billed and bounded:
  - cost: input tokens are charged on *every* turn, so an unbounded history
    makes each turn more expensive than the last
  - context window: history + new input eventually exceeds the window and the
    API rejects the request
  - quality: a long, mostly irrelevant history gives the model more to be
    distracted by

CONCEPT: why token count, not message count
"Keep the last 10 messages" is a proxy for the thing that matters, and a poor
one. One message might be 5 tokens ("thanks") or 5,000 (a pasted paper, or an
arXiv tool result with five abstracts). Ten messages could be 200 tokens or
50,000, so a message cap tells you nothing about what you'll pay or whether
you'll overflow the window. Tokens are the unit the API bills and limits, so
the budget is in tokens.

Two strategies, swappable via RESEARCH_COPILOT_MEMORY_STRATEGY:
  trim       drop the oldest messages until the history fits. Cheap and exact
             for what survives, but dropped turns are gone permanently.
  summarize  replace the oldest turns with one SystemMessage summarizing them.
             Keeps the gist at a fraction of the tokens, but costs an extra
             model call per compression and loses detail - and the summary is
             itself model output, so it can be wrong.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    trim_messages,
)
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.output_parsers import StrOutputParser

from research_copilot.config import get_settings
from research_copilot.models import get_chat_model
from research_copilot.prompts import SUMMARY_PROMPT

MemoryStrategy = Literal["trim", "summarize"]


@dataclass
class PruneReport:
    """What pruning did on one turn - printed by the chat CLI so the policy is visible."""

    strategy: MemoryStrategy
    tokens_before: int
    tokens_after: int
    messages_before: int
    messages_after: int
    summarized: bool = False

    def __str__(self) -> str:
        change = (
            "unchanged"
            if self.messages_before == self.messages_after
            else f"{self.messages_before}->{self.messages_after} messages, "
            f"{self.tokens_before}->{self.tokens_after} tokens"
        )
        suffix = " (summary updated)" if self.summarized else ""
        return f"{self.strategy}: {change}{suffix}"


@dataclass
class ConversationMemory:
    """The running message list, plus the policy that keeps it small.

    `messages` holds only the conversation turns (Human/AI). The persona lives in
    CHAT_PROMPT, and the optional summary is rendered as its own SystemMessage at
    prompt time, so pruning never has to worry about accidentally dropping the
    system prompt.
    """

    strategy: MemoryStrategy = "trim"
    max_tokens: int = 1200
    # How many recent messages the summarizer always keeps verbatim. Too low and
    # the model loses the thread of the current exchange; too high and
    # summarization never frees enough tokens to matter.
    keep_last_messages: int = 4
    summary_model: BaseChatModel | None = None
    # CONCEPT: approximate vs exact token counting
    # count_tokens_approximately is a local character-based estimate: free,
    # instant, and good enough to decide when to prune. The exact alternative is
    # the model itself (trim_messages accepts a BaseChatModel as token_counter,
    # and ChatAnthropic then calls Anthropic's count_tokens endpoint) - exact,
    # but a network round trip on every turn. Estimate to make a pruning
    # decision; count exactly only when a number is user-facing, like a bill.
    token_counter: Callable[[Sequence[BaseMessage]], int] = count_tokens_approximately
    messages: list[BaseMessage] = field(default_factory=list)
    summary: str | None = None

    def add_user_turn(self, text: str) -> None:
        self.messages.append(HumanMessage(content=text))

    def add_ai_turn(self, message: AIMessage) -> None:
        # Store the whole AIMessage, not just its text: it carries tool calls and
        # (with Claude) thinking blocks that have to go back unchanged.
        self.messages.append(message)

    def token_count(self) -> int:
        return self.token_counter(self._summary_messages() + self.messages)

    def prompt_variables(self) -> dict[str, list[BaseMessage]]:
        """The variables CHAT_PROMPT expects."""
        variables: dict[str, list[BaseMessage]] = {"history": list(self.messages)}
        summary_messages = self._summary_messages()
        if summary_messages:
            variables["earlier_summary"] = summary_messages
        return variables

    def prune(self) -> PruneReport:
        """Apply the configured strategy. Call this after each completed turn."""
        tokens_before, messages_before = self.token_count(), len(self.messages)
        summarized = False
        if self.strategy == "summarize":
            summarized = self._summarize()
        else:
            self._trim()
        return PruneReport(
            strategy=self.strategy,
            tokens_before=tokens_before,
            tokens_after=self.token_count(),
            messages_before=messages_before,
            messages_after=len(self.messages),
            summarized=summarized,
        )

    def _summary_messages(self) -> list[BaseMessage]:
        if not self.summary:
            return []
        return [
            SystemMessage(content=f"Summary of earlier conversation:\n{self.summary}")
        ]

    def _trim(self) -> None:
        # CONCEPT: trim_messages
        # A Runnable-friendly helper that drops messages until the list fits the
        # budget. The options encode rules the API cares about:
        #   strategy="last"      keep the newest turns (the past is what goes)
        #   start_on="human"     never leave the history starting on an AI reply;
        #                        the API expects the first turn to be the user's
        #   include_system=False the persona isn't in this list to begin with
        #   allow_partial=False  never cut a message in half - half a tool result
        #                        or half an answer is worse than none
        self.messages = trim_messages(
            self.messages,
            max_tokens=self.max_tokens,
            token_counter=self.token_counter,
            strategy="last",
            start_on="human",
            include_system=False,
            allow_partial=False,
        )

    def _summarize(self) -> bool:
        if self.token_count() <= self.max_tokens:
            return False
        older, recent = self._split_for_summary()
        if not older:
            return False

        model = self.summary_model or get_chat_model(max_tokens=1024)
        # An ordinary LCEL chain - the summarizer is just another prompt|model|parser.
        chain = SUMMARY_PROMPT | model | StrOutputParser()
        self.summary = chain.invoke(
            {
                "previous_summary": self.summary or "(none)",
                "conversation": render_messages(older),
            }
        ).strip()
        self.messages = recent
        return True

    def _split_for_summary(self) -> tuple[list[BaseMessage], list[BaseMessage]]:
        """Split into (to summarize, to keep verbatim), keeping a valid start."""
        cut = max(0, len(self.messages) - self.keep_last_messages)
        # The kept tail must begin with a human turn, so walk the cut forward
        # until it lands on one.
        while cut < len(self.messages) and not isinstance(
            self.messages[cut], HumanMessage
        ):
            cut += 1
        return self.messages[:cut], self.messages[cut:]


def render_messages(messages: Sequence[BaseMessage]) -> str:
    """Flatten messages into plain text for the summarizer to read."""
    labels = {"human": "User", "ai": "Assistant", "system": "System", "tool": "Tool"}
    lines = []
    for message in messages:
        text = message.text.strip()
        if text:
            lines.append(f"{labels.get(message.type, message.type)}: {text}")
    return "\n".join(lines)


def memory_from_settings(
    *, strategy: MemoryStrategy | None = None, max_tokens: int | None = None, **kwargs
) -> ConversationMemory:
    """Build memory from .env, with optional CLI overrides."""
    settings = get_settings()
    return ConversationMemory(
        strategy=strategy or settings.memory_strategy,
        max_tokens=max_tokens or settings.max_history_tokens,
        **kwargs,
    )
