"""Phase 4's `prune_history`, lifted out so the multi-agent graph can use it. (6.4)

The logic is Phase 4's, line for line in substance: pick the survivors with
`trim_messages`, express everything else as `RemoveMessage`, and under the
"summarize" strategy fold the dropped turns into `summary` with Phase 2's
SUMMARY_PROMPT. The long explanation - why delete rather than filter, why
"deleted from state" is not "deleted from disk", why prune at the entry and
never inside a tool loop - is in graph.py's Part B docstring and
`prune_history`'s own docstring. It is not repeated here.

Why a copy rather than a refactor of graph.py: Phase 4's node is a closure
inside `build_graph`, and the rule for this project is that earlier phases'
files stay as they were built, so they can be read as that phase's shape. The
cost is two copies of ~40 lines. The tests for both (tests/test_pruning.py for
Phase 4, tests/test_multi_agent_persistence.py here) are what keep them from
drifting in behaviour.
"""

from collections.abc import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage, trim_messages
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.output_parsers import StrOutputParser

from research_copilot.memory import render_messages
from research_copilot.prompts import SUMMARY_PROMPT


def summary_messages(summary: str | None) -> list:
    if not summary:
        return []
    return [SystemMessage(content=f"Summary of earlier conversation:\n{summary}")]


def make_prune_history(
    *,
    strategy: str,
    budget: int,
    summarizer: Callable[[], BaseChatModel],
):
    """Build a prune node. `summarizer` is called lazily, only when the
    "summarize" strategy actually has something to fold away - so compiling the
    graph never needs a model."""
    strategy = strategy.lower()

    def prune_history(state) -> dict:
        messages = list(state.get("messages", []))
        summary = state.get("summary", "")

        if strategy == "none" or budget <= 0 or not messages:
            return {}
        if count_tokens_approximately(summary_messages(summary) + messages) <= budget:
            return {}

        kept = trim_messages(
            messages,
            max_tokens=budget,
            token_counter=count_tokens_approximately,
            strategy="last",
            start_on="human",
            include_system=False,
            allow_partial=False,
        )
        # A budget smaller than the current question: keep the tail from the
        # last human turn. Over budget, but coherent (Phase 4's rule).
        if not kept:
            starts = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]
            kept = messages[starts[-1]:] if starts else messages[-1:]

        kept_ids = {m.id for m in kept}
        dropped = [m for m in messages if m.id not in kept_ids]
        if not dropped:
            return {}

        update: dict = {"messages": [RemoveMessage(id=m.id) for m in dropped]}
        if strategy == "summarize":
            chain = SUMMARY_PROMPT | summarizer() | StrOutputParser()
            update["summary"] = chain.invoke(
                {
                    "previous_summary": summary or "(none)",
                    "conversation": render_messages(dropped),
                }
            ).strip()
        return update

    return prune_history
