"""Phase 7 Part B: state that survives two agents running at the same time.

The bar for a concurrency fix: show the OLD pattern losing an update in a real
parallel graph - not just assert the new one is safe - then show the new one
keeping both. Offline, with scripted models.

What these do not claim: that the multi-agent graph *does* fan out. It still
runs one agent at a time. Part B makes the state safe for the day it doesn't.
"""

from typing import Annotated, TypedDict

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph

from research_copilot.agents.critic import build_critic
from research_copilot.agents.researcher import build_researcher
from research_copilot.multi_agent_state import (
    AGENTS,
    BUDGET_FIELDS,
    OWNERS,
    MultiAgentState,
    merge_budgets,
)


class Fake(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


@tool("search_arxiv")
def search(query: str, max_results: int = 5) -> str:
    """Stub."""
    return "[1] A paper"


@tool("verify_citation")
def verify(arxiv_id: str) -> str:
    """Stub."""
    return f"FOUND: {arxiv_id} - a paper"


def parallel(*branches):
    """START fans out to every branch in the same super-step, then END."""

    def build(state_type):
        b = StateGraph(state_type)
        for name, node in branches:
            b.add_node(name, node)
            b.add_edge(START, name)
            b.add_edge(name, END)
        return b.compile()

    return build


# --- the old pattern, reproduced -----------------------------------------------------


class OldState(TypedDict, total=False):
    budgets: Annotated[dict, merge_budgets]


def old_style_agent(agent: str):
    """What the 6.3 subgraphs effectively did: read the whole shared dict,
    update their own entry, return the WHOLE dict (the other entries as
    passthrough) - because a subgraph's output schema works per key."""

    class Io(TypedDict, total=False):
        budgets: Annotated[dict, merge_budgets]

    def spend(state):
        whole = {k: dict(v) for k, v in state["budgets"].items()}
        whole[agent]["used"] += 1
        return {"budgets": whole}

    b = StateGraph(Io)
    b.add_node("spend", spend)
    b.add_edge(START, "spend")
    b.add_edge("spend", END)
    return b.compile()


def test_the_old_shared_dict_loses_an_update_under_fan_out():
    """Pinned as the *reason* for Part B. Both agents spend one call; one of
    the two spends vanishes, and nothing raises."""
    g = parallel(("researcher", old_style_agent("researcher")), ("critic", old_style_agent("critic")))(OldState)
    out = g.invoke({"budgets": {"researcher": {"used": 2}, "critic": {"used": 1}}})
    spent = (out["budgets"]["researcher"]["used"], out["budgets"]["critic"]["used"])
    assert spent != (3, 2)             # an update was lost...
    assert spent in {(3, 1), (2, 2)}   # ...exactly one of the two, silently


# --- the real subgraphs, now ---------------------------------------------------------


def test_the_real_researcher_and_critic_in_parallel_keep_both_budgets():
    """Before Part B, this exact graph returned the Critic's spend as 1 (the
    Researcher's stale passthrough copy won). Demonstrated before the change;
    see multi_agent_state.py, 'why budgets became three fields'."""
    g = parallel(
        ("researcher", build_researcher(model=Fake(responses=[AIMessage(content="notes")]), tools=[search])),
        ("critic", build_critic(model=Fake(responses=[AIMessage(content="APPROVE")]), tools=[verify])),
    )(MultiAgentState)
    out = g.invoke({
        "question": "Q", "mode": "live-search", "messages": [HumanMessage(content="Q")],
        "draft": "RAGAS [2309.15217]",
        "researcher_budget": {"used": 2, "cap": 6}, "critic_budget": {"used": 1, "cap": 4},
    })
    assert out["researcher_budget"]["used"] == 3
    assert out["critic_budget"]["used"] == 2
    # And each agent's real output survived too.
    assert out["research_notes"] == "notes" and out["verdict"] == "approve"


# --- the audit: which fields have more than one writer? -------------------------------

# Every field written by more than one node, and why that is safe. Anything
# else with two writers fails the audit - including a future edit that adds
# one without writing down why.
DOCUMENTED_SHARED = {
    # The turn boundary resets everything per-turn. It runs alone, at START.
    **{field: "begin_turn" for field in OWNERS["begin_turn"]},
    # The per-round reset. Runs alone, between rounds.
    **{field: "start_revision" for field in BUDGET_FIELDS.values()},
    "revisions": "start_revision",
    # Appended by finalize_answer at the end of a turn, pruned by
    # prune_history at the start; add_messages is id-based. Never concurrent.
    "messages": "prune_history + finalize_answer",
}
LIFECYCLE = {"begin_turn", "start_revision", "prune_history"}


def test_every_field_has_one_agent_writer_or_a_documented_reason():
    writers: dict[str, set[str]] = {}
    for node, fields in OWNERS.items():
        for field in fields:
            writers.setdefault(field, set()).add(node)
    undocumented = {
        field: nodes for field, nodes in writers.items()
        if len(nodes - LIFECYCLE) > 1 or (len(nodes) > 1 and field not in DOCUMENTED_SHARED)
    }
    assert undocumented == {}


def test_no_two_agents_share_a_field():
    """The fan-out property itself: any two agents could run in parallel
    without writing the same key."""
    agents = ["researcher", "writer", "critic", "supervisor"]
    for i, a in enumerate(agents):
        for b in agents[i + 1:]:
            assert not (OWNERS[a] & OWNERS[b]), (a, b, OWNERS[a] & OWNERS[b])


def test_each_agent_owns_exactly_its_own_budget_field():
    for agent in AGENTS:
        budget_fields = {f for f in OWNERS[agent] if f.endswith("_budget") or f == "budgets"}
        assert budget_fields == {BUDGET_FIELDS[agent]}
