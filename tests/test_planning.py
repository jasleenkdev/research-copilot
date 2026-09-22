"""Phase 5 Part D: the planning node, offline.

The planner is a scripted model, so nothing here says anything about whether a
real model decomposes questions *well*. What these pin down is the mechanics
around it: that an undecomposed question is the normal answer rather than a
failure, that a bad plan cannot displace what Phase 4 would have retrieved, and
that the node is a no-op when the flag is off.
"""

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool

from research_copilot.graph import _parse_plan, build_graph, final_answer, run_graph


class FakeToolCallingModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


@tool
def search_arxiv(query: str) -> str:
    """Stub search."""
    return f"[1] A paper about {query}"


def scripted(*texts):
    return FakeToolCallingModel(responses=[AIMessage(content=t) for t in texts])


def recording_retriever(by_query, log):
    """A retriever that answers per query and records what it was asked."""

    def _retrieve(question):
        log.append(question)
        return by_query.get(question, [])

    return RunnableLambda(_retrieve)


def planned_graph(*, planner, model, retriever=None, **kwargs):
    return build_graph(
        model=model,
        planner_model=planner,
        retriever=retriever,
        tools=[search_arxiv],
        enable_planning=True,
        memory_strategy="none",
        **kwargs,
    )


# --- the node itself ----------------------------------------------------------


def test_planning_off_is_a_no_op_and_costs_no_model_call():
    """Same as prune_history under --memory none: the node stays, does nothing.

    The planner is scripted with a single response; if plan_question called it,
    the writer's own script would not be the one consumed.
    """
    graph = build_graph(
        model=scripted("the answer"),
        planner_model=scripted("- never asked"),
        tools=[search_arxiv],
        memory_strategy="none",
    )
    state = run_graph("Q", mode="live-search", graph=graph)

    assert final_answer(state) == "the answer"
    assert state.get("sub_questions", []) == []


def test_a_simple_question_comes_back_undecomposed():
    """NONE is a result, not a failure. A planner that always plans is a cost
    multiplier, so this is the case that has to work."""
    graph = planned_graph(planner=scripted("NONE"), model=scripted("answer"))
    state = run_graph("Who wrote the BERT paper?", mode="live-search", graph=graph)

    assert state["sub_questions"] == []
    assert final_answer(state) == "answer"


def test_a_complex_question_is_split_and_the_plan_lands_in_state():
    graph = planned_graph(
        planner=scripted("- How does RAG cost scale?\n- How does long context scale?"),
        model=scripted("answer"),
    )
    state = run_graph("Compare RAG and long context", mode="live-search", graph=graph)

    assert state["sub_questions"] == [
        "How does RAG cost scale?",
        "How does long context scale?",
    ]


def test_the_plan_reaches_the_writer_as_a_checklist():
    seen: list[list] = []

    class RecordingWriter(FakeToolCallingModel):
        def invoke(self, input, config=None, **kwargs):
            seen.append(list(input))
            return super().invoke(input, config, **kwargs)

    graph = planned_graph(
        planner=scripted("- part one\n- part two"),
        model=RecordingWriter(responses=[AIMessage(content="answer")]),
    )
    run_graph("Q", mode="live-search", graph=graph)

    request_text = "\n".join(m.text for m in seen[0])
    assert "- part one" in request_text
    assert "- part two" in request_text
    # A single connected answer, not one reply per sub-question. Fan-out is
    # Phase 6.
    assert "single connected answer" in request_text


# --- interaction with retrieval (knowledge-base mode) -------------------------


def test_retrieval_runs_once_per_sub_question_as_well_as_the_whole_question():
    asked: list[str] = []
    retriever = recording_retriever(
        {
            "Q": [Document(page_content="whole")],
            "sub a": [Document(page_content="a")],
            "sub b": [Document(page_content="b")],
        },
        asked,
    )
    graph = planned_graph(
        planner=scripted("- sub a\n- sub b"),
        model=scripted("grounded answer"),
        retriever=retriever,
    )
    state = run_graph("Q", mode="knowledge-base", graph=graph)

    assert asked == ["Q", "sub a", "sub b"]
    # The question's own chunks come first, so a bad plan can only ever add.
    assert [d.page_content for d in state["documents"]] == ["whole", "a", "b"]


def test_chunks_shared_between_sub_questions_are_not_listed_twice():
    """Sub-questions overlap by construction - they are parts of one question.

    A prompt listing the same excerpt as [2] and [5] invites the model to cite
    it twice as if it were two sources.
    """
    shared = Document(page_content="the same chunk")
    asked: list[str] = []
    retriever = recording_retriever(
        {"Q": [shared], "sub a": [shared], "sub b": [Document(page_content="other")]},
        asked,
    )
    graph = planned_graph(
        planner=scripted("- sub a\n- sub b"),
        model=scripted("answer"),
        retriever=retriever,
    )
    state = run_graph("Q", mode="knowledge-base", graph=graph)

    assert [d.page_content for d in state["documents"]] == ["the same chunk", "other"]
    assert state["context"].count("the same chunk") == 1


def test_an_empty_plan_retrieves_exactly_what_phase_4_retrieved():
    asked: list[str] = []
    retriever = recording_retriever({"Q": [Document(page_content="only")]}, asked)
    graph = planned_graph(
        planner=scripted("NONE"), model=scripted("answer"), retriever=retriever
    )
    state = run_graph("Q", mode="knowledge-base", graph=graph)

    assert asked == ["Q"]
    assert [d.page_content for d in state["documents"]] == ["only"]


# --- planning and the reflection loop together --------------------------------


def test_the_plan_survives_a_revision_and_is_not_remade():
    """`start_revision` resets the round's budget, not the turn's plan.

    The planner is scripted with one response; a second call would exhaust it.
    Re-planning per revision would also mean the critic's objection to draft 1
    could silently change what draft 2 is even trying to answer.
    """
    graph = planned_graph(
        planner=scripted("- part one\n- part two"),
        model=scripted("v1", "v2"),
        critic_model=scripted("REJECT\nmissed part two", "APPROVE"),
        enable_critic=True,
        max_revisions=2,
    )
    state = run_graph("Q", mode="live-search", graph=graph)

    assert state["sub_questions"] == ["part one", "part two"]
    assert state["revisions"] == 1
    assert final_answer(state) == "v2"


# --- parsing ------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("NONE", []),
        ("none", []),
        ("  NONE.  ", []),
        ("", []),
        ("- a\n- b", ["a", "b"]),
        ("* a\n* b", ["a", "b"]),
        ("1. a\n2. b", ["a", "b"]),
        # Preamble prose has no marker and is dropped rather than searched for.
        ("Here are the sub-questions:\n- a\n- b", ["a", "b"]),
        # Unparseable falls back to no plan: failing closed means doing less.
        ("I think this is quite a complex question honestly", []),
    ],
)
def test_plan_parsing(raw, expected):
    assert _parse_plan(raw, 4) == expected


def test_the_plan_is_capped():
    """A planner with no ceiling is a cost multiplier - each sub-question is
    another retrieval or another tool call."""
    raw = "\n".join(f"- q{i}" for i in range(10))
    assert _parse_plan(raw, 3) == ["q0", "q1", "q2"]
