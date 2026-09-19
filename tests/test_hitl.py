"""Phase 4 Part C: interrupt / resume, entirely offline.

No real answer is ever generated here. The fake model returns a fixed string,
and every assertion is about the *mechanics* - where the run stops, what is and
is not in the transcript while it is stopped, and what `Command(resume=...)`
does when it starts again.
"""

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from research_copilot.checkpointing import checkpointer_scope, thread_config
from research_copilot.graph import (
    _parse_verdict,
    build_graph,
    final_answer,
    pending_interrupt,
    resume_graph,
    run_graph,
)


class FakeToolCallingModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


@tool
def search_arxiv(query: str) -> str:
    """Stub search."""
    return f"[1] A paper about {query}"


def tool_call(name, args, call_id):
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


def hitl_graph(saver, *responses, **kwargs):
    return build_graph(
        model=FakeToolCallingModel(responses=list(responses)),
        tools=[search_arxiv],
        checkpointer=saver,
        require_approval=True,
        memory_strategy="none",
        **kwargs,
    )


# --- the pause ----------------------------------------------------------------


def test_the_graph_stops_at_the_interrupt_and_reports_the_draft():
    with checkpointer_scope("memory") as saver:
        graph = hitl_graph(saver, AIMessage(content="a draft answer"))
        state = run_graph("Q", graph=graph, thread_id="t")

        # .invoke() returned, but the run is not finished: __interrupt__ is the
        # signal, and its payload is what the human is being asked.
        assert "__interrupt__" in state
        payload = state["__interrupt__"][0].value
        assert payload["draft"] == "a draft answer"
        assert payload["question"] == "Q"

        assert state["status"] == "awaiting_approval"
        assert state["draft"] == "a draft answer"


def test_the_draft_is_not_in_the_transcript_while_it_is_pending():
    """The reason `draft` has to be its own key.

    Nothing has been said to the user yet, so `messages` must not claim it has.
    """
    with checkpointer_scope("memory") as saver:
        graph = hitl_graph(saver, AIMessage(content="unapproved"))
        state = run_graph("Q", graph=graph, thread_id="t")

        assert [m.type for m in state["messages"]] == ["human"]
        assert "unapproved" not in [m.text for m in state["messages"]]


def test_the_paused_run_is_readable_from_the_checkpoint_alone():
    """What a *different process* would see: no in-memory handle, just a thread_id."""
    with checkpointer_scope("memory") as saver:
        graph = hitl_graph(saver, AIMessage(content="draft"))
        run_graph("Q", graph=graph, thread_id="t")

        # A freshly compiled graph sharing only the saver - the CLI's `review`
        # command does exactly this.
        reader = hitl_graph(saver, AIMessage(content="unused"))
        snapshot = reader.get_state(thread_config("t"))

        assert snapshot.next == ("review_draft",)
        assert snapshot.interrupts
        assert pending_interrupt(reader, thread_id="t")["draft"] == "draft"


# --- the resume ---------------------------------------------------------------


def test_approve_commits_the_draft_to_the_transcript():
    with checkpointer_scope("memory") as saver:
        graph = hitl_graph(saver, AIMessage(content="the answer"))
        run_graph("Q", graph=graph, thread_id="t")

        state = resume_graph(graph, {"decision": "approve"}, thread_id="t")

        assert state["status"] == "approved"
        assert final_answer(state) == "the answer"
        assert [m.type for m in state["messages"]] == ["human", "ai"]
        # The draft is cleared, so the next turn's state dump shows nothing pending.
        assert state["draft"] == ""
        assert pending_interrupt(graph, thread_id="t") is None


def test_edit_puts_the_humans_words_in_the_transcript_not_the_models():
    with checkpointer_scope("memory") as saver:
        graph = hitl_graph(saver, AIMessage(content="model wording"))
        run_graph("Q", graph=graph, thread_id="t")

        state = resume_graph(
            graph,
            {"decision": "edit", "text": "human wording", "note": "clearer"},
            thread_id="t",
        )

        assert state["status"] == "approved"
        assert final_answer(state) == "human wording"
        assert "model wording" not in [m.text for m in state["messages"]]
        assert state["human_feedback"] == "clearer"


def test_reject_withholds_the_answer_and_records_why():
    with checkpointer_scope("memory") as saver:
        graph = hitl_graph(saver, AIMessage(content="too speculative"))
        run_graph("Q", graph=graph, thread_id="t")

        state = resume_graph(
            graph, {"decision": "reject", "note": "no sources"}, thread_id="t"
        )

        assert state["status"] == "rejected"
        assert state["human_feedback"] == "no sources"
        assert "too speculative" not in [m.text for m in state["messages"]]
        # The turn is still answered - a rejection is a response, and leaving the
        # HumanMessage unanswered would put two human turns back to back.
        assert [m.type for m in state["messages"]] == ["human", "ai"]
        assert "withheld" in final_answer(state)


def test_a_bare_string_verdict_works_too():
    with checkpointer_scope("memory") as saver:
        graph = hitl_graph(saver, AIMessage(content="fine"))
        run_graph("Q", graph=graph, thread_id="t")
        state = resume_graph(graph, "approve", thread_id="t")
        assert final_answer(state) == "fine"


# --- the resume path, when it is used wrongly --------------------------------


def test_resuming_a_thread_with_nothing_parked_is_a_silent_no_op():
    """Not an error - which is the thing to know.

    LangGraph finds no interrupted task, runs nothing, and hands back the state
    as it already was. A caller that does not check `pending_interrupt` first
    cannot tell "approved" from "there was nothing to approve".
    """
    with checkpointer_scope("memory") as saver:
        graph = hitl_graph(saver, AIMessage(content="answer"))
        run_graph("Q", graph=graph, thread_id="t")
        resume_graph(graph, "approve", thread_id="t")
        before = graph.get_state(thread_config("t")).values

        again = resume_graph(graph, "approve", thread_id="t")

        assert again["messages"] == before["messages"]
        assert pending_interrupt(graph, thread_id="t") is None


def test_resuming_an_unknown_thread_does_nothing_rather_than_raising():
    with checkpointer_scope("memory") as saver:
        graph = hitl_graph(saver, AIMessage(content="answer"))
        state = resume_graph(graph, "approve", thread_id="no-such-thread")
        assert state in ({}, None) or not state.get("messages")


def test_resume_without_a_thread_id_is_refused():
    """The one case the project raises on, because it cannot possibly be right."""
    with checkpointer_scope("memory") as saver:
        graph = hitl_graph(saver, AIMessage(content="answer"))
        with pytest.raises(RuntimeError, match="needs a thread_id"):
            resume_graph(graph, "approve")


def test_approval_without_a_checkpointer_is_refused_at_build_time():
    """interrupt() has nowhere to park a run, so the combination never compiles."""
    with pytest.raises(RuntimeError, match="needs a checkpointer"):
        build_graph(
            model=FakeToolCallingModel(responses=[AIMessage(content="x")]),
            require_approval=True,
        )


# --- verdict parsing ----------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        ("approve", "approve"),
        ("  APPROVE  ", "approve"),
        ({"decision": "edit", "text": "t"}, "edit"),
        ({"decision": "reject"}, "reject"),
        # Everything unrecognized fails closed.
        ({"decision": "maybe"}, "reject"),
        ("yes please", "reject"),
        (None, "reject"),
        (42, "reject"),
        ({}, "reject"),
    ],
)
def test_unrecognized_verdicts_fail_closed(value, expected):
    """An approval gate must never read confusion as consent."""
    assert _parse_verdict(value)[0] == expected


def test_a_garbled_verdict_reaches_the_graph_as_a_rejection():
    with checkpointer_scope("memory") as saver:
        graph = hitl_graph(saver, AIMessage(content="draft"))
        run_graph("Q", graph=graph, thread_id="t")

        state = resume_graph(graph, {"decision": "loks good"}, thread_id="t")

        assert state["status"] == "rejected"
        assert "draft" not in [m.text for m in state["messages"]]


# --- interaction with the rest of the graph ----------------------------------


def test_the_tool_loop_runs_to_completion_before_review_is_asked_for():
    """Only the final answer is reviewed; intermediate tool calls are not.

    `status` is also proof the loop did not misroute: call_model sets it back to
    "drafting" whenever it emits a tool call, so the review branch fires exactly
    once, at the end.
    """
    with checkpointer_scope("memory") as saver:
        graph = hitl_graph(
            saver,
            tool_call("search_arxiv", {"query": "rag"}, "c1"),
            AIMessage(content="final, citing [1]"),
        )
        state = run_graph("Q", graph=graph, thread_id="t")

        assert state["iterations"] == 2
        assert [m.type for m in state["messages"]] == ["human", "ai", "tool"]
        assert state["status"] == "awaiting_approval"

        done = resume_graph(graph, "approve", thread_id="t")
        assert [m.type for m in done["messages"]] == ["human", "ai", "tool", "ai"]
        assert final_answer(done) == "final, citing [1]"
        assert [m for m in done["messages"] if isinstance(m, ToolMessage)]


def test_approval_survives_across_separate_graph_objects_and_connections(tmp_path):
    """The whole Part C cycle the way the CLI runs it: pause in one process,
    resume in another.

    Each `with` block is a new connection and a new compiled graph. The only
    thing shared is the SQLite file and the thread_id string.
    """
    db = tmp_path / "cp.sqlite3"

    with checkpointer_scope("sqlite", db_path=db) as saver:
        graph = hitl_graph(saver, AIMessage(content="drafted in process one"))
        state = run_graph("Q", graph=graph, thread_id="t")
        assert "__interrupt__" in state

    with checkpointer_scope("sqlite", db_path=db) as saver:
        graph = hitl_graph(saver, AIMessage(content="never called"))
        assert pending_interrupt(graph, thread_id="t")["draft"] == "drafted in process one"
        state = resume_graph(graph, "approve", thread_id="t")

    assert final_answer(state) == "drafted in process one"


def test_a_pending_review_does_not_leak_into_another_thread():
    with checkpointer_scope("memory") as saver:
        graph = hitl_graph(saver, AIMessage(content="d1"), AIMessage(content="d2"))
        run_graph("Q", graph=graph, thread_id="alice")

        assert pending_interrupt(graph, thread_id="alice") is not None
        assert pending_interrupt(graph, thread_id="bob") is None


def test_a_reviewed_turn_is_ordinary_history_for_the_next_turn():
    """After approval the thread is idle again and the next turn just continues."""
    with checkpointer_scope("memory") as saver:
        graph = hitl_graph(saver, AIMessage(content="A1"), AIMessage(content="A2"))

        run_graph("Q1", graph=graph, thread_id="t")
        resume_graph(graph, "approve", thread_id="t")

        run_graph("Q2", graph=graph, thread_id="t")
        state = resume_graph(graph, "approve", thread_id="t")

        assert [m.text for m in state["messages"]] == ["Q1", "A1", "Q2", "A2"]
        assert state["iterations"] == 1
