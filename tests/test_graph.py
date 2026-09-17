"""Graph tests. Offline: fake model, fake retriever, stub tool.

These mirror tests/test_agent_loop.py deliberately. The same scenarios run
against the graph, so the two implementations can be compared directly.
"""

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool

from research_copilot.graph import build_graph, final_answer, run_graph


class FakeToolCallingModel(FakeMessagesListChatModel):
    """Replays scripted AIMessages. bind_tools is a no-op because the script
    already decides when tools get called."""

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


def fake_retriever(docs):
    return RunnableLambda(lambda _question: docs)


# --- live-search: the Phase 1 loop, as a graph --------------------------------


def test_graph_runs_tool_then_returns_final_answer():
    model = FakeToolCallingModel(
        responses=[
            tool_call("search_arxiv", {"query": "rag"}, "call_1"),
            AIMessage(content="Final answer citing [1]."),
        ]
    )
    state = run_graph(
        "What is RAG?",
        mode="live-search",
        graph=build_graph(model=model, tools=[search_arxiv]),
    )

    assert final_answer(state) == "Final answer citing [1]."
    # call_model ran twice: once to ask for the tool, once to answer.
    assert state["iterations"] == 2

    tool_messages = [m for m in state["messages"] if isinstance(m, ToolMessage)]
    assert tool_messages[0].tool_call_id == "call_1"
    assert "A paper about rag" in tool_messages[0].content

    # The transcript is the whole run: question, request, result, answer.
    assert [m.type for m in state["messages"]] == ["human", "ai", "tool", "ai"]


def test_add_messages_accumulates_rather_than_overwriting():
    """The reducer is the reason the transcript survives four nodes."""
    model = FakeToolCallingModel(
        responses=[
            tool_call("search_arxiv", {"query": "a"}, "c1"),
            tool_call("search_arxiv", {"query": "b"}, "c2"),
            AIMessage(content="done"),
        ]
    )
    state = run_graph(
        "Q", mode="live-search", graph=build_graph(model=model, tools=[search_arxiv])
    )

    # Without add_messages, each node's return value would replace the list and
    # this would be 1.
    assert len(state["messages"]) == 6
    assert state["messages"][0] == HumanMessage(content="Q", id=state["messages"][0].id)


def test_parallel_tool_calls_become_one_tool_message_each():
    model = FakeToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "search_arxiv", "args": {"query": "a"}, "id": "c1", "type": "tool_call"},
                    {"name": "search_arxiv", "args": {"query": "b"}, "id": "c2", "type": "tool_call"},
                ],
            ),
            AIMessage(content="done"),
        ]
    )
    state = run_graph(
        "Q", mode="live-search", graph=build_graph(model=model, tools=[search_arxiv])
    )
    tool_messages = [m for m in state["messages"] if isinstance(m, ToolMessage)]
    assert [m.tool_call_id for m in tool_messages] == ["c1", "c2"]


def test_unknown_tool_is_reported_to_model_not_raised():
    model = FakeToolCallingModel(
        responses=[tool_call("web_search", {"q": "x"}, "c1"), AIMessage(content="ok")]
    )
    state = run_graph(
        "Q", mode="live-search", graph=build_graph(model=model, tools=[search_arxiv])
    )
    error = next(m for m in state["messages"] if isinstance(m, ToolMessage))
    assert error.status == "error"
    assert "no tool named 'web_search'" in error.content
    assert final_answer(state) == "ok"


def test_max_iterations_stops_a_model_that_never_finishes():
    """The guard on the cycle. Without it this graph never reaches END.

    Each scripted response is a *distinct* AIMessage. FakeMessagesListChatModel
    cycles its list, so a single response would hand back the identical object
    every turn - and `add_messages` replaces a message whose `id` it has already
    seen instead of appending it, which quietly breaks the loop. A real model
    returns a fresh message each call; a fake has to be made to do the same.
    """
    model = FakeToolCallingModel(
        responses=[tool_call("search_arxiv", {"query": "rag"}, f"c{i}") for i in range(4)]
    )
    state = run_graph(
        "Q",
        mode="live-search",
        graph=build_graph(model=model, tools=[search_arxiv], max_iterations=3),
    )
    assert state["iterations"] == 3
    # It stopped cleanly and returned state, rather than raising
    # GraphRecursionError the way LangGraph's own backstop would.

    # Two tool results, not three - and this is a real difference from Phase 1.
    # `run_tool_loop` checks its budget with a `for` statement, so it executes
    # the tools *then* discovers it is out of iterations: 3 iterations, 3 tool
    # calls, and a transcript ending on a ToolMessage. The graph checks the
    # budget in should_continue, which sits between call_model and call_tool, so
    # the third model call's tool request is never run: 3 model calls, 2 tool
    # calls, and a transcript ending on an AIMessage. Ending on an AIMessage is
    # the better place to stop - there is something to show the user, and the
    # transcript is a valid conversation rather than one with an unanswered tool
    # result dangling at the end.
    assert len([m for m in state["messages"] if isinstance(m, ToolMessage)]) == 2
    assert isinstance(state["messages"][-1], AIMessage)


# --- knowledge-base: Phase 2's RAG, as a graph path ---------------------------


def test_knowledge_base_mode_retrieves_then_answers():
    docs = [Document(page_content="RAG retrieves then generates.", metadata={"source": "kb.md"})]
    model = FakeToolCallingModel(responses=[AIMessage(content="Grounded answer [1].")])

    state = run_graph(
        "What is RAG?",
        mode="knowledge-base",
        graph=build_graph(model=model, tools=[search_arxiv], retriever=fake_retriever(docs)),
    )

    assert final_answer(state) == "Grounded answer [1]."
    assert state["documents"] == docs
    assert "[1] (kb.md)" in state["context"]
    # One model call, no tool loop: the excerpts are the only source allowed.
    assert state["iterations"] == 1
    assert not [m for m in state["messages"] if isinstance(m, ToolMessage)]


def test_knowledge_base_mode_skips_the_tool_node_entirely():
    """should_continue ends a knowledge-base run even if the model asks for a tool.

    The edge from call_model to call_tool exists for every run; only the routing
    function keeps the knowledge-base path off it.
    """
    model = FakeToolCallingModel(
        responses=[tool_call("search_arxiv", {"query": "x"}, "c1")]
    )
    state = run_graph(
        "Q",
        mode="knowledge-base",
        graph=build_graph(model=model, tools=[search_arxiv], retriever=fake_retriever([])),
    )
    assert state["iterations"] == 1
    assert not [m for m in state["messages"] if isinstance(m, ToolMessage)]


def test_empty_retrieval_still_reaches_the_model():
    model = FakeToolCallingModel(responses=[AIMessage(content="The excerpts don't cover that.")])
    state = run_graph(
        "Q",
        mode="knowledge-base",
        graph=build_graph(model=model, tools=[search_arxiv], retriever=fake_retriever([])),
    )
    assert state["context"] == "(no excerpts retrieved)"
    assert final_answer(state) == "The excerpts don't cover that."


# --- routing ------------------------------------------------------------------


def test_unknown_mode_fails_loudly_at_the_router():
    model = FakeToolCallingModel(responses=[AIMessage(content="x")])
    graph = build_graph(model=model, tools=[search_arxiv])
    with pytest.raises(ValueError, match="unknown mode"):
        graph.invoke({"question": "Q", "mode": "knowledge base", "messages": []})


def test_call_tool_with_no_tool_calls_is_a_no_op():
    """Reached only by jumping straight to the node, as Studio lets you do."""
    model = FakeToolCallingModel(responses=[AIMessage(content="x")])
    graph = build_graph(model=model, tools=[search_arxiv])
    node = graph.nodes["call_tool"]
    assert node.invoke({"messages": [HumanMessage(content="no tool calls here")]}) == {}


# --- structure ----------------------------------------------------------------


def test_compiled_graph_has_the_expected_shape():
    """The structure is inspectable without running anything - this is what
    Studio draws."""
    model = FakeToolCallingModel(responses=[AIMessage(content="x")])
    drawn = build_graph(model=model, tools=[search_arxiv]).get_graph()

    assert {"retrieve_docs", "call_model", "call_tool"} <= set(drawn.nodes)

    edges = {(e.source, e.target) for e in drawn.edges}
    assert ("retrieve_docs", "call_model") in edges
    assert ("call_tool", "call_model") in edges  # the cycle
    assert ("call_model", "call_tool") in edges
    assert ("call_model", "__end__") in edges  # the way out of the cycle
