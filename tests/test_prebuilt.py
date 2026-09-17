"""The prebuilt agent, tested the same way as the hand-rolled graph.

The point of these is the comparison: the same scripted model produces the same
transcript through `create_react_agent` as through `build_graph`, which is the
evidence that Part B really did rebuild what the prebuilt does for free.
"""

from langchain_core.messages import AIMessage, ToolMessage

from research_copilot.graph import final_answer
from research_copilot.prebuilt import build_prebuilt_agent, run_prebuilt_agent
from tests.test_graph import FakeToolCallingModel, search_arxiv, tool_call


def test_prebuilt_agent_runs_the_same_loop():
    model = FakeToolCallingModel(
        responses=[
            tool_call("search_arxiv", {"query": "rag"}, "call_1"),
            AIMessage(content="Final answer citing [1]."),
        ]
    )
    state = run_prebuilt_agent(
        "What is RAG?",
        agent=build_prebuilt_agent(model=model, tools=[search_arxiv]),
    )

    assert final_answer(state) == "Final answer citing [1]."
    # The identical transcript shape the hand-rolled graph produces.
    assert [m.type for m in state["messages"]] == ["human", "ai", "tool", "ai"]

    tool_messages = [m for m in state["messages"] if isinstance(m, ToolMessage)]
    assert tool_messages[0].tool_call_id == "call_1"


def test_prebuilt_state_carries_only_the_transcript():
    """The reason Phase 4 onwards can't use it: there is nowhere to put anything.

    Our State has `question`, `mode`, `documents`, `context`, and `iterations`
    alongside `messages`. The prebuilt's default state has `messages` (and its
    own `remaining_steps` budget), so a value like `mode` has no home in it.
    """
    model = FakeToolCallingModel(responses=[AIMessage(content="done")])
    state = run_prebuilt_agent(
        "Q", agent=build_prebuilt_agent(model=model, tools=[search_arxiv])
    )
    assert "mode" not in state
    assert "documents" not in state
    assert set(state) <= {"messages", "remaining_steps", "structured_response"}


def test_prebuilt_graph_is_two_nodes_and_one_branch():
    model = FakeToolCallingModel(responses=[AIMessage(content="done")])
    drawn = build_prebuilt_agent(model=model, tools=[search_arxiv]).get_graph()

    # Compare with test_graph.py's structure test: three nodes and two branches
    # there, because of retrieve_docs and the entry router.
    assert {"agent", "tools"} <= set(drawn.nodes)
    assert "retrieve_docs" not in drawn.nodes

    edges = {(e.source, e.target) for e in drawn.edges}
    assert ("tools", "agent") in edges  # the same cycle
    assert ("agent", "tools") in edges
