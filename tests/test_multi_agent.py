"""Phase 6.1: Researcher and Writer with a fixed hand-off, entirely offline.

Every model here is scripted. That is enough to test what 6.1 actually adds:
  - ownership. Each agent writes only its own fields, and a trespass raises.
  - the private channel. The Researcher's tool traffic never reaches the shared
    transcript, the Writer, or the next turn.
  - the hand-off. What the Writer is shown, including when the Researcher
    produced nothing or ran out of budget.

What these tests cannot tell you is whether a real Researcher writes useful
notes, or whether a real Writer answers well from notes alone without seeing
the searches. Those are quality questions and need a real key.
"""

from typing import Annotated, TypedDict

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from research_copilot import cli
from research_copilot.agents.writer import NO_NOTES, WRITER_KB_RULES
from research_copilot.graph import final_answer
from research_copilot.multi_agent_graph import (
    build_multi_agent_graph,
    make_graph,
    multi_agent_turn_input,
    run_multi_agent,
)
from research_copilot.multi_agent_state import (
    OWNERS,
    OwnershipError,
    ResearcherOutput,
    owns,
)


class RecordingModel(FakeMessagesListChatModel):
    """A scripted model that also records every request it was sent.

    Recording is the point. Most of 6.1's guarantees are about what an agent
    is *shown*, so the tests need to see each agent's inputs, not only its
    outputs.
    """

    requests: list = []

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.requests.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def scripted(*replies):
    return RecordingModel(
        responses=[r if isinstance(r, AIMessage) else AIMessage(content=r) for r in replies]
    )


def tool_call(query, call_id):
    return AIMessage(
        content="",
        tool_calls=[
            {"name": "search_arxiv", "args": {"query": query}, "id": call_id, "type": "tool_call"}
        ],
    )


@tool
def search_arxiv(query: str) -> str:
    """Stub search."""
    return f"[1] A paper about {query}\n    URL: http://arxiv.org/abs/{query.replace(' ', '')}"


def stub_retriever(by_query, log=None):
    def _retrieve(question):
        if log is not None:
            log.append(question)
        return by_query.get(question, [])

    return RunnableLambda(_retrieve)


def graph(*, researcher=None, writer=None, **kwargs):
    return build_multi_agent_graph(
        researcher_model=researcher,
        writer_model=writer,
        tools=[search_arxiv],
        **kwargs,
    )


def text_of(messages):
    return "\n".join(m.text for m in messages)


# --- ownership ----------------------------------------------------------------


def test_researcher_output_schema_and_owners_table_agree():
    """The subgraph's structural contract and the runtime table must not drift.

    `owns()` checks plain nodes against OWNERS. The Researcher is held to the
    rule by its output_schema instead, so the two definitions have to be the
    same set, or one of them is lying about what the Researcher may write.
    """
    assert OWNERS["researcher"] == frozenset(ResearcherOutput.__annotations__)


def test_a_node_writing_a_field_it_does_not_own_raises():
    @owns("writer")
    def overreaching_writer(state):
        return {"draft": "the answer", "research_notes": "I tidied these up for you"}

    with pytest.raises(OwnershipError, match="research_notes"):
        overreaching_writer({})


def test_ownership_is_checked_inside_the_running_graph(monkeypatch):
    """Not just a decorator that works in isolation: the graph really uses it."""
    def trespassing_make_writer(*, model=None):
        return lambda state: {"draft": "x", "research_notes": "overwritten"}

    monkeypatch.setattr(
        "research_copilot.multi_agent_graph.make_writer", trespassing_make_writer
    )
    g = graph(researcher=scripted("notes"), writer=scripted("unused"))
    with pytest.raises(OwnershipError):
        run_multi_agent("Q", graph=g)


# --- the fixed hand-off, live-search -----------------------------------------


def test_researcher_then_writer_end_to_end():
    researcher = scripted(tool_call("rag", "c1"), "Findings: RAG helps [1]\nSources: ...")
    writer = scripted("RAG helps, per the literature.")
    state = run_multi_agent("Does RAG help?", graph=graph(researcher=researcher, writer=writer))

    assert final_answer(state) == "RAG helps, per the literature."
    assert state["research_notes"].startswith("Findings: RAG helps")
    assert state["draft"] == "RAG helps, per the literature."
    # Two Researcher model calls: one asked for a search, one wrote notes.
    assert state["research_iterations"] == 2


def test_tool_traffic_never_reaches_the_shared_transcript():
    """The private channel, from the outside: the transcript is one Q and one A."""
    researcher = scripted(tool_call("a", "c1"), tool_call("b", "c2"), "notes")
    state = run_multi_agent(
        "Q", graph=graph(researcher=researcher, writer=scripted("answer"))
    )

    assert [m.type for m in state["messages"]] == ["human", "ai"]
    assert not any(isinstance(m, ToolMessage) for m in state["messages"])
    assert not any(getattr(m, "tool_calls", None) for m in state["messages"])
    # And the private key itself never crosses the subgraph boundary.
    assert "research_messages" not in state


def test_writer_sees_notes_but_not_the_searches():
    """The private channel, from the Writer's side."""
    researcher = scripted(tool_call("secret query", "c1"), "Findings: X [1]")
    writer = scripted("answer")
    run_multi_agent("Q", graph=graph(researcher=researcher, writer=writer))

    (request,) = writer.requests
    assert "Findings: X [1]" in request[-1].text
    assert "Question: Q" in request[-1].text
    assert not any(isinstance(m, ToolMessage) for m in request)
    assert not any(getattr(m, "tool_calls", None) for m in request)
    assert "secretquery" not in text_of(request)


def test_researcher_sees_its_own_tool_loop():
    """...while the Researcher does see its own searches: private, not absent."""
    researcher = scripted(tool_call("rag", "c1"), "notes")
    run_multi_agent("Q", graph=graph(researcher=researcher, writer=scripted("a")))

    first, second = researcher.requests
    assert not any(isinstance(m, ToolMessage) for m in first)
    assert isinstance(second[-1], ToolMessage)
    assert "A paper about rag" in second[-1].text


# --- when an agent produces nothing ------------------------------------------


def test_empty_research_reaches_the_writer_as_an_explicit_no_evidence_note():
    """A blank notes section reads as "no constraints". The Writer is told
    plainly that nothing was found instead."""
    retriever = stub_retriever({})
    writer = scripted("No supporting evidence was found.")
    state = run_multi_agent(
        "Q",
        mode="knowledge-base",
        graph=graph(writer=writer, retriever=retriever),
    )

    assert state["research_notes"] == ""
    assert state["documents"] == []
    assert NO_NOTES in writer.requests[0][-1].text
    assert final_answer(state) == "No supporting evidence was found."


def test_budget_exhausted_mid_search_hands_over_raw_results_not_nothing():
    researcher = scripted(tool_call("a", "c1"), tool_call("b", "c2"), tool_call("c", "c3"))
    writer = scripted("answer")
    state = run_multi_agent(
        "Q",
        graph=graph(researcher=researcher, writer=writer, max_research_iterations=2),
    )

    assert state["research_iterations"] == 2
    notes = state["research_notes"]
    assert "budget ran out" in notes
    assert "A paper about a" in notes
    # The cap tripped with search "b" requested but never run. Only results
    # that actually came back are handed over.
    assert "A paper about b" not in notes
    assert "budget ran out" in writer.requests[0][-1].text


def test_budget_exhausted_before_any_result_is_empty_notes():
    researcher = scripted(tool_call("a", "c1"))
    writer = scripted("answer")
    state = run_multi_agent(
        "Q", graph=graph(researcher=researcher, writer=writer, max_research_iterations=1)
    )
    assert state["research_notes"] == ""
    assert NO_NOTES in writer.requests[0][-1].text


def test_an_empty_draft_still_commits_a_reply_to_keep_roles_alternating():
    state = run_multi_agent(
        "Q", graph=graph(researcher=scripted("notes"), writer=scripted(""))
    )
    assert [m.type for m in state["messages"]] == ["human", "ai"]
    assert "no answer" in final_answer(state)


# --- knowledge-base mode ------------------------------------------------------


def test_knowledge_base_research_is_retrieval_with_no_model_call(monkeypatch):
    def no_model(**kwargs):
        raise AssertionError("the Researcher must not call a model in knowledge-base mode")

    monkeypatch.setattr("research_copilot.agents.researcher.get_chat_model", no_model)
    docs = [Document(page_content="Chunk about RAG.", metadata={"source": "notes.md"})]
    writer = scripted("From your docs: RAG [1].")
    state = run_multi_agent(
        "What is RAG?",
        mode="knowledge-base",
        graph=graph(writer=writer, retriever=stub_retriever({"What is RAG?": docs})),
    )

    assert state["research_iterations"] == 0
    assert state["research_notes"].startswith("[1]")
    assert "Chunk about RAG." in state["research_notes"]
    assert WRITER_KB_RULES.strip() in writer.requests[0][0].text


def test_plan_reaches_both_agents():
    log = []
    docs = {
        "Q": [Document(page_content="whole", metadata={"source": "a"})],
        "part one": [Document(page_content="one", metadata={"source": "b"})],
    }
    writer = scripted("answer")
    state = run_multi_agent(
        "Q",
        mode="knowledge-base",
        graph=graph(
            writer=writer,
            planner_model=scripted("- part one\n- part two"),
            retriever=stub_retriever(docs, log),
            enable_planning=True,
        ),
    )

    assert state["sub_questions"] == ["part one", "part two"]
    assert log == ["Q", "part one", "part two"]
    assert "- part one" in text_of(writer.requests[0])


# --- the subgraph boundary ----------------------------------------------------


def test_research_messages_start_empty_every_invocation():
    """Two turns on one checkpointed thread. Turn 2's Researcher sees turn 1's
    *conversation* (question and answer) but none of turn 1's searches."""
    researcher = scripted(
        tool_call("first", "c1"), "notes one",
        tool_call("second", "c2"), "notes two",
    )
    writer = scripted("answer one", "answer two")
    g = graph(researcher=researcher, writer=writer, checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "t"}}

    run_multi_agent("first question", graph=g, config=config)
    state = run_multi_agent("second question", graph=g, config=config)

    turn_two_first_request = researcher.requests[2]
    assert "answer one" in text_of(turn_two_first_request)
    assert not any(isinstance(m, ToolMessage) for m in turn_two_first_request)
    assert "A paper about first" not in text_of(turn_two_first_request)

    assert [m.type for m in state["messages"]] == ["human", "ai", "human", "ai"]
    assert "research_messages" not in g.get_state(config).values


def test_private_channel_is_hidden_from_state_not_from_disk():
    """The limit of "private". The parent's state never holds the tool loop,
    but the Researcher subgraph checkpoints under its own namespace, and those
    checkpoints do hold it. Phase 4's deleted-from-state-is-not-deleted-from-
    disk, one level down."""

    @tool
    def search_arxiv(query: str) -> str:
        """Stub search with a recognisable result."""
        return "RAW-RESULT-7f3a"

    saver = MemorySaver()
    g = build_multi_agent_graph(
        researcher_model=scripted(tool_call("x", "c1"), "notes"),
        writer_model=scripted("answer"),
        tools=[search_arxiv],
        checkpointer=saver,
    )
    config = {"configurable": {"thread_id": "t"}}
    run_multi_agent("Q", graph=g, config=config)

    assert "RAW-RESULT-7f3a" not in repr(g.get_state(config).values)

    by_namespace = {}
    for checkpoint in saver.list(None):
        ns = checkpoint.config["configurable"].get("checkpoint_ns", "")
        by_namespace.setdefault(ns, []).append(checkpoint.checkpoint["channel_values"])
    researcher_ns = [ns for ns in by_namespace if ns.startswith("researcher:")]
    assert researcher_ns
    assert any(
        "RAW-RESULT-7f3a" in repr(values.get("research_messages", []))
        for ns in researcher_ns
        for values in by_namespace[ns]
    )
    assert not any("research_messages" in values for values in by_namespace[""])


def test_turn_boundary_clears_last_turns_notes():
    """Ownership's lifecycle: turn 2 must not inherit turn 1's findings."""
    docs = {"one": [Document(page_content="turn one evidence", metadata={"source": "a"})]}
    writer = scripted("a1", "a2")
    g = graph(writer=writer, retriever=stub_retriever(docs), checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "t"}}

    run_multi_agent("one", mode="knowledge-base", graph=g, config=config)
    state = run_multi_agent("two", mode="knowledge-base", graph=g, config=config)

    assert state["research_notes"] == ""
    assert "turn one evidence" not in writer.requests[1][-1].text


def test_output_schema_is_what_stops_a_same_named_channel_leaking():
    """Why the private key has its own name, *and* the subgraph has an
    output_schema. Rename `research_messages` to `messages` and it becomes the
    parent's channel. Then only the output_schema stands between the tool loop
    and the shared transcript. Remove that too, and it leaks.
    """

    class Parent(TypedDict, total=False):
        messages: Annotated[list[BaseMessage], add_messages]

    class ChildOut(TypedDict, total=False):
        research_notes: str

    class Child(Parent, ChildOut, total=False):
        pass

    def child_node(state):
        return {"messages": [AIMessage(content="scratch work")], "research_notes": "n"}

    def parent_with(child):
        class P(Parent, ChildOut, total=False):
            pass

        builder = StateGraph(P)
        builder.add_node("researcher", child)
        builder.add_edge(START, "researcher")
        builder.add_edge("researcher", END)
        return builder.compile()

    def child_graph(**schemas):
        builder = StateGraph(Child, **schemas)
        builder.add_node("work", child_node)
        builder.add_edge(START, "work")
        builder.add_edge("work", END)
        return builder.compile()

    guarded = parent_with(child_graph(output_schema=ChildOut)).invoke(
        {"messages": [HumanMessage(content="Q")]}
    )
    unguarded = parent_with(child_graph()).invoke({"messages": [HumanMessage(content="Q")]})

    assert [m.text for m in guarded["messages"]] == ["Q"]
    assert [m.text for m in unguarded["messages"]] == ["Q", "scratch work"]


def test_stream_shows_the_researchers_inner_steps_under_its_namespace():
    """What Studio and `multi-agent`'s trace show: inner steps are visible
    while running, even though they are absent from the final state."""
    g = graph(researcher=scripted(tool_call("x", "c1"), "notes"), writer=scripted("a"))
    events = list(
        g.stream(multi_agent_turn_input("Q"), stream_mode="updates", subgraphs=True)
    )

    outer = [node for ns, update in events if not ns for node in update]
    inner = [node for ns, update in events if ns for node in update]
    assert outer == ["plan_question", "researcher", "writer", "finalize_answer"]
    assert inner == ["research_model", "research_tools", "research_model", "compile_notes"]
    assert all(ns[0].startswith("researcher:") for ns, _ in events if ns)


def test_nested_updates_stream_hides_private_writes_but_tasks_stream_shows_them():
    """Found while building the CLI trace. For a nested subgraph, "updates" is
    narrowed to its output schema, so a step that wrote only the private
    channel looks like it did nothing. "tasks" shows the full write. If a
    LangGraph upgrade changes either behaviour, this is where to find out."""
    def run(mode):
        g = graph(researcher=scripted(tool_call("x", "c1"), "notes"), writer=scripted("a"))
        return [
            event
            for ns, event in g.stream(multi_agent_turn_input("Q"), stream_mode=mode, subgraphs=True)
            if ns
        ]

    updates = {node: set(update or {}) for event in run("updates") for node, update in event.items()}
    tasks = [e for e in run("tasks") if "result" in e]

    assert updates["research_tools"] == set()
    assert any(e["name"] == "research_tools" and "research_messages" in e["result"] for e in tasks)


def test_graph_compiles_and_draws_without_an_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    g = make_graph()

    collapsed = set(g.get_graph().nodes)
    expanded = set(g.get_graph(xray=True).nodes)
    assert "researcher" in collapsed and "researcher:research_model" not in collapsed
    assert {"researcher:research_model", "researcher:research_tools"} <= expanded
    # The Writer is a node, so there is nothing inside it to expand.
    assert not any(n.startswith("writer:") for n in expanded)


# --- CLI ----------------------------------------------------------------------


def test_cli_multi_agent_prints_answer_and_hand_offs(monkeypatch, capsys):
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    # One scripted model serving every agent, in call order - the CLI has no
    # per-agent model flags, same as Phase 5's CLI tests.
    shared = scripted(tool_call("rag", "c1"), "Findings: RAG [1]", "Final: RAG works.")
    for module in ("agents.researcher", "agents.writer", "multi_agent_graph"):
        monkeypatch.setattr(f"research_copilot.{module}.get_chat_model", lambda **kw: shared)
    monkeypatch.setattr("research_copilot.agents.researcher.search_arxiv", search_arxiv)

    assert cli.main(["multi-agent", "Does RAG work?"]) == 0
    out, err = capsys.readouterr()

    assert "Final: RAG works." in out
    assert "[researcher/research_model]" in err
    # The private write is visible in the trace ("tasks" stream mode), even
    # though it never reaches the final state.
    assert "[researcher/research_tools] research_messages=[1]" in err
    assert "[writer]" in err
    assert "leaked" not in err
