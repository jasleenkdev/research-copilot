"""Phase 6.2: the Supervisor, the hub, and the Researcher's new contract - offline.

The Supervisor here is scripted: it returns whatever decisions a test hands it.
That is enough to pin down everything that is *mechanism*:
  - what the Supervisor is shown, and in what form it must answer
  - that code guards overrule it, and that every override is logged
  - that an unusable reply falls back to 6.1's hand-off
  - that no sequence of decisions, however adversarial, can loop forever
  - that a second research pass is steered by a brief and merges, not replaces

It cannot tell you whether a real model routes *sensibly*: whether it spots a
missing source, whether its rationale matches its choice for the right reasons,
whether it re-researches too eagerly. Those need a real key.
"""

import itertools

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

from research_copilot import cli
from research_copilot.agents.researcher import FOLLOW_UP_HEADER
from research_copilot.agents.supervisor import (
    SupervisorDecision,
    apply_guards,
    draft_is_current,
    fixed_policy,
    render_supervisor_view,
)
from research_copilot.graph import final_answer
from research_copilot.multi_agent_graph import build_multi_agent_graph, run_multi_agent

CAPS = {"researcher": 2, "writer": 2}


# --- fakes ----------------------------------------------------------------------


class RecordingModel(FakeMessagesListChatModel):
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


class ScriptedSupervisor:
    """Stands in for a chat model's `with_structured_output`.

    Each script entry is one reply: a SupervisorDecision, a dict (validated by
    the node, as a provider's raw JSON would be), or an Exception to raise.
    Records the method it was asked for and every view it was shown.
    """

    def __init__(self, script):
        self.script = iter(script)
        self.views: list[str] = []
        self.methods: list[str] = []

    def with_structured_output(self, schema, *, method="function_calling", **kwargs):
        assert schema is SupervisorDecision
        self.methods.append(method)

        def reply(messages):
            self.views.append(messages[-1].content)
            item = next(self.script)
            if isinstance(item, Exception):
                raise item
            return item

        return RunnableLambda(reply)


def decide(next_, rationale="because", brief=""):
    return SupervisorDecision(rationale=rationale, next=next_, researcher_brief=brief)


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
    return f"[1] A paper about {query}"


def stub_retriever(by_query):
    return RunnableLambda(lambda q: by_query.get(q, []))


def hub(*, supervisor, researcher=None, writer=None, **kwargs):
    return build_multi_agent_graph(
        supervisor_model=supervisor,
        researcher_model=researcher,
        writer_model=writer,
        tools=[search_arxiv],
        **kwargs,
    )


def routes(state):
    return [e["routed_to"] for e in state["supervisor_log"]]


# --- structured output ------------------------------------------------------------


def test_supervisor_asks_for_native_json_schema_structured_output():
    """Not the default function_calling method: that forces a tool call, which
    the Anthropic API rejects when thinking is on."""
    supervisor = ScriptedSupervisor([decide("researcher"), decide("writer"), decide("finish")])
    run_multi_agent(
        "Q", graph=hub(supervisor=supervisor, researcher=scripted("notes"), writer=scripted("a"))
    )
    assert supervisor.methods == ["json_schema"]


def test_rationale_is_generated_before_the_route():
    """Field order is generation order under constrained decoding."""
    fields = list(SupervisorDecision.model_fields)
    assert fields.index("rationale") < fields.index("next")


def test_happy_path_is_researcher_writer_finish_with_rationales_logged():
    supervisor = ScriptedSupervisor(
        [
            decide("researcher", "nothing gathered yet"),
            decide("writer", "notes cover the question"),
            decide("finish", "draft answers it"),
        ]
    )
    state = run_multi_agent(
        "Q", graph=hub(supervisor=supervisor, researcher=scripted("notes"), writer=scripted("the answer"))
    )

    assert final_answer(state) == "the answer"
    assert routes(state) == ["researcher", "writer", "finish"]
    assert [e["rationale"] for e in state["supervisor_log"]] == [
        "nothing gathered yet", "notes cover the question", "draft answers it",
    ]
    assert all(e["override"] == "" for e in state["supervisor_log"])
    assert state["dispatches"] == {"researcher": 1, "writer": 1}


# --- what the Supervisor sees ---------------------------------------------------------


def test_view_shows_outcome_budget_staleness_and_its_own_history():
    state = {
        "question": "Q",
        "research_notes": "notes v2",
        "research_outcome": "findings",
        "draft": "draft from v1",
        "dispatches": {"researcher": 2, "writer": 1},
        "supervisor_log": [
            {"step": 1, "proposed": "researcher", "rationale": "r1", "routed_to": "researcher", "override": "", "brief": ""},
            {"step": 2, "proposed": "writer", "rationale": "r2", "routed_to": "writer", "override": "", "brief": ""},
            {"step": 3, "proposed": "researcher", "rationale": "r3", "routed_to": "researcher", "override": "", "brief": "benchmarks"},
        ],
    }
    view = render_supervisor_view(state, CAPS)

    assert "Latest research pass: findings" in view
    assert "researcher: 2 of 2 used" in view
    assert "writer: 1 of 2 used" in view
    assert "STALE" in view
    assert "(brief: benchmarks)" in view
    assert "notes v2" in view


def test_view_marks_truncation_rather_than_cutting_silently():
    view = render_supervisor_view({"question": "Q", "research_notes": "x" * 10_000}, CAPS)
    assert "truncated" in view


# --- guards --------------------------------------------------------------------------


def test_guard_capped_agent_is_never_dispatched():
    state = {"dispatches": {"researcher": 2, "writer": 0}}
    route, why = apply_guards("researcher", state, CAPS)
    assert route == "writer"
    assert "cap" in why


def test_guard_no_writing_before_research():
    route, why = apply_guards("writer", {"dispatches": {}}, CAPS)
    assert (route, why) == ("researcher", "writer proposed before any research")


def test_guard_no_finishing_without_a_draft():
    state = {"dispatches": {"researcher": 1}, "draft": ""}
    assert apply_guards("finish", state, CAPS) == ("writer", "finish proposed with no draft")


def test_guard_no_finishing_on_a_draft_older_than_the_notes():
    state = {
        "dispatches": {"researcher": 2, "writer": 1},
        "draft": "old draft",
        "research_outcome": "findings",
        "supervisor_log": [{"routed_to": "writer"}, {"routed_to": "researcher"}],
    }
    assert not draft_is_current(state)
    route, why = apply_guards("finish", state, CAPS)
    assert route == "writer"
    assert "predates" in why


def test_a_follow_up_pass_that_found_nothing_leaves_the_draft_current():
    state = {
        "draft": "draft",
        "research_outcome": "nothing_found",
        "supervisor_log": [{"routed_to": "writer"}, {"routed_to": "researcher"}],
    }
    assert draft_is_current(state)
    assert apply_guards("finish", state, CAPS) == ("finish", "")


def test_finish_is_allowed_when_the_writer_is_capped_even_if_stale():
    """The bound beats the preference: with no writer budget left, a stale
    draft is committed rather than looping."""
    state = {
        "dispatches": {"researcher": 2, "writer": 2},
        "draft": "old",
        "research_outcome": "findings",
        "supervisor_log": [{"routed_to": "researcher"}],
    }
    assert apply_guards("finish", state, CAPS) == ("finish", "")


def test_overrides_are_logged_beside_the_proposal_inside_a_run():
    supervisor = ScriptedSupervisor(
        [decide("writer", "just write it"), decide("writer"), decide("finish")]
    )
    state = run_multi_agent(
        "Q", graph=hub(supervisor=supervisor, researcher=scripted("notes"), writer=scripted("a"))
    )
    first = state["supervisor_log"][0]
    assert first["proposed"] == "writer"
    assert first["routed_to"] == "researcher"
    assert first["override"] == "writer proposed before any research"
    assert first["rationale"] == "just write it"


# --- unusable output -------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        ValueError("model returned prose"),
        {"rationale": "use the critic", "next": "critic"},  # not an agent (yet)
        {"next": "writer"},  # missing rationale
    ],
    ids=["raises", "unknown-agent", "missing-field"],
)
def test_unusable_supervisor_output_falls_back_to_the_fixed_policy(bad):
    supervisor = ScriptedSupervisor([bad, decide("writer"), decide("finish")])
    state = run_multi_agent(
        "Q", graph=hub(supervisor=supervisor, researcher=scripted("notes"), writer=scripted("a"))
    )
    first = state["supervisor_log"][0]
    assert first["proposed"] is None
    assert first["override"] == "fallback to fixed policy"
    assert "unusable" in first["rationale"]
    assert first["routed_to"] == "researcher"
    assert final_answer(state) == "a"


def test_a_supervisor_that_always_fails_degrades_to_exactly_6_1():
    supervisor = ScriptedSupervisor(itertools.repeat(RuntimeError("down")))
    state = run_multi_agent(
        "Q", graph=hub(supervisor=supervisor, researcher=scripted("notes"), writer=scripted("a"))
    )
    assert routes(state) == ["researcher", "writer", "finish"]
    assert final_answer(state) == "a"


# --- termination -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "script",
    [
        itertools.repeat(decide("researcher", brief="more")),
        itertools.repeat(decide("writer")),
        itertools.repeat(decide("finish")),
        itertools.cycle([decide("researcher", brief="b"), decide("writer")]),
    ],
    ids=["always-research", "always-write", "always-finish", "ping-pong"],
)
def test_no_supervisor_can_loop_forever(script):
    """The bound is sum(caps) + 1 decisions, whatever the model says. It is
    enforced by code, not by the model's good sense."""
    caps = {"researcher": 2, "writer": 2}
    state = run_multi_agent(
        "Q",
        graph=hub(
            supervisor=ScriptedSupervisor(script),
            researcher=scripted(*["notes"] * 5),
            writer=scripted(*["draft"] * 5),
            dispatch_caps=caps,
        ),
    )
    assert len(state["supervisor_log"]) <= sum(caps.values()) + 1
    assert routes(state)[-1] == "finish"
    assert all(state["dispatches"].get(a, 0) <= caps[a] for a in caps)
    assert [m.type for m in state["messages"]] == ["human", "ai"]


# --- the Researcher's new contract: brief, outcome, merge-on-rerun -------------------------


def test_second_research_pass_is_steered_by_the_brief_and_merges():
    researcher = scripted(
        "Findings: method X [paper A]",
        tool_call("X benchmarks", "c1"),
        "Findings: X scores 90 on B [paper B]",
    )
    writer = scripted("draft one", "draft two")
    supervisor = ScriptedSupervisor(
        [
            decide("researcher"),
            decide("writer"),
            decide("researcher", "draft lacks benchmarks", brief="benchmark results for X"),
            decide("writer"),
            decide("finish"),
        ]
    )
    state = run_multi_agent(
        "Q", graph=hub(supervisor=supervisor, researcher=researcher, writer=writer)
    )

    notes = state["research_notes"]
    # Merged, in order, with a code-written header naming the brief.
    assert notes.index("method X") < notes.index("scores 90")
    assert FOLLOW_UP_HEADER.format(reason="brief: benchmark results for X") in notes
    assert state["research_outcome"] == "findings"

    # The second pass was shown the brief and the notes it already had.
    second_pass_first_call = researcher.requests[1]
    system_text = "\n".join(m.text for m in second_pass_first_call if m.type == "system")
    assert "benchmark results for X" in system_text
    assert "method X" in system_text

    # The Writer's second draft was written from the merged notes.
    assert "scores 90" in writer.requests[1][-1].text
    assert "method X" in writer.requests[1][-1].text
    assert final_answer(state) == "draft two"


def test_a_follow_up_that_finds_nothing_new_keeps_the_notes_unchanged():
    researcher = scripted("Findings: A", "NOTHING NEW")
    supervisor = ScriptedSupervisor(
        [decide("researcher"), decide("writer"), decide("researcher", brief="more"), decide("finish")]
    )
    state = run_multi_agent(
        "Q", graph=hub(supervisor=supervisor, researcher=researcher, writer=scripted("d"))
    )
    assert state["research_notes"] == "Findings: A"
    assert state["research_outcome"] == "nothing_found"
    # Nothing changed, so the draft is still current and finish was accepted.
    assert state["supervisor_log"][-1]["override"] == ""


def test_knowledge_base_follow_up_appends_excerpts_and_keeps_their_numbers():
    first = Document(page_content="first chunk", metadata={"source": "a.md"})
    second = Document(page_content="second chunk", metadata={"source": "b.md"})
    retriever = stub_retriever({"Q": [first], "more on b": [first, second]})
    supervisor = ScriptedSupervisor(
        [decide("researcher"), decide("researcher", brief="more on b"), decide("writer"), decide("finish")]
    )
    state = run_multi_agent(
        "Q",
        mode="knowledge-base",
        graph=hub(supervisor=supervisor, writer=scripted("d"), retriever=retriever),
    )
    assert [d.page_content for d in state["documents"]] == ["first chunk", "second chunk"]
    assert state["research_notes"].index("[1]") < state["research_notes"].index("first chunk")
    assert "[2]" in state["research_notes"]
    assert state["research_outcome"] == "findings"


def test_knowledge_base_follow_up_with_no_new_excerpts_is_nothing_found():
    doc = Document(page_content="only chunk", metadata={"source": "a.md"})
    supervisor = ScriptedSupervisor(
        [decide("researcher"), decide("researcher", brief="again"), decide("writer"), decide("finish")]
    )
    state = run_multi_agent(
        "Q",
        mode="knowledge-base",
        graph=hub(supervisor=supervisor, writer=scripted("d"), retriever=stub_retriever({"Q": [doc], "again": [doc]})),
    )
    assert state["research_outcome"] == "nothing_found"
    assert len(state["documents"]) == 1


def test_budget_exhausted_is_an_outcome_not_just_a_header():
    researcher = scripted(tool_call("a", "c1"), tool_call("b", "c2"))
    supervisor = ScriptedSupervisor([decide("researcher"), decide("writer"), decide("finish")])
    state = run_multi_agent(
        "Q",
        graph=hub(supervisor=supervisor, researcher=researcher, writer=scripted("d"), max_research_iterations=2),
    )
    assert state["research_outcome"] == "budget_exhausted"
    assert "Latest research pass: budget_exhausted" in supervisor.views[1]


def test_a_brief_does_not_leak_into_a_later_dispatch_without_one():
    researcher = scripted("n1", "n2")
    supervisor = ScriptedSupervisor(
        [
            decide("researcher", brief="first focus"),
            decide("writer"),
            # A plain research dispatch with no brief. The first dispatch's
            # brief is still in state, and it must not steer this one.
            decide("researcher", brief=""),
            decide("writer"),
            decide("finish"),
        ]
    )
    state = run_multi_agent(
        "Q", graph=hub(supervisor=supervisor, researcher=researcher, writer=scripted("d1", "d2"))
    )
    second_pass = "\n".join(m.text for m in researcher.requests[1] if m.type == "system")
    assert "first focus" not in second_pass
    assert state["researcher_brief"] == ""


def test_an_overridden_route_carries_no_brief():
    """A guard that turns "writer" into "researcher" has no brief to pass on."""
    supervisor = ScriptedSupervisor(
        [SupervisorDecision(rationale="r", next="writer", researcher_brief="ignored"), decide("writer"), decide("finish")]
    )
    state = run_multi_agent(
        "Q", graph=hub(supervisor=supervisor, researcher=scripted("n"), writer=scripted("d"))
    )
    assert state["supervisor_log"][0]["brief"] == ""


# --- fixed routing and the turn boundary -------------------------------------------------


def test_fixed_policy_is_6_1s_hand_off():
    assert fixed_policy({}, CAPS) == "researcher"
    assert fixed_policy({"dispatches": {"researcher": 1}}, CAPS) == "writer"
    assert fixed_policy(
        {"dispatches": {"researcher": 1, "writer": 1}, "draft": "d", "supervisor_log": [{"routed_to": "writer"}]},
        CAPS,
    ) == "finish"


def test_routing_fixed_needs_no_supervisor_model(monkeypatch):
    def no_model(**kwargs):
        raise AssertionError("fixed routing must not build a supervisor model")

    monkeypatch.setattr("research_copilot.agents.supervisor.get_chat_model", no_model)
    state = run_multi_agent(
        "Q",
        graph=build_multi_agent_graph(
            researcher_model=scripted("n"), writer_model=scripted("d"), tools=[search_arxiv], routing="fixed"
        ),
    )
    assert routes(state) == ["researcher", "writer", "finish"]


def test_turn_boundary_resets_dispatches_and_the_log():
    supervisor = ScriptedSupervisor(
        [decide("researcher"), decide("writer"), decide("finish")] * 2
    )
    g = hub(
        supervisor=supervisor,
        researcher=scripted("n1", "n2"),
        writer=scripted("d1", "d2"),
        checkpointer=MemorySaver(),
    )
    config = {"configurable": {"thread_id": "t"}}
    run_multi_agent("one", graph=g, config=config)
    state = run_multi_agent("two", graph=g, config=config)

    assert len(state["supervisor_log"]) == 3
    assert state["dispatches"] == {"researcher": 1, "writer": 1}


def test_graph_draws_the_hub_without_an_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    drawn = build_multi_agent_graph().get_graph()
    edges = {(e.source, e.target) for e in drawn.edges}
    assert {
        ("supervisor", "researcher"),
        ("supervisor", "writer"),
        ("supervisor", "finalize_answer"),
        ("researcher", "supervisor"),
        ("writer", "supervisor"),
    } <= edges
    assert ("researcher", "writer") not in edges


# --- CLI ------------------------------------------------------------------------------------


def test_cli_prints_each_decision_with_its_rationale_and_overrides(monkeypatch, capsys):
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    shared = scripted("Findings: A", "The answer.")
    supervisor = ScriptedSupervisor(
        [decide("writer", "skip research"), decide("writer", "notes suffice"), decide("finish", "done")]
    )
    monkeypatch.setattr("research_copilot.agents.researcher.get_chat_model", lambda **k: shared)
    monkeypatch.setattr("research_copilot.agents.writer.get_chat_model", lambda **k: shared)
    monkeypatch.setattr("research_copilot.agents.supervisor.get_chat_model", lambda **k: supervisor)

    assert cli.main(["multi-agent", "Q"]) == 0
    out, err = capsys.readouterr()

    assert "The answer." in out
    assert "[routing] supervisor" in err
    assert "[supervisor] -> researcher  (proposed writer; OVERRIDDEN: writer proposed before any research)" in err
    assert "why: skip research" in err
    assert "decisions: 3 (1 overridden)" in err
