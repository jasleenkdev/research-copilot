"""Phase 6.3: rejections through the Supervisor, staleness, the human gate, and
per-agent budgets - offline.

Scripted Supervisor, scripted agents. These pin the mechanics: that a
rejection reaches the Supervisor and not a fixed node, that the cap still
overrules the verdict, that stale critiques and stale approvals are recognised
from dispatch order, that budgets are per round and per agent. Whether a real
Supervisor classifies a critique *correctly* is not something a fake can say.
"""

import itertools

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from research_copilot import cli
from research_copilot.agents.supervisor import (
    SupervisorDecision,
    apply_guards,
    critique_is_current,
    default_dispatch_caps,
    draft_was_rejected,
    render_supervisor_view,
)
from research_copilot.graph import final_answer
from research_copilot.multi_agent_graph import (
    build_multi_agent_graph,
    multi_agent_turn_input,
    run_multi_agent,
)
from research_copilot.multi_agent_state import (
    AGENTS,
    OwnershipError,
    budget_of,
    merge_budgets,
    owns,
)

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
    def __init__(self, script):
        self.script = iter(script)
        self.views: list[str] = []

    def with_structured_output(self, schema, **kwargs):
        def reply(messages):
            self.views.append(messages[-1].content)
            return next(self.script)

        return RunnableLambda(reply)


def decide(next_, rationale="because", brief=""):
    return SupervisorDecision(rationale=rationale, next=next_, researcher_brief=brief)


def tool_call(query, call_id):
    return AIMessage(
        content="",
        tool_calls=[{"name": "search_arxiv", "args": {"query": query}, "id": call_id, "type": "tool_call"}],
    )


@tool
def search_arxiv(query: str) -> str:
    """Stub search."""
    return f"[1] A paper about {query}"


@tool
def verify_citation(arxiv_id: str) -> str:
    """Stub verifier."""
    return f"FOUND: {arxiv_id} - A paper"


def build(*, supervisor, researcher=None, writer=None, critic=None, **kwargs):
    return build_multi_agent_graph(
        supervisor_model=supervisor,
        researcher_model=researcher or scripted(*["notes"] * 5),
        writer_model=writer or scripted(*[f"draft {i}" for i in range(1, 8)]),
        critic_model=critic,
        tools=[search_arxiv],
        critic_tools=[verify_citation],
        enable_critic=critic is not None,
        **kwargs,
    )


def routes(state):
    return [e["routed_to"] for e in state["supervisor_log"]]


def log_entry(routed_to, revision=0):
    return {
        "step": 0, "proposed": routed_to, "rationale": "", "routed_to": routed_to,
        "override": "", "brief": "", "revision": revision,
    }


# --- Part B: rejections go to the Supervisor --------------------------------------


def test_structural_rejection_routed_to_the_writer():
    supervisor = ScriptedSupervisor(
        [
            decide("researcher"), decide("writer"), decide("critic"),
            decide("writer", "structural: second sub-question unanswered"),
            decide("critic"),
        ]
    )
    state = run_multi_agent(
        "Q",
        graph=build(
            supervisor=supervisor,
            critic=scripted("REJECT\nWriting: skips part two though the notes cover it.", "APPROVE"),
        ),
    )
    assert routes(state) == ["researcher", "writer", "critic", "writer", "critic"]
    assert state["revisions"] == 1
    assert final_answer(state) == "draft 2"
    # The Supervisor saw the rejection and whose problem the critic said it was.
    rejection_view = supervisor.views[3]
    assert "Critique (REJECT, about the CURRENT draft)" in rejection_view
    assert "skips part two" in rejection_view
    assert "REJECTED: not rewritten since the last rejection" in rejection_view


def test_evidence_rejection_routed_to_the_researcher_then_writer():
    researcher = scripted("Findings: A", "Findings: benchmark B")
    writer = scripted("draft 1", "draft 2")
    supervisor = ScriptedSupervisor(
        [
            decide("researcher"), decide("writer"), decide("critic"),
            decide("researcher", "evidence: no benchmark source", brief="benchmarks for A"),
            decide("writer"), decide("critic"),
        ]
    )
    state = run_multi_agent(
        "Q",
        graph=build(
            supervisor=supervisor, researcher=researcher, writer=writer,
            critic=scripted("REJECT\nEvidence: no source for the benchmark claim.", "APPROVE"),
        ),
    )
    assert routes(state) == ["researcher", "writer", "critic", "researcher", "writer", "critic"]
    assert "benchmark B" in state["research_notes"]
    # The revising Writer got both the merged notes and the critique.
    revising = writer.requests[1]
    assert "benchmark B" in revising[-1].text
    assert "no source for the benchmark claim" in "\n".join(m.text for m in revising)


def test_the_cap_still_overrules_a_reviewer_that_never_approves():
    supervisor = ScriptedSupervisor(itertools.cycle([decide("researcher"), decide("writer"), decide("critic")]))
    state = run_multi_agent(
        "Q",
        graph=build(
            supervisor=ScriptedSupervisor(
                [decide("researcher"), decide("writer"), decide("critic")]
                + [decide("writer"), decide("critic")] * 5
            ),
            critic=scripted(*["REJECT\nstill no"] * 6),
            max_revisions=2,
        ),
    )
    assert state["revisions"] == 2
    assert final_answer(state).startswith("(draft withheld after 2 revisions - critic: still no")


def test_no_hardcoded_back_edge_to_the_writer_remains():
    """Every path out of a rejection goes through start_revision -> supervisor."""
    edges = {(e.source, e.target) for e in build_multi_agent_graph().get_graph().edges}
    assert ("critic", "writer") not in edges
    assert ("start_revision", "supervisor") in edges
    assert ("start_revision", "writer") not in edges
    assert ("review_draft", "writer") not in edges


# --- staleness, second case ---------------------------------------------------------


def test_critique_is_current_only_if_the_critic_ran_after_the_last_writer():
    base = {"verdict": "reject"}
    assert critique_is_current({**base, "supervisor_log": [log_entry("writer"), log_entry("critic")]})
    assert not critique_is_current(
        {**base, "supervisor_log": [log_entry("writer"), log_entry("critic"), log_entry("writer")]}
    )


def test_a_critique_superseded_twice_is_still_just_stale():
    state = {
        "verdict": "approve",
        "draft": "draft 3",
        "supervisor_log": [
            log_entry("researcher"), log_entry("writer"), log_entry("critic"),
            log_entry("writer"), log_entry("writer"),
        ],
        "dispatches": {"researcher": 1, "writer": 3, "critic": 1},
    }
    assert not critique_is_current(state)
    view = render_supervisor_view(state, {"researcher": 2, "writer": 4, "critic": 3}, roster=AGENTS)
    assert "about an EARLIER draft, since rewritten" in view
    route, why = apply_guards("finish", state, {"researcher": 2, "writer": 4, "critic": 3}, roster=AGENTS)
    assert (route, why) == ("critic", "finish proposed on a draft whose approval is stale")


def test_a_rejected_draft_is_not_finishable_until_rewritten_this_round():
    state = {
        "draft": "d", "revisions": 1, "verdict": "reject",
        "supervisor_log": [log_entry("researcher"), log_entry("writer"), log_entry("critic")],
        "dispatches": {"researcher": 1, "writer": 1, "critic": 1},
    }
    assert draft_was_rejected(state)
    caps = {"researcher": 2, "writer": 4, "critic": 3}
    assert apply_guards("finish", state, caps, roster=AGENTS) == ("writer", "finish proposed on a rejected draft")
    assert apply_guards("critic", state, caps, roster=AGENTS)[0] == "writer"

    rewritten = {**state, "supervisor_log": state["supervisor_log"] + [log_entry("writer", revision=1)]}
    assert not draft_was_rejected(rewritten)


def test_finish_without_critic_approval_is_sent_to_the_critic():
    supervisor = ScriptedSupervisor([decide("researcher"), decide("writer"), decide("finish", "looks fine")])
    state = run_multi_agent("Q", graph=build(supervisor=supervisor, critic=scripted("APPROVE")))
    last = state["supervisor_log"][-1]
    assert (last["proposed"], last["routed_to"]) == ("finish", "critic")
    assert last["override"] == "finish proposed before the critic approved this draft"
    assert final_answer(state) == "draft 1"


# --- the human gate --------------------------------------------------------------------


def gated(supervisor, critic=None, **kwargs):
    return build(supervisor=supervisor, critic=critic, require_approval=True, checkpointer=MemorySaver(), **kwargs)


def test_critic_first_then_human_approves():
    g = gated(ScriptedSupervisor([decide("researcher"), decide("writer"), decide("critic")]), critic=scripted("APPROVE"))
    config = {"configurable": {"thread_id": "t"}}
    paused = g.invoke(multi_agent_turn_input("Q"), config)
    assert "__interrupt__" in paused
    assert paused["verdict"] == "approve"  # the critic went first

    state = g.invoke(Command(resume="approve"), config)
    assert final_answer(state) == "draft 1"


def test_human_edit_lands_in_its_own_field_not_in_the_writers_draft():
    g = gated(ScriptedSupervisor([decide("researcher"), decide("writer"), decide("finish")]))
    config = {"configurable": {"thread_id": "t"}}
    g.invoke(multi_agent_turn_input("Q"), config)
    state = g.invoke(Command(resume={"decision": "edit", "text": "human text"}), config)

    assert final_answer(state) == "human text"
    assert state["draft"] == "draft 1"
    assert state["human_edit"] == "human text"


def test_human_rejection_is_classified_by_the_supervisor_too():
    supervisor = ScriptedSupervisor(
        [decide("researcher"), decide("writer"), decide("finish"), decide("writer", "human wants it shorter"), decide("finish")]
    )
    g = gated(supervisor)
    config = {"configurable": {"thread_id": "t"}}
    g.invoke(multi_agent_turn_input("Q"), config)
    g.invoke(Command(resume={"decision": "reject", "note": "too long"}), config)
    state = g.invoke(Command(resume="approve"), config)

    assert "Human reviewer REJECTED the draft:\ntoo long" in supervisor.views[3]
    assert state["revisions"] == 1
    assert final_answer(state) == "draft 2"


def test_approval_without_a_checkpointer_is_refused_at_build_time():
    with pytest.raises(RuntimeError, match="needs a checkpointer"):
        build_multi_agent_graph(require_approval=True)


# --- Part C: budgets ---------------------------------------------------------------------


def test_merge_budgets_merges_per_agent_and_per_field():
    existing = {"researcher": {"used": 3, "cap": 6}, "writer": {"used": 1, "cap": 2}}
    merged = merge_budgets(existing, {"writer": {"used": 2}})
    assert merged == {"researcher": {"used": 3, "cap": 6}, "writer": {"used": 2, "cap": 2}}
    assert existing["writer"]["used"] == 1  # not mutated


def test_a_plain_node_may_write_only_its_own_budget_field():
    """Part B: ordinary per-field ownership now covers budgets."""

    @owns("writer")
    def writer_touching_critic_budget(state):
        return {"draft": "x", "writer_budget": {"used": 1}, "critic_budget": {"used": 0}}

    with pytest.raises(OwnershipError, match="critic_budget"):
        writer_touching_critic_budget({})


def test_nothing_may_write_the_legacy_budgets_dict():
    @owns("writer")
    def writer_using_the_old_dict(state):
        return {"draft": "x", "budgets": {"writer": {"used": 1}}}

    with pytest.raises(OwnershipError, match="budgets"):
        writer_using_the_old_dict({})


def test_all_three_agents_have_budget_entries_after_a_full_run():
    state = run_multi_agent(
        "Q",
        graph=build(
            supervisor=ScriptedSupervisor([decide("researcher"), decide("writer"), decide("critic")]),
            researcher=scripted(tool_call("x", "c1"), "notes"),
            critic=scripted("APPROVE"),
        ),
    )
    assert state["researcher_budget"] == {"used": 2, "cap": 6}
    assert state["writer_budget"] == {"used": 1, "cap": 2}
    assert state["critic_budget"] == {"used": 1, "cap": 4}


def test_research_budget_is_shared_across_passes_within_a_round():
    """The 6.1 gap, closed: a second pass in the same round gets what is left,
    not a fresh budget."""
    researcher = scripted(tool_call("a", "1"), tool_call("b", "2"), "notes one", tool_call("c", "3"), "never reached")
    state = run_multi_agent(
        "Q",
        graph=build(
            supervisor=ScriptedSupervisor(
                [decide("researcher"), decide("researcher", brief="more"), decide("writer"), decide("finish")]
            ),
            researcher=researcher,
            max_research_iterations=4,
        ),
    )
    assert state["researcher_budget"]["used"] == 4
    assert state["research_outcome"] == "budget_exhausted"
    assert len(researcher.requests) == 4


def test_start_revision_resets_every_agents_round_budget_but_not_dispatches():
    state = run_multi_agent(
        "Q",
        graph=build(
            supervisor=ScriptedSupervisor(
                [decide("researcher"), decide("writer"), decide("critic"), decide("writer"), decide("critic")]
            ),
            researcher=scripted(tool_call("x", "1"), "notes"),
            critic=scripted("REJECT\nfix it", "APPROVE"),
        ),
    )
    # Round 2 (after the rejection): only the Writer and Critic ran.
    assert state["researcher_budget"]["used"] == 0
    assert state["writer_budget"]["used"] == 1
    assert state["critic_budget"]["used"] == 1
    # Per turn, never reset by the revision.
    assert state["dispatches"] == {"researcher": 1, "writer": 2, "critic": 2}


def test_guard_refuses_an_agent_whose_round_budget_is_spent():
    state = {"dispatches": {"researcher": 1}, "budgets": {"researcher": {"used": 6, "cap": 6}}}
    route, why = apply_guards(
        "researcher", state, {"researcher": 2, "writer": 2}, round_caps={"researcher": 6}
    )
    assert route == "writer"
    assert why == "researcher round budget (6 model calls) spent"


def test_dispatch_caps_leave_room_for_every_revision():
    caps = default_dispatch_caps(enable_critic=True, max_revisions=3)
    assert caps["critic"] == 4
    assert caps["writer"] >= 4
    assert default_dispatch_caps(enable_critic=False, max_revisions=3)["critic"] == 0


@pytest.mark.parametrize(
    "script",
    [
        itertools.repeat(decide("critic")),
        itertools.repeat(decide("finish")),
        itertools.cycle([decide("writer"), decide("critic")]),
        itertools.cycle([decide("researcher", brief="b"), decide("critic")]),
    ],
    ids=["always-critic", "always-finish", "writer-critic", "research-critic"],
)
def test_no_supervisor_can_loop_forever_with_a_critic_that_always_rejects(script):
    max_revisions = 2
    state = run_multi_agent(
        "Q",
        graph=build(
            supervisor=ScriptedSupervisor(script),
            critic=scripted(*["REJECT\nno"] * 10),
            max_revisions=max_revisions,
        ),
    )
    caps = default_dispatch_caps(enable_critic=True, max_revisions=max_revisions)
    assert len(state["supervisor_log"]) <= sum(caps.values()) + max_revisions + 1
    assert state["revisions"] <= max_revisions
    assert [m.type for m in state["messages"]] == ["human", "ai"]


# --- schema migration: a thread from before 6.3 ------------------------------------


def test_a_pre_6_3_thread_with_no_budgets_resumes_cleanly():
    """Phase 4's rename-a-key exercise, for the budget structure. A thread
    checkpointed by 6.2 has `dispatches` but no `budgets`, `verdict`, or
    `revisions`. `budget_of` reads missing entries as "nothing used,
    configured cap", and the turn boundary writes every entry explicitly, so
    the next turn starts complete."""
    assert budget_of({}, "critic", 4) == {"used": 0, "cap": 4}
    assert budget_of({"budgets": {"writer": {"used": 1}}}, "writer", 2) == {"used": 1, "cap": 2}

    saver = MemorySaver()
    config = {"configurable": {"thread_id": "old"}}
    g = build(
        supervisor=ScriptedSupervisor([decide("researcher"), decide("writer"), decide("critic")]),
        critic=scripted("APPROVE"),
        checkpointer=saver,
    )
    # Write a 6.2-shaped state directly: no 6.3 keys at all.
    g.update_state(
        config,
        {
            "messages": [HumanMessage(content="old q"), AIMessage(content="old a")],
            "dispatches": {"researcher": 1, "writer": 1},
            "supervisor_log": [],
        },
        as_node="finalize_answer",
    )
    # Found while writing this test: a key declared with a reducer does not
    # read back as *missing* on an old thread. LangGraph gives a reducer
    # channel an empty default, so `budgets` is `{}`. A reader that tested for
    # the key's presence would be wrong. `budget_of` reads entries with
    # defaults, so both shapes - absent and empty - behave the same.
    assert g.get_state(config).values.get("budgets") == {}
    assert "verdict" not in g.get_state(config).values  # plain keys *are* missing

    state = g.invoke(multi_agent_turn_input("new q"), config)
    assert final_answer(state) == "draft 1"
    assert all(state.get(f"{agent}_budget") is not None for agent in AGENTS)
    assert state["dispatches"] == {"researcher": 1, "writer": 1, "critic": 1}


def test_the_researcher_returns_only_its_own_budget():
    """Part B: what replaced 6.3's passthrough test. The subgraph's output
    carries no other agent's budget - not even an unchanged copy - so there is
    nothing a parallel branch could overwrite."""
    from research_copilot.multi_agent_state import ResearcherOutput, CriticOutput

    assert {k for k in ResearcherOutput.__annotations__ if "budget" in k} == {"researcher_budget"}
    assert {k for k in CriticOutput.__annotations__ if "budget" in k} == {"critic_budget"}
    assert "budgets" not in ResearcherOutput.__annotations__


# --- CLI ------------------------------------------------------------------------------


def test_cli_critic_flag_puts_the_critic_on_the_roster(monkeypatch, capsys):
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    shared = scripted("Findings: A", "The answer.", "APPROVE")
    supervisor = ScriptedSupervisor([decide("researcher"), decide("writer"), decide("finish")])
    for module in ("agents.researcher", "agents.writer", "agents.critic"):
        monkeypatch.setattr(f"research_copilot.{module}.get_chat_model", lambda **k: shared)
    monkeypatch.setattr("research_copilot.agents.supervisor.get_chat_model", lambda **k: supervisor)

    assert cli.main(["multi-agent", "Q", "--critic"]) == 0
    out, err = capsys.readouterr()
    assert "The answer." in out
    assert "OVERRIDDEN: finish proposed before the critic approved this draft" in err
    assert "[critic] APPROVE" in err
    assert "roster: researcher, writer, critic" in err


# --- Phase 7: "incomplete" is not "rejected" (E2E01, Groq) -------------------------------

GROQ_TOOL_ERROR = RuntimeError("Error code: 400 - tool_use_failed: not in request.tools")


class FailingThenScripted(RecordingModel):
    """Raises for the first `failures` calls, then follows its script."""

    failures: int = 0

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if self.failures > 0:
            self.failures -= 1
            self.requests.append(list(messages))
            raise GROQ_TOOL_ERROR
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def test_an_incomplete_review_spends_no_revision_and_forces_no_rewrite():
    """E2E01's shape: the first review cannot finish (model fails twice). The
    Supervisor sends the Critic again - and the old guard would have forced a
    rewrite here. Now: no revision spent, the Writer is not re-dispatched, and
    the second review approves the same draft."""
    critic = FailingThenScripted(responses=[AIMessage(content="APPROVE")], failures=2)
    supervisor = ScriptedSupervisor(
        [decide("researcher"), decide("writer"), decide("critic"),
         decide("critic", "the review could not finish; run it again")]
    )
    state = run_multi_agent("Q", graph=build(supervisor=supervisor, critic=critic))

    assert routes(state) == ["researcher", "writer", "critic", "critic"]
    assert all(e["override"] == "" for e in state["supervisor_log"])
    assert state["revisions"] == 0
    assert state["dispatches"]["writer"] == 1
    assert final_answer(state) == "draft 1"
    assert "INCOMPLETE" in supervisor.views[3]


def test_a_critic_that_never_finishes_ends_in_a_delivered_unreviewed_answer():
    """When the Critic cannot finish and its dispatches run out, the draft is
    delivered - not withheld, which delivered nothing in E2E01 - and marked as
    unreviewed. Never presented as approved."""
    critic = FailingThenScripted(responses=[AIMessage(content="unused")], failures=100)
    state = run_multi_agent(
        "Q",
        graph=build(
            supervisor=ScriptedSupervisor(itertools.repeat(decide("critic"))),
            critic=critic, routing="fixed", max_revisions=1,
        ),
    )
    assert state["verdict"] == "incomplete"
    assert state["revisions"] == 0
    answer = final_answer(state)
    assert answer.startswith("(Note: this answer was not checked by the reviewer")
    assert "draft 1" in answer


def test_a_garbled_verdict_is_still_a_rejection():
    """Only a review that could not *finish* is incomplete. One that finished
    and said something unreadable still fails closed (Phase 5)."""
    state = run_multi_agent(
        "Q",
        graph=build(
            supervisor=ScriptedSupervisor([decide("researcher"), decide("writer"), decide("critic")]),
            critic=scripted("Looks fine I guess"), max_revisions=0,
        ),
    )
    assert state["verdict"] == "reject"
    assert "withheld" in final_answer(state)


def test_budget_of_reads_the_legacy_dict_on_an_old_thread():
    """Threads checkpointed between 6.3 and Part B have `budgets`, not the
    per-agent fields. budget_of falls back to it - the migration path."""
    legacy = {"budgets": {"critic": {"used": 3, "cap": 4}}}
    assert budget_of(legacy, "critic", 4) == {"used": 3, "cap": 4}
    # The new field wins once it exists (a new turn writes it at begin_turn).
    assert budget_of({**legacy, "critic_budget": {"used": 0}}, "critic", 4) == {"used": 0, "cap": 4}


def test_a_thread_from_before_part_b_resumes_on_the_new_fields():
    saver = MemorySaver()
    config = {"configurable": {"thread_id": "pre-b"}}
    g = build(
        supervisor=ScriptedSupervisor([decide("researcher"), decide("writer"), decide("critic")]),
        critic=scripted("APPROVE"), checkpointer=saver,
    )
    g.update_state(
        config,
        {"messages": [HumanMessage(content="old q"), AIMessage(content="old a")],
         "budgets": {"researcher": {"used": 5, "cap": 6}, "critic": {"used": 4, "cap": 4}},
         "supervisor_log": []},
        as_node="finalize_answer",
    )
    state = g.invoke(multi_agent_turn_input("new q"), config)
    assert final_answer(state) == "draft 1"
    assert state["critic_budget"]["used"] == 1       # fresh turn, fresh field
    assert state["budgets"]["critic"]["used"] == 4   # legacy left as it was, unread
