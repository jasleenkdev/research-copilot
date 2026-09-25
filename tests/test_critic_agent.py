"""Phase 6.3: the Critic agent - its private channel, verdicts, and tool.

The leak-check tests at the top were written *before* the Critic existed, on
purpose. 6.1 found the Researcher's disk-persistence gap after the fact; this
time the privacy contract is pinned down first and the agent is built to pass
it.

As in every offline test: the critic is scripted. These tests say nothing
about whether a real Critic finds real problems, or whether its citation checks
catch real fabrications.
"""

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

from research_copilot.agents.supervisor import SupervisorDecision
from research_copilot.graph import final_answer
from research_copilot.multi_agent_graph import build_multi_agent_graph, run_multi_agent
from research_copilot.multi_agent_state import OWNERS, CriticOutput

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
            item = next(self.script)
            if isinstance(item, Exception):
                raise item
            return item

        return RunnableLambda(reply)


def decide(next_, rationale="because", brief=""):
    return SupervisorDecision(rationale=rationale, next=next_, researcher_brief=brief)


def verify(arxiv_id, call_id):
    return AIMessage(
        content="",
        tool_calls=[
            {"name": "verify_citation", "args": {"arxiv_id": arxiv_id}, "id": call_id, "type": "tool_call"}
        ],
    )


def stub_verifier(known: dict[str, str]):
    """A verify_citation stand-in: known ids are found, others are not."""

    @tool
    def verify_citation(arxiv_id: str) -> str:
        """Stub verifier."""
        if arxiv_id in known:
            return f"FOUND: {arxiv_id} - {known[arxiv_id]}"
        return f"NOT FOUND: no arXiv paper has the id {arxiv_id}"

    return verify_citation


@tool
def search_arxiv(query: str) -> str:
    """Stub search."""
    return f"[1] A paper about {query}"


def critic_run(*, critic, supervisor_script, writer=None, researcher=None, verifier=None, **kwargs):
    supervisor = ScriptedSupervisor(supervisor_script)
    graph = build_multi_agent_graph(
        supervisor_model=supervisor,
        researcher_model=researcher or scripted("Findings: X (2309.15217)"),
        writer_model=writer or scripted("Draft citing 2309.15217"),
        critic_model=critic,
        tools=[search_arxiv],
        critic_tools=[verifier or stub_verifier({"2309.15217": "RAGAS"})],
        enable_critic=True,
        **kwargs,
    )
    return graph, supervisor


# --- the privacy contract, written first -------------------------------------------


def test_critic_verification_traffic_never_reaches_the_shared_transcript():
    graph, _ = critic_run(
        critic=scripted("APPROVE"),
        supervisor_script=[decide("researcher"), decide("writer"), decide("critic")],
    )
    state = run_multi_agent("Q", graph=graph)

    assert [m.type for m in state["messages"]] == ["human", "ai"]
    assert not any(isinstance(m, ToolMessage) for m in state["messages"])
    assert not any(getattr(m, "tool_calls", None) for m in state["messages"])
    assert "critic_messages" not in state


def test_critic_verification_is_not_shown_to_the_writer_on_revision():
    writer = scripted("draft one citing 2309.15217", "draft two")
    graph, _ = critic_run(
        critic=scripted("REJECT\nCitation 9999.99999 does not exist.", "APPROVE"),
        writer=writer,
        supervisor_script=[
            decide("researcher"), decide("writer"), decide("critic"),
            decide("writer", "structural: remove the bad citation"), decide("critic"),
        ],
    )
    run_multi_agent("Q", graph=graph)

    revision_request = writer.requests[1]
    assert not any(isinstance(m, ToolMessage) for m in revision_request)
    assert "NOT FOUND" not in "\n".join(m.text for m in revision_request)
    # The critique text itself is what the Writer should get.
    assert "does not exist" in "\n".join(m.text for m in revision_request)


def test_critic_private_channel_is_hidden_from_state_not_from_disk():
    """Same limit as the Researcher's (6.1): private from other agents and from
    get_state(), but checkpointed under the critic's own namespace."""
    saver = MemorySaver()
    graph, _ = critic_run(
        critic=scripted("APPROVE"),
        supervisor_script=[decide("researcher"), decide("writer"), decide("critic")],
        checkpointer=saver,
    )
    config = {"configurable": {"thread_id": "t"}}
    run_multi_agent("Q", graph=graph, config=config)

    assert "FOUND: 2309.15217" not in repr(graph.get_state(config).values)
    critic_namespaces = [
        c.checkpoint["channel_values"]
        for c in saver.list(None)
        if c.config["configurable"].get("checkpoint_ns", "").startswith("critic:")
    ]
    assert critic_namespaces
    # Phase 7: the raw lookups live in the private `lookups` key now.
    assert any("FOUND: 2309.15217" in repr(v.get("lookups", [])) for v in critic_namespaces)


def test_critic_output_schema_and_owners_table_agree():
    assert OWNERS["critic"] == frozenset(CriticOutput.__annotations__)


# --- verdicts: Phase 5's parsing, ported -----------------------------------------------


def test_an_approving_critic_commits_the_draft():
    graph, _ = critic_run(
        critic=scripted("APPROVE"),
        supervisor_script=[decide("researcher"), decide("writer"), decide("critic")],
    )
    state = run_multi_agent("Q", graph=graph)
    assert final_answer(state) == "Draft citing 2309.15217"
    assert state["verdict"] == "approve"
    assert state.get("revisions", 0) == 0


@pytest.mark.parametrize("garbled", ["Looks good to me!", "", "I would not APPROVE this"])
def test_unparseable_critique_fails_closed_to_a_rejection(garbled):
    graph, _ = critic_run(
        critic=scripted(garbled),
        supervisor_script=[decide("researcher"), decide("writer"), decide("critic")],
        max_revisions=0,
    )
    state = run_multi_agent("Q", graph=graph)
    assert state["verdict"] == "reject"
    assert "withheld" in final_answer(state)


def test_citation_checks_are_run_by_code_for_every_cited_id():
    """Phase 7: the Critic's model no longer decides what to look up. Every
    arXiv reference in the draft is checked before the judging call."""
    critic = scripted("REJECT\n1111.11111 does not exist")
    graph, _ = critic_run(
        critic=critic,
        writer=scripted("RAGAS [http://arxiv.org/abs/2309.15217v2] and X [1111.11111], RAGAS again [2309.15217]"),
        supervisor_script=[decide("researcher"), decide("writer"), decide("critic")],
        max_revisions=0,
    )
    state = run_multi_agent("Q", graph=graph)
    assert state["citation_checks"] == [
        {"arxiv_id": "2309.15217", "status": "found"},
        {"arxiv_id": "1111.11111", "status": "not_found"},
    ]
    judging_request = "\n".join(m.text for m in critic.requests[0])
    assert "Citation checks (already run; treat as fact)" in judging_request
    assert "1111.11111: NOT FOUND" in judging_request
    # One judging call, no tool loop.
    assert len(critic.requests) == 1


def test_critic_is_shown_the_research_notes_it_checks_the_draft_against():
    critic = scripted("APPROVE")
    graph, _ = critic_run(
        critic=critic,
        researcher=scripted("Findings: unique-evidence-marker"),
        supervisor_script=[decide("researcher"), decide("writer"), decide("critic")],
    )
    run_multi_agent("Q", graph=graph)
    assert "unique-evidence-marker" in "\n".join(m.text for m in critic.requests[0])


def test_a_review_costs_one_model_call_however_many_citations():
    graph, _ = critic_run(
        critic=scripted("APPROVE"),
        writer=scripted("A [2309.15217], B [2005.11401], C [1706.03762], D [2311.09476]"),
        supervisor_script=[decide("researcher"), decide("writer"), decide("critic")],
    )
    state = run_multi_agent("Q", graph=graph)
    assert len(state["citation_checks"]) == 4
    assert state["critic_budget"]["used"] == 1
    assert state["verdict"] == "approve"


def test_a_critic_that_cannot_run_is_incomplete_not_rejected():
    from research_copilot.agents.critic import build_critic

    critic = build_critic(model=scripted("unused"), tools=[stub_verifier({})], max_iterations=1)
    out = critic.invoke({"question": "Q", "draft": "d", "critic_budget": {"used": 1, "cap": 1}})
    assert out["verdict"] == "incomplete"
    assert "says nothing about the draft" in out["critique"]



# --- Part B: the Writer's flag reaches the Critic --------------------------------------


def test_the_critic_is_shown_the_writers_unsupported_citations():
    from research_copilot.agents.critic import build_critic

    critic = scripted("REJECT\nARES is not in the notes")
    build_critic(model=critic, tools=[stub_verifier({"2311.09476": "ARES"})]).invoke(
        {"question": "Q", "draft": "ARES [2311.09476]", "research_notes": "notes",
         "unsupported_citations": ["arXiv 2311.09476"]}
    )
    sent = "\n".join(m.text for m in critic.requests[0])
    assert "NOT in the research notes (found by code; treat as fact)" in sent
    assert "- arXiv 2311.09476" in sent


def test_with_no_flags_the_critic_is_told_so_explicitly():
    from research_copilot.agents.critic import build_critic

    critic = scripted("APPROVE")
    build_critic(model=critic, tools=[stub_verifier({})]).invoke({"question": "Q", "draft": "d", "research_notes": "n"})
    assert "(none - every citation in the draft appears in the notes)" in "\n".join(m.text for m in critic.requests[0])


def test_in_the_graph_the_writers_flag_reaches_the_critic():
    """The coordination gap, closed end to end: the Writer's own check flags
    a citation the notes lack; the Critic's judging request contains it."""
    critic = scripted("REJECT\nunsupported", "APPROVE")
    graph, _ = critic_run(
        critic=critic,
        researcher=scripted("Findings: X (2309.15217)"),
        # Both drafts cite a paper the notes lack, so the Writer's retry does
        # not clear the flag.
        writer=scripted("RAGAS [2309.15217] and ARES [2311.09476]", "still ARES [2311.09476]"),
        supervisor_script=[decide("researcher"), decide("writer"), decide("critic")],
        max_revisions=0,
    )
    state = run_multi_agent("Q", graph=graph)
    assert state["unsupported_citations"] == ["arXiv 2311.09476"]
    assert "- arXiv 2311.09476" in "\n".join(m.text for m in critic.requests[0])
    assert state["verdict"] == "reject"
