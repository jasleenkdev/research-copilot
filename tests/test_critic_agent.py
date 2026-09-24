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
        critic=scripted(verify("2309.15217", "v1"), "APPROVE"),
        supervisor_script=[decide("researcher"), decide("writer"), decide("critic")],
    )
    state = run_multi_agent("Q", graph=graph)

    assert [m.type for m in state["messages"]] == ["human", "ai"]
    assert not any(isinstance(m, ToolMessage) for m in state["messages"])
    assert not any(getattr(m, "tool_calls", None) for m in state["messages"])
    assert "critic_messages" not in state


def test_critic_verification_is_not_shown_to_the_writer_on_revision():
    writer = scripted("draft one", "draft two")
    graph, _ = critic_run(
        critic=scripted(verify("9999.99999", "v1"), "REJECT\nCitation 9999.99999 does not exist.", "APPROVE"),
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
        critic=scripted(verify("2309.15217", "v1"), "APPROVE"),
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
    assert any("FOUND: 2309.15217" in repr(v.get("critic_messages", [])) for v in critic_namespaces)


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


def test_citation_checks_are_recorded_as_structured_results():
    graph, _ = critic_run(
        critic=scripted(
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "verify_citation", "args": {"arxiv_id": "2309.15217"}, "id": "a", "type": "tool_call"},
                    {"name": "verify_citation", "args": {"arxiv_id": "1111.11111"}, "id": "b", "type": "tool_call"},
                ],
            ),
            "REJECT\n1111.11111 does not exist",
        ),
        supervisor_script=[decide("researcher"), decide("writer"), decide("critic")],
        max_revisions=0,
    )
    state = run_multi_agent("Q", graph=graph)
    assert state["citation_checks"] == [
        {"arxiv_id": "2309.15217", "status": "found"},
        {"arxiv_id": "1111.11111", "status": "not_found"},
    ]


def test_critic_is_shown_the_research_notes_it_checks_the_draft_against():
    critic = scripted("APPROVE")
    graph, _ = critic_run(
        critic=critic,
        researcher=scripted("Findings: unique-evidence-marker"),
        supervisor_script=[decide("researcher"), decide("writer"), decide("critic")],
    )
    run_multi_agent("Q", graph=graph)
    assert "unique-evidence-marker" in "\n".join(m.text for m in critic.requests[0])


def test_critic_budget_running_out_mid_verification_fails_closed():
    graph, _ = critic_run(
        critic=scripted(verify("a", "1"), verify("b", "2"), verify("c", "3")),
        supervisor_script=[decide("researcher"), decide("writer"), decide("critic")],
        max_critic_iterations=2,
        max_revisions=0,
    )
    state = run_multi_agent("Q", graph=graph)
    assert state["verdict"] == "reject"
    assert "budget" in state["critique"]
    assert state["budgets"]["critic"]["used"] == 2
