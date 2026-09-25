"""Phase 7 Part C: streaming a multi-agent run - offline.

What these pin down: the normalised events (order, content), that nothing
private leaks through them, that interventions arrive *live* (inside the node,
not after it), and the two kinds of ending - a clean `stopped` for an expected
provider-side condition, and a loud `error` for our own bugs.
"""

import asyncio
import json
from pathlib import Path

import httpx
import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel, GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

from research_copilot import cli
from research_copilot.multi_agent_graph import build_multi_agent_graph, multi_agent_turn_input
from research_copilot.streaming import astream_run

FIXTURES = json.loads((Path(__file__).parent / "fixtures" / "provider_errors.json").read_text())


class NoStream(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


class Streams(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


@tool("search_arxiv")
def search(query: str, max_results: int = 5) -> str:
    """Stub."""
    return "[1] RAW-SEARCH-RESULT-TEXT http://arxiv.org/abs/2309.15217"


@tool("verify_citation")
def verify(arxiv_id: str) -> str:
    """Stub."""
    return f"FOUND: {arxiv_id} - RAW-LOOKUP-TEXT"


SEARCH = AIMessage(content="", tool_calls=[{"name": "search_arxiv", "args": {"query": "ragas"}, "id": "c1", "type": "tool_call"}])
NOTES = "- Findings: RAGAS (http://arxiv.org/abs/2309.15217)\n- Sources: s\n- Gaps: none"


def graph(*, writer_text="RAGAS evaluates RAG pipelines [2309.15217].", critic=None, **kwargs):
    return build_multi_agent_graph(
        researcher_model=NoStream(responses=[SEARCH, AIMessage(content=NOTES)]),
        writer_model=Streams(messages=iter([AIMessage(content=writer_text)])),
        critic_model=critic, enable_critic=critic is not None,
        tools=[search], critic_tools=[verify], routing="fixed", memory_strategy="none", **kwargs,
    )


def collect(g, graph_input=None, config=None, **kwargs):
    async def run():
        return [e async for e in astream_run(g, graph_input or multi_agent_turn_input("Q"), config, **kwargs)]
    return asyncio.run(run())


# --- the happy path ------------------------------------------------------------------------


def test_a_run_streams_decisions_tools_tokens_and_ends_done():
    events = collect(graph())
    kinds = [e["type"] for e in events]
    assert kinds[-1] == "done"
    assert [e["routed_to"] for e in events if e["type"] == "decision"] == ["researcher", "writer", "finish"]
    assert [(e["phase"], e["name"], e.get("arg")) for e in events if e["type"] == "tool"] == [
        ("start", "search_arxiv", "ragas"), ("end", "search_arxiv", None)]
    tokens = "".join(e["text"] for e in events if e["type"] == "token")
    assert tokens == events[-1]["answer"] == "RAGAS evaluates RAG pipelines [2309.15217]."
    # Tokens arrive while the Writer runs - between its start and end.
    writer_start = next(i for i, e in enumerate(events) if e == {"type": "node", "phase": "start", "agent": "writer", "node": "writer", "depth": 0})
    first_token = kinds.index("token")
    assert writer_start < first_token


def test_inner_subgraph_steps_are_reported_under_their_agent():
    inner = [(e["agent"], e["node"]) for e in collect(graph()) if e["type"] == "node" and e["depth"] == 1 and e["phase"] == "start"]
    assert inner == [("researcher", "research_model"), ("researcher", "research_tools"),
                     ("researcher", "research_model"), ("researcher", "compile_notes")]


def test_nothing_private_leaks_into_the_stream():
    """Measured in Part C: raw astream_events carries the Researcher's private
    research_messages (tool calls and raw results) and the Critic's raw
    lookups. The normalised stream is built from an allow-list, so none of it
    reaches a consumer."""
    critic = NoStream(responses=[AIMessage(content="APPROVE")])
    text = json.dumps(collect(graph(critic=critic, max_revisions=0)), default=str)
    for private in ("research_messages", "critic_messages", "lookups", "RAW-SEARCH-RESULT-TEXT", "RAW-LOOKUP-TEXT"):
        assert private not in text, private


def test_tokens_come_from_the_writer_only_by_default():
    critic = Streams(messages=iter([AIMessage(content="APPROVE")]))
    events = collect(graph(critic=critic, max_revisions=0))
    assert {e["agent"] for e in events if e["type"] == "token"} == {"writer"}
    wider = collect(graph(critic=Streams(messages=iter([AIMessage(content="APPROVE")])), max_revisions=0),
                    token_agents=("writer", "critic"))
    assert {e["agent"] for e in wider if e["type"] == "token"} == {"writer", "critic"}


def test_the_critics_verdict_is_an_event():
    events = collect(graph(critic=NoStream(responses=[AIMessage(content="APPROVE")]), max_revisions=0))
    (verdict,) = [e for e in events if e["type"] == "verdict"]
    assert verdict["verdict"] == "approve"
    assert verdict["citation_checks"] == [{"arxiv_id": "2309.15217", "status": "found"}]


# --- interventions, live ---------------------------------------------------------------------


def test_an_intervention_is_announced_inside_the_node_not_after_it():
    """The point of the custom event: the retry is visible while the Critic
    is still running. The state field shows it only once the Critic ends."""
    class Invents(NoStream):
        failures: int = 1

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            if self.failures:
                self.failures -= 1
                raise RuntimeError("Error code: 400 - tool_use_failed: attempted to call tool 'open_file' which was not in request.tools")
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    events = collect(graph(critic=Invents(responses=[AIMessage(content="APPROVE")]), max_revisions=0))
    start = next(i for i, e in enumerate(events) if e["type"] == "node" and e["node"] == "critic" and e["phase"] == "start" and e["depth"] == 0)
    end = next(i for i, e in enumerate(events) if e["type"] == "node" and e["node"] == "critic" and e["phase"] == "end" and e["depth"] == 0)
    (i, retry) = next((i, e) for i, e in enumerate(events) if e["type"] == "intervention")
    assert start < i < end
    assert (retry["node"], retry["kind"]) == ("critic", "hint_retry")


def test_trims_are_announced_live_too(monkeypatch):
    monkeypatch.setenv("RESEARCH_COPILOT_MAX_REQUEST_TOKENS", "300")
    events = collect(graph(writer_text="An answer."))
    trims = [e for e in events if e["type"] == "intervention" and e["kind"] == "trim"]
    assert trims and {t["node"] for t in trims} <= {"researcher", "writer"}


# --- the endings -----------------------------------------------------------------------------


def test_a_spent_quota_ends_the_stream_with_one_clean_stopped_event(monkeypatch):
    """Real daily-limit body, through the real Groq SDK."""
    from research_copilot.models import QuotaAwareTransport, get_chat_model

    monkeypatch.setenv("RESEARCH_COPILOT_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_not_real")
    fixture = FIXTURES["groq_429_tokens_per_day"]
    spent = get_chat_model()
    handler = lambda r: httpx.Response(fixture["status"], json=fixture["body"], headers=fixture["headers"])  # noqa: E731
    spent.client._client._client = httpx.Client(transport=QuotaAwareTransport(httpx.MockTransport(handler)))
    spent.async_client._client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    g = build_multi_agent_graph(researcher_model=spent, writer_model=spent, tools=[search],
                                routing="fixed", memory_strategy="none")
    events = collect(g)   # no exception escapes
    assert events[-1]["type"] == "stopped"
    assert events[-1]["reason"] == "quota_exhausted"
    assert "used 199939 of 200000" in events[-1]["detail"]


def test_a_bug_in_our_own_code_is_reported_and_then_re_raised(monkeypatch):
    """The other ending: not tidied into a clean stop (the README lesson)."""
    monkeypatch.setattr("research_copilot.agents.writer.unsupported_citations",
                        lambda *a, **k: (_ for _ in ()).throw(KeyError("our bug")))
    seen = []

    async def run():
        async for e in astream_run(graph(), multi_agent_turn_input("Q")):
            seen.append(e)

    with pytest.raises(KeyError, match="our bug"):
        asyncio.run(run())
    assert seen[-1]["type"] == "error" and "our bug" in seen[-1]["detail"]


def test_a_run_parked_at_the_human_gate_ends_paused():
    g = graph(require_approval=True, checkpointer=MemorySaver())
    events = collect(g, config={"configurable": {"thread_id": "t"}})
    assert events[-1]["type"] == "paused"
    assert events[-1]["payload"]["draft"] == "RAGAS evaluates RAG pipelines [2309.15217]."


def test_a_stream_started_with_bare_input_still_gets_a_fresh_turn():
    """The 6.4 lesson, for streaming callers: no helper to remember."""
    g = build_multi_agent_graph(
        researcher_model=NoStream(responses=[AIMessage(content=NOTES)] * 2),
        writer_model=Streams(messages=iter([AIMessage(content="a1"), AIMessage(content="a2")])),
        tools=[search], routing="fixed", memory_strategy="none", checkpointer=MemorySaver(),
    )
    config = {"configurable": {"thread_id": "t"}}
    bare = lambda q: {"question": q, "messages": [HumanMessage(content=q)]}  # noqa: E731
    collect(g, bare("one"), config)
    done = collect(g, bare("two"), config)[-1]
    assert done["type"] == "done" and done["answer"] == "a2"
    assert done["dispatches"] == {"researcher": 1, "writer": 1}


# --- CLI ----------------------------------------------------------------------------------------


def test_cli_live_flag_streams_the_answer_and_the_decisions(monkeypatch, capsys):
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setattr("research_copilot.agents.researcher.get_chat_model",
                        lambda **k: NoStream(responses=[AIMessage(content=NOTES)]))
    monkeypatch.setattr("research_copilot.agents.writer.get_chat_model",
                        lambda **k: Streams(messages=iter([AIMessage(content="Streamed answer [2309.15217].")])))
    assert cli.main(["multi-agent", "Q", "--routing", "fixed", "--checkpointer", "none", "--memory", "none", "--live"]) == 0
    out, err = capsys.readouterr()
    assert "Streamed answer [2309.15217]." in out
    assert "[decision] -> researcher" in err and "[decision] -> finish" in err
