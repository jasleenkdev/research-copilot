"""Phase 7 A1: a ceiling on the size of one model request - offline.

From E2E01's third attempt on Groq (`413 Request too large: Requested 8849`,
limit 8000). The fix sizes each agent's request before sending it, cuts only
*material* (search results, notes - never the draft being judged), logs every
cut in state, and retries once smaller if the provider still refuses.
"""

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from research_copilot.agents.critic import build_critic
from research_copilot.agents.researcher import build_researcher
from research_copilot.agents.supervisor import render_supervisor_view
from research_copilot.agents.writer import make_writer
from research_copilot.request_budget import (
    DEFAULT_MAX_REQUEST_TOKENS,
    fit_text,
    max_request_tokens,
    text_tokens,
)

TOO_LARGE = RuntimeError(
    "Error code: 413 - Request too large for model on tokens per minute (TPM): Limit 8000, Requested 8849"
)
LONG = ("Findings: a long line of evidence about retrieval evaluation. " * 400).strip()


class Recording(FakeMessagesListChatModel):
    requests: list = []
    fail_first: int = 0

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.requests.append(list(messages))
        if self.fail_first > 0:
            self.fail_first -= 1
            raise TOO_LARGE
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def scripted(*replies, fail_first=0):
    return Recording(responses=[r if isinstance(r, AIMessage) else AIMessage(content=r) for r in replies],
                     fail_first=fail_first)


def text_of(messages):
    return "\n".join(m.text for m in messages)


@tool("search_arxiv")
def big_search(query: str, max_results: int = 5) -> str:
    """Stub returning a very long result."""
    return f"[1] Result for {query}\n    URL: http://arxiv.org/abs/2309.15217\n    Abstract: " + LONG


@tool("verify_citation")
def verify(arxiv_id: str) -> str:
    """Stub."""
    return f"FOUND: {arxiv_id} - a paper"


@pytest.fixture
def small_limit(monkeypatch):
    monkeypatch.setenv("RESEARCH_COPILOT_MAX_REQUEST_TOKENS", "1500")
    return 1500


# --- the primitives -------------------------------------------------------------------------


def test_fit_text_keeps_the_head_and_marks_the_cut():
    fitted, before, after = fit_text(LONG, 200)
    assert fitted.startswith("Findings: a long line")
    assert "characters cut to fit the request size limit" in fitted
    assert after < before and after <= 260


def test_fit_text_leaves_short_text_alone():
    assert fit_text("short", 100) == ("short", text_tokens("short"), text_tokens("short"))


def test_limits_per_provider_and_override(monkeypatch):
    monkeypatch.delenv("RESEARCH_COPILOT_MAX_REQUEST_TOKENS", raising=False)
    assert max_request_tokens("groq") == DEFAULT_MAX_REQUEST_TOKENS["groq"] < 8000
    assert max_request_tokens("anthropic") == 100_000
    monkeypatch.setenv("RESEARCH_COPILOT_MAX_REQUEST_TOKENS", "1234")
    assert max_request_tokens("groq") == 1234


# --- each agent -------------------------------------------------------------------------------


def researcher_input(**extra):
    return {"question": "Q", "mode": "live-search", "messages": [HumanMessage(content="Q")], **extra}


def search_call(query, call_id):
    return AIMessage(content="", tool_calls=[{"name": "search_arxiv", "args": {"query": query}, "id": call_id, "type": "tool_call"}])


def test_researcher_trims_search_results_and_logs_it(small_limit):
    model = scripted(search_call("a", "1"), search_call("b", "2"), "- Findings: x\n- Sources: y\n- Gaps: none")
    out = build_researcher(model=model, tools=[big_search], max_iterations=5).invoke(researcher_input())
    # Every request after results came back is inside the limit (estimated).
    from research_copilot.request_budget import estimate

    assert all(estimate(r) <= small_limit * 1.1 for r in model.requests[1:])
    assert out["researcher_trims"]
    assert out["researcher_trims"][0]["node"] == "researcher"
    assert "search results" in out["researcher_trims"][0]["part"]
    assert out["research_outcome"] == "findings"


def test_researcher_final_call_trims_the_flattened_results(small_limit):
    model = scripted(search_call("a", "1"), "- Findings: x\n- Sources: y\n- Gaps: none")
    out = build_researcher(model=model, tools=[big_search], max_iterations=2).invoke(researcher_input())
    assert any(t["part"] == "search results (final call)" for t in out["researcher_trims"])


def test_critic_never_cuts_the_draft_only_the_notes(small_limit):
    draft = "The draft cites RAGAS [2309.15217]. " + "Important claim. " * 20
    critic = scripted("APPROVE")
    out = build_critic(model=critic, tools=[verify]).invoke(
        {"question": "Q", "draft": draft, "research_notes": LONG}
    )
    sent = text_of(critic.requests[0])
    assert draft.strip() in sent
    assert "characters cut to fit" in sent
    assert out["critic_trims"][0]["part"] == "research notes"


def test_writer_trims_notes_and_logs_it(small_limit):
    writer = scripted("An answer.")
    out = make_writer(model=writer)({"question": "Q", "research_notes": LONG, "messages": []})
    assert "characters cut to fit" in text_of(writer.requests[0])
    assert out["writer_trims"][0]["part"] == "research notes"


def test_no_trim_no_log_under_the_default_limit(monkeypatch):
    monkeypatch.delenv("RESEARCH_COPILOT_MAX_REQUEST_TOKENS", raising=False)
    out = make_writer(model=scripted("An answer."))({"question": "Q", "research_notes": "short notes", "messages": []})
    assert "writer_trims" not in out


# --- the provider still refuses -------------------------------------------------------------


def test_provider_refusal_retries_once_smaller_and_logs_it(monkeypatch):
    monkeypatch.setenv("RESEARCH_COPILOT_MAX_REQUEST_TOKENS", "4000")
    critic = scripted("APPROVE", fail_first=1)
    out = build_critic(model=critic, tools=[verify]).invoke(
        {"question": "Q", "draft": "RAGAS [2309.15217]", "research_notes": LONG}
    )
    assert out["verdict"] == "approve"
    assert len(critic.requests) == 2
    assert len(text_of(critic.requests[1])) < len(text_of(critic.requests[0]))
    assert [t["limit"] for t in out["critic_trims"]] == [4000, 2800]


def test_a_request_that_can_never_fit_degrades_instead_of_crashing(monkeypatch):
    monkeypatch.setenv("RESEARCH_COPILOT_MAX_REQUEST_TOKENS", "4000")
    critic = scripted("unused", fail_first=2)
    out = build_critic(model=critic, tools=[verify]).invoke(
        {"question": "Q", "draft": "RAGAS [2309.15217]", "research_notes": LONG}
    )
    assert out["verdict"] == "incomplete"


# --- visibility -------------------------------------------------------------------------------


def test_supervisor_is_shown_the_trims():
    view = render_supervisor_view(
        {"question": "Q", "critic_trims": [{"node": "critic", "part": "research notes",
                                            "tokens_before": 9000, "tokens_after": 5000, "limit": 6500}]},
        {"researcher": 2, "writer": 2},
    )
    assert "Requests trimmed to fit the size limit this turn: critic research notes 9000->5000" in view


def test_trims_reach_the_parent_state_and_reset_each_turn(small_limit):
    from langgraph.checkpoint.memory import MemorySaver

    from research_copilot.multi_agent_graph import build_multi_agent_graph, run_multi_agent

    g = build_multi_agent_graph(
        researcher_model=scripted(search_call("a", "1"), "notes one", "notes two"),
        writer_model=scripted("answer one", "answer two"),
        tools=[big_search], routing="fixed", checkpointer=MemorySaver(), memory_strategy="none",
    )
    config = {"configurable": {"thread_id": "t"}}
    first = run_multi_agent("Q1", graph=g, config=config)
    assert first["researcher_trims"]
    second = run_multi_agent("Q2", graph=g, config=config)  # no search: nothing to trim
    assert second["researcher_trims"] == []


def test_call_recorder_names_the_node_and_subgraph_of_every_call():
    from research_copilot.live_check.runner import CallRecorder
    from research_copilot.multi_agent_graph import build_multi_agent_graph, multi_agent_turn_input

    g = build_multi_agent_graph(
        researcher_model=scripted("notes"), writer_model=scripted("answer"),
        tools=[big_search], routing="fixed", memory_strategy="none",
    )
    recorder = CallRecorder()
    g.invoke(multi_agent_turn_input("Q"), {"callbacks": [recorder]})
    seen = [(c["agent"], c["node"], c["status"]) for c in recorder.calls]
    # `agent` is the top-level node the call happened under; `node` is the node
    # that made it (inside the subgraph, for the Researcher).
    assert seen == [("researcher", "research_model", "ok"), ("writer", "writer", "ok")]
    assert all(c["est_tokens"] > 0 for c in recorder.calls)
