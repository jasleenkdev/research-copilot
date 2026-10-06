"""Phase 7 Part F: the HTTP API - offline.

Fake models through `create_app(graph_factory=...)`, a MemorySaver, no model
check. What these pin down: the stream is `astream_run`'s events plus one
`thread` event first; the CLI's refusals come back as status codes before any
stream starts; a review resumes under the thread's stored policy, not the
request's; nothing private leaks through either the stream or the thread view.
"""

import json

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel, GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

from research_copilot.api import create_app
from research_copilot.multi_agent_graph import build_multi_agent_graph


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
ANSWER = "RAGAS evaluates RAG pipelines [2309.15217]."


def fake_factory(seen_policies: list):
    """Every graph the API builds gets fresh fakes. Routing is forced to
    `fixed` (no Supervisor model to script); the rest of the policy is the
    one the API passed, which is what the tests check."""

    def factory(policy, saver):
        seen_policies.append(dict(policy))
        p = {**policy, "routing": "fixed", "memory_strategy": "none"}
        return build_multi_agent_graph(
            **p,
            researcher_model=NoStream(responses=[SEARCH, AIMessage(content=NOTES)]),
            writer_model=Streams(messages=iter([AIMessage(content=ANSWER)] * 3)),
            critic_model=NoStream(responses=[AIMessage(content="APPROVE")] * 3),
            tools=[search], critic_tools=[verify], checkpointer=saver,
        )

    return factory


@pytest.fixture
def api():
    policies: list = []
    app = create_app(saver=MemorySaver(), graph_factory=fake_factory(policies), model_check=None)
    with TestClient(app) as client:
        client.policies = policies
        yield client


def events_of(response) -> list[dict]:
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")
    return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]


def run(client, **body) -> list[dict]:
    return events_of(client.post("/api/runs", json={"question": "How are RAG pipelines evaluated?", **body}))


# --- runs ---------------------------------------------------------------------------------


def test_a_run_streams_thread_first_then_astream_run_events_and_ends_done(api):
    events = run(api)
    assert events[0]["type"] == "thread" and events[0]["is_new"] is True
    assert events[-1]["type"] == "done"
    assert events[-1]["answer"] == ANSWER
    assert "".join(e["text"] for e in events if e["type"] == "token") == ANSWER
    assert [e["routed_to"] for e in events if e["type"] == "decision"] == ["researcher", "writer", "finish"]


def test_the_policy_is_built_from_the_request(api):
    events = run(api, critic=True, max_revisions=1, plan=False, max_research_iterations=3)
    assert api.policies[-1]["enable_critic"] is True
    assert api.policies[-1]["max_revisions"] == 1
    assert api.policies[-1]["max_research_iterations"] == 3
    assert events[0]["policy"] == api.policies[-1]
    assert any(e["type"] == "verdict" and e["verdict"] == "approve" for e in events)


def test_nothing_private_leaks_through_the_stream_or_the_thread_view(api):
    events = run(api, critic=True, max_revisions=0)
    tid = events[0]["thread_id"]
    text = json.dumps(events) + api.get(f"/api/threads/{tid}").text
    for private in ("research_messages", "critic_messages", "RAW-SEARCH-RESULT-TEXT", "RAW-LOOKUP-TEXT"):
        assert private not in text, private


def test_a_second_turn_continues_the_thread(api):
    tid = run(api)[0]["thread_id"]
    events = run(api, thread_id=tid)
    assert events[0] == {"type": "thread", "thread_id": tid, "is_new": False, "policy": api.policies[-1]}
    view = api.get(f"/api/threads/{tid}").json()
    assert [m["role"] for m in view["messages"]] == ["user", "assistant", "user", "assistant"]


def test_request_validation(api):
    assert api.post("/api/runs", json={"question": ""}).status_code == 422
    assert api.post("/api/runs", json={"question": "q", "mode": "web"}).status_code == 422
    assert api.post("/api/runs", json={"question": "q", "max_revisions": 9}).status_code == 422


# --- the review gate ----------------------------------------------------------------------


def test_approve_parks_the_run_and_the_thread_view_shows_the_draft(api):
    events = run(api, approve=True)
    assert events[-1]["type"] == "paused"
    assert events[-1]["payload"]["draft"] == ANSWER
    tid = events[0]["thread_id"]
    view = api.get(f"/api/threads/{tid}").json()
    assert view["pending_review"]["draft"] == ANSWER
    assert view["messages"] == [{"role": "user", "content": "How are RAG pipelines evaluated?"}]


def test_a_new_turn_on_a_parked_thread_is_refused(api):
    tid = run(api, approve=True)[0]["thread_id"]
    r = api.post("/api/runs", json={"question": "next", "thread_id": tid})
    assert r.status_code == 409 and "awaiting approval" in r.json()["detail"]


def test_review_approve_finishes_under_the_stored_policy(api):
    tid = run(api, approve=True, critic=True, max_revisions=0)[0]["thread_id"]
    stored = api.policies[-1]
    events = events_of(api.post(f"/api/threads/{tid}/review", json={"decision": "approve"}))
    assert events[-1]["type"] == "done" and events[-1]["answer"] == ANSWER
    assert api.policies[-1] == stored  # rebuilt from run_policy, not from the request
    assert api.get(f"/api/threads/{tid}").json()["pending_review"] is None


def test_review_edit_delivers_the_edited_text(api):
    tid = run(api, approve=True)[0]["thread_id"]
    events = events_of(api.post(f"/api/threads/{tid}/review", json={"decision": "edit", "text": "EDITED", "note": "n"}))
    assert events[-1]["type"] == "done" and events[-1]["answer"] == "EDITED"


def test_review_refusals(api):
    assert api.post("/api/threads/nope/review", json={"decision": "approve"}).status_code == 404
    tid = run(api)[0]["thread_id"]
    assert api.post(f"/api/threads/{tid}/review", json={"decision": "approve"}).status_code == 409
    parked = run(api, approve=True)[0]["thread_id"]
    assert api.post(f"/api/threads/{parked}/review", json={"decision": "edit", "text": " "}).status_code == 400


# --- threads ------------------------------------------------------------------------------


def test_threads_lists_multi_agent_threads_newest_first(api):
    a = run(api, question="first")[0]["thread_id"]
    b = run(api, question="second")[0]["thread_id"]
    listed = api.get("/api/threads").json()
    assert [t["thread_id"] for t in listed] == [b, a]
    assert listed[0]["question"] == "second" and listed[0]["turns"] == 1


def test_a_thread_is_listed_under_its_first_question(api):
    tid = run(api, question="first")[0]["thread_id"]
    run(api, question="follow-up", thread_id=tid)
    (listed,) = [t for t in api.get("/api/threads").json() if t["thread_id"] == tid]
    assert (listed["question"], listed["turns"]) == ("first", 2)


def test_delete_forgets_the_thread_including_subgraph_checkpoints(api):
    tid = run(api)[0]["thread_id"]
    assert api.delete(f"/api/threads/{tid}").status_code == 204
    assert api.get(f"/api/threads/{tid}").status_code == 404
    assert tid not in [t["thread_id"] for t in api.get("/api/threads").json()]


def test_unknown_thread_is_404(api):
    assert api.get("/api/threads/missing").status_code == 404


def test_health_reports_provider_and_checkpointer(api):
    body = api.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["checkpointer"] in ("InMemorySaver", "MemorySaver")
    assert body["provider"] in ("anthropic", "groq")
    assert isinstance(body["api_key_set"], bool)


def test_sqlite_checkpointer_is_async_and_shared_with_the_cli(tmp_path, monkeypatch):
    """The server opens AsyncSqliteSaver on the CLI's database file, so a
    thread written by one is visible to the other."""
    monkeypatch.setenv("RESEARCH_COPILOT_CHECKPOINTER", "sqlite")
    monkeypatch.setenv("RESEARCH_COPILOT_CHECKPOINT_DB", str(tmp_path / "cp.sqlite3"))
    app = create_app(graph_factory=fake_factory([]), model_check=None)
    with TestClient(app) as client:
        assert client.get("/api/health").json()["checkpointer"] == "AsyncSqliteSaver"
        tid = run(client)[0]["thread_id"]
    from research_copilot.checkpointing import checkpointer_scope, list_thread_ids

    with checkpointer_scope("sqlite", db_path=tmp_path / "cp.sqlite3") as saver:
        assert tid in list_thread_ids(saver)


# --- configuration failures ---------------------------------------------------------------


def test_a_failed_model_check_is_a_stopped_event_not_a_500():
    def bad():
        raise RuntimeError("GROQ_API_KEY is not set.")

    app = create_app(saver=MemorySaver(), graph_factory=fake_factory([]), model_check=bad)
    with TestClient(app) as client:
        events = run(client)
    assert [e["type"] for e in events] == ["thread", "stopped"]
    assert events[-1]["reason"] == "not_configured"
