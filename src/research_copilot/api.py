"""The HTTP API: the multi-agent graph behind FastAPI. (Phase 7, Part F)

    GET    /api/health                    provider, model, checkpointer, whether a key is set
    GET    /api/threads                   multi-agent threads, newest first, with a one-line summary
    GET    /api/threads/{id}              one thread: transcript, last turn's work, a pending review
    DELETE /api/threads/{id}              forget a thread - checkpoints, private channels and all
    POST   /api/runs                      start a turn (new or existing thread), streamed as SSE
    POST   /api/threads/{id}/review       resolve a parked draft, streamed as SSE

Run it:  uvicorn research_copilot.api:app --reload      (pip install -e ".[api]")

CONCEPT: the API streams `astream_run`'s events and nothing else
Part C built `streaming.astream_run` for this endpoint: an allow-list of
normalised events, because raw `astream_events` carries the Researcher's
private search results. The API adds exactly one event of its own, `thread`,
sent first, so a client learns the thread_id before the run produces anything.
Every other event is `astream_run`'s, byte for byte. A second serializer here
would be a second place for a private field to slip through.

CONCEPT: the same refusals as the CLI, as status codes
The CLI's 6.4 checks become 4xx responses before any stream starts:
  409  a new turn on a thread parked at the review gate (the parked draft would
       be silently abandoned, and the next model call sees two human turns)
  409  a second run on a thread that already has one in flight (both would
       write the same checkpoints)
  409  a review on a thread with nothing awaiting approval
  422  a thread written by the single-agent graph (get_state would lie: it is a
       view through *this* graph's schema - see cli._raw_channels)
  404  a thread with no saved state
Once the stream has started, failures are events (`stopped`, `error`), since
an HTTP status has already been sent.

CONCEPT: the checkpointer is the process's, opened once
The CLI opens a saver per command. A server opens one at startup and shares it
across requests, so it has to be an *async* saver: `astream_events` calls the
checkpointer's async methods, which the sync SqliteSaver does not implement.
`sqlite` here means AsyncSqliteSaver on the same file the CLI uses, so a thread
started in a terminal can be reviewed from the browser and vice versa.
`none` is upgraded to `memory`: threads and the review gate are the API's
whole point, and both need somewhere to park.

CONCEPT: the policy is stored, as in multi-review
A review rebuilds the graph from the thread's own `run_policy` (6.4), never
from the request. The browser sends only the decision. That is what makes a
review from a different tab, or the CLI, resume under the policy the pause was
started under.

Retention (an open item from Phase 6): private channels are persisted under
subgraph namespaces, and `delete_thread` is the only thing that removes them.
DELETE /api/threads/{id} is that story for the API.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.runnables import Runnable
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from pydantic import BaseModel, Field

from research_copilot.checkpointing import new_thread_id, resolve_checkpoint_db, thread_config
from research_copilot.config import get_settings
from research_copilot.multi_agent_graph import build_multi_agent_graph, multi_agent_turn_input, run_policy
from research_copilot.streaming import astream_run

log = logging.getLogger("research_copilot.api")

GraphFactory = Callable[[dict, BaseCheckpointSaver], Runnable]
ModelCheck = Callable[[], None]


# --------------------------------------------------------------------- requests


class RunRequest(BaseModel):
    """One turn. The fields are the CLI's `multi-agent` flags, with its defaults."""

    question: str = Field(min_length=1, max_length=4000)
    mode: Literal["live-search", "knowledge-base"] = "live-search"
    thread_id: str | None = None
    routing: Literal["supervisor", "fixed"] = "supervisor"
    critic: bool = False
    approve: bool = False
    plan: bool = False
    max_revisions: int | None = Field(default=None, ge=0, le=5)
    max_research_iterations: int = Field(default=5, ge=1, le=10)


class ReviewRequest(BaseModel):
    decision: Literal["approve", "reject", "edit"]
    text: str = ""
    note: str = ""


# ---------------------------------------------------------------------- helpers


def default_graph_factory(policy: dict, saver: BaseCheckpointSaver) -> Runnable:
    return build_multi_agent_graph(**policy, checkpointer=saver)


def default_model_check() -> None:
    from research_copilot.models import check_model_available

    check_model_available()


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, default=str)}\n\n"


def _is_multi_agent(values: dict) -> bool:
    # Same test as cli._is_multi_agent_thread, on raw channel values.
    return "run_policy" in values or "supervisor_log" in values


async def _raw_channels(saver: BaseCheckpointSaver, thread_id: str) -> dict:
    found = await saver.aget_tuple(thread_config(thread_id))
    return dict(found.checkpoint.get("channel_values", {})) if found else {}


def _transcript(values: dict) -> list[dict]:
    """Human and AI turns only. Tool traffic never reaches the shared transcript
    (Phase 6's invariant); filtering here keeps it out even if it ever did."""
    out = []
    for m in values.get("messages") or []:
        if m.type in ("human", "ai") and not getattr(m, "tool_calls", None) and m.text:
            out.append({"role": "user" if m.type == "human" else "assistant", "content": m.text})
    return out


def _thread_view(thread_id: str, values: dict, interrupts) -> dict:
    """An allow-list of shared state, as `_done` is for the stream. Private
    channels are not in the parent state at all; nothing here reaches for them."""
    return {
        "thread_id": thread_id,
        "question": values.get("question", ""),
        "mode": values.get("mode", ""),
        "policy": values.get("run_policy") or {},
        "messages": _transcript(values),
        "last_turn": {
            "sub_questions": list(values.get("sub_questions") or []),
            "research_notes": values.get("research_notes", ""),
            "research_outcome": values.get("research_outcome", ""),
            "draft": values.get("draft", ""),
            "verdict": values.get("verdict", ""),
            "critique": values.get("critique", ""),
            "citation_checks": list(values.get("citation_checks") or []),
            "unsupported_citations": list(values.get("unsupported_citations") or []),
            "revisions": values.get("revisions", 0),
            "dispatches": dict(values.get("dispatches") or {}),
            "supervisor_log": list(values.get("supervisor_log") or []),
            "human_verdict": values.get("human_verdict", ""),
        },
        "pending_review": interrupts[0].value if interrupts else None,
    }


# -------------------------------------------------------------------------- app


def create_app(
    *,
    saver: BaseCheckpointSaver | None = None,
    graph_factory: GraphFactory = default_graph_factory,
    model_check: ModelCheck | None = default_model_check,
    cors_origins: list[str] | None = None,
) -> FastAPI:
    """Build the app. Tests pass a MemorySaver, a factory that injects fake
    models, and `model_check=None`; production passes nothing."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if saver is not None:
            app.state.saver = saver
            yield
            return
        kind = get_settings().checkpointer
        if kind == "sqlite":
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

            async with AsyncSqliteSaver.from_conn_string(str(resolve_checkpoint_db())) as s:
                await s.setup()
                app.state.saver = s
                yield
        else:
            app.state.saver = MemorySaver()
            yield

    app = FastAPI(title="Research Copilot API", version="0.1.0", lifespan=lifespan)
    origins = cors_origins if cors_origins is not None else [
        o.strip() for o in os.getenv("RESEARCH_COPILOT_CORS_ORIGINS", "*").split(",") if o.strip()
    ]
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])

    busy: set[str] = set()
    model_ok = {"checked": model_check is None}

    async def ensure_model() -> dict | None:
        """Part D row F (fail fast), once per process. A failure is returned as
        the stream's `stopped` event, exactly as astream_run would report it."""
        if model_ok["checked"]:
            return None
        from research_copilot.evaluation.record import ending_for

        try:
            await asyncio.to_thread(model_check)
        except Exception as exc:  # reported to the client, then logged
            log.warning("model check failed: %s", exc)
            ending = ending_for(exc)
            if ending["type"] == "error":
                ending = {"type": "stopped", "reason": "not_configured", "detail": str(exc)[:300]}
            return ending
        model_ok["checked"] = True
        return None

    def stream(thread_id: str, events: AsyncIterator[dict]) -> StreamingResponse:
        async def body():
            try:
                async for e in events:
                    yield _sse(e)
            except Exception:
                # astream_run already yielded an `error` event and re-raised;
                # the bug stays loud in the server log, the client has its event.
                log.exception("run on thread %s failed", thread_id)
            finally:
                busy.discard(thread_id)

        return StreamingResponse(body(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    def claim(thread_id: str) -> None:
        if thread_id in busy:
            raise HTTPException(409, f"thread {thread_id} already has a run in progress")
        busy.add(thread_id)

    # ---------------------------------------------------------------- routes

    @app.get("/api/health")
    async def health() -> dict:
        s = get_settings()
        key = "GROQ_API_KEY" if s.provider == "groq" else "ANTHROPIC_API_KEY"
        return {
            "status": "ok",
            "provider": s.provider,
            "model": s.groq_model if s.provider == "groq" else s.model,
            "api_key_set": bool(os.getenv(key)),
            "checkpointer": type(app.state.saver).__name__,
            "defaults": {"max_revisions": s.max_revisions, "critic": s.enable_critic, "plan": s.enable_planning},
        }

    @app.get("/api/threads")
    async def threads(limit: int = 50) -> list[dict]:
        sv: BaseCheckpointSaver = app.state.saver
        seen: dict[str, dict] = {}
        async for cp in sv.alist(None):
            cfg = cp.config.get("configurable", {})
            tid = cfg.get("thread_id")
            if tid is None or tid in seen or cfg.get("checkpoint_ns"):
                continue
            values = cp.checkpoint.get("channel_values", {})
            if not _is_multi_agent(values):
                seen[tid] = {}  # remembered so an older checkpoint can't re-add it
                continue
            humans = [m for m in values.get("messages") or [] if m.type == "human"]
            # Titled by its first question, which names the conversation; the
            # `question` channel holds only the latest turn's. (Pruning may
            # have dropped the first message; then the oldest survivor names it.)
            seen[tid] = {"thread_id": tid,
                         "question": humans[0].text if humans else values.get("question", ""),
                         "turns": len(humans), "updated_at": cp.checkpoint.get("ts")}
        found = [t for t in seen.values() if t]
        # MemorySaver lists thread by thread, not newest first; sort explicitly.
        found.sort(key=lambda t: t["updated_at"] or "", reverse=True)
        return found[:limit]

    @app.get("/api/threads/{thread_id}")
    async def thread(thread_id: str) -> dict:
        sv = app.state.saver
        raw = await _raw_channels(sv, thread_id)
        if not raw:
            raise HTTPException(404, f"thread {thread_id} has no saved state")
        if not _is_multi_agent(raw):
            raise HTTPException(422, f"thread {thread_id} was written by the single-agent graph")
        snapshot = await build_multi_agent_graph(checkpointer=sv).aget_state(thread_config(thread_id))
        return _thread_view(thread_id, snapshot.values, snapshot.interrupts)

    @app.delete("/api/threads/{thread_id}", status_code=204)
    async def delete(thread_id: str) -> None:
        if thread_id in busy:
            raise HTTPException(409, f"thread {thread_id} has a run in progress")
        await app.state.saver.adelete_thread(thread_id)

    @app.post("/api/runs")
    async def start_run(req: RunRequest) -> StreamingResponse:
        sv = app.state.saver
        s = get_settings()
        if req.mode == "knowledge-base":
            from research_copilot.retrieval import count_chunks, get_vector_store

            if await asyncio.to_thread(lambda: count_chunks(get_vector_store())) == 0:
                raise HTTPException(400, "the knowledge base is empty: run `research-copilot ingest <path>` first")

        is_new = req.thread_id is None
        tid = req.thread_id or new_thread_id()
        if not is_new:
            raw = await _raw_channels(sv, tid)
            if raw and not _is_multi_agent(raw):
                raise HTTPException(422, f"thread {tid} was written by the single-agent graph")
            if raw:
                snap = await build_multi_agent_graph(checkpointer=sv).aget_state(thread_config(tid))
                if snap.interrupts:
                    raise HTTPException(409, f"thread {tid} has a draft awaiting approval; review it first")

        policy = run_policy(
            routing=req.routing, enable_critic=req.critic, require_approval=req.approve,
            max_revisions=s.max_revisions if req.max_revisions is None else req.max_revisions,
            max_research_iterations=req.max_research_iterations, enable_planning=req.plan,
        )
        claim(tid)

        async def events():
            yield {"type": "thread", "thread_id": tid, "is_new": is_new, "policy": policy}
            stopped = await ensure_model()
            if stopped:
                yield stopped
                return
            graph = graph_factory(policy, sv)
            async for e in astream_run(graph, multi_agent_turn_input(req.question, req.mode, policy=policy),
                                       thread_config(tid)):
                yield e

        return stream(tid, events())

    @app.post("/api/threads/{thread_id}/review")
    async def review(thread_id: str, req: ReviewRequest) -> StreamingResponse:
        sv = app.state.saver
        raw = await _raw_channels(sv, thread_id)
        if not raw:
            raise HTTPException(404, f"thread {thread_id} has no saved state")
        if not _is_multi_agent(raw):
            raise HTTPException(422, f"thread {thread_id} was written by the single-agent graph")
        snap = await build_multi_agent_graph(checkpointer=sv).aget_state(thread_config(thread_id))
        policy = snap.values.get("run_policy")
        if not policy:
            raise HTTPException(422, f"thread {thread_id} predates 6.4 and has no stored run policy")
        if not snap.interrupts:
            raise HTTPException(409, f"thread {thread_id} has nothing awaiting approval")
        if req.decision == "edit" and not req.text.strip():
            raise HTTPException(400, "an edit needs the edited text")

        verdict: dict[str, Any] = {"decision": req.decision, "note": req.note}
        if req.decision == "edit":
            verdict["text"] = req.text
        claim(thread_id)

        async def events():
            yield {"type": "thread", "thread_id": thread_id, "is_new": False, "policy": policy}
            graph = graph_factory(policy, sv)
            async for e in astream_run(graph, Command(resume=verdict), thread_config(thread_id)):
                yield e

        return stream(thread_id, events())

    return app


def _app_from_env() -> FastAPI:
    from dotenv import load_dotenv

    load_dotenv()
    return create_app()


# `uvicorn research_copilot.api:app`
app = _app_from_env()
