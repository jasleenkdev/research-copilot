"""Watching a multi-agent run as it happens. (Phase 7, Part C)

`astream_run(graph, graph_input, config)` is an async generator of small,
normalised progress events:

    node          an agent (or one of its inner steps) started or finished
    decision      the Supervisor routed: proposed, routed_to, override, rationale
    verdict       the Critic ruled: approve / reject / incomplete, and why
    intervention  a trim, retry or fallback - the moment it happens
    token         a piece of the Writer's answer as it is generated
    tool          a tool call started or finished (name and the one argument
                  that identifies it: the search query, the arXiv id)
    paused        the run is parked at the human gate (payload for the reviewer)
    done          the run finished (the answer, and the turn's counters)
    stopped       the run was stopped on purpose: spent quota, missing model
    error         anything else went wrong (then re-raised - see below)

CONCEPT: astream_events, and why it is wrapped rather than passed through
LangChain's `astream_events(version="v2")` reports everything that happens
inside a run: every chain, node, model call, token and tool, with metadata
saying which graph node it belongs to (`langgraph_node`) and, inside a
subgraph, which agent (`checkpoint_ns`). It is the richest stream LangGraph
offers - richer than the `updates` / `tasks` / `values` modes the CLI trace
has used since 6.1.

That richness is exactly why a caller must not forward it raw. Measured in
Part C: the Researcher's private `research_messages` - every tool call and raw
search result, the thing Phase 6 built a private channel to keep from the
other agents - appears in these events, both in the inner steps' outputs and
in the parent-level `researcher` node's end event. The 6.1 finding was the
opposite for `updates`/`values`, which hide private writes. Each stream mode
exposes something different, and the only safe assumption is none. So this
module builds every event from an **allow-list** of fields. Anything not
named below never leaves it. Part F's API streams these events, never the
raw ones.

CONCEPT: what a live stream must show that the final state cannot
A node's state update arrives when the node finishes. Interventions happen
*inside* it - a hinted retry, a switch to the fallback model - and would only
be visible afterwards, so the stream would show less than the final state.
They are also announced as custom events the moment they happen
(`resilience.emit_intervention`), and arrive here as `intervention` events,
in order, between the node's start and end.

CONCEPT: two kinds of ending, deliberately different
  stopped  QuotaExhausted, ModelNotAvailable: expected, provider-side, and
           the run cannot continue. The stream ends with one clean event and
           no exception, so a consumer (the CLI, Part F's API) has nothing
           extra to handle.
  error    anything else - most likely a bug in our own code. It is reported
           as an event (so a client sees *something*) and then RE-RAISED. A
           streaming layer that turned our bugs into a tidy "error" event and
           carried on would be exactly the over-broad handler the README warns
           about.

CONCEPT: nothing here that a caller has to remember
The input is the ordinary turn input. `begin_turn`, the per-turn reset, is
the graph's first node (6.4), so a stream started with bare input resets
exactly like any other run. Tested, since 6.4 found the one time that was not
true.

Tokens come from the Writer only, by default. Under astream_events every
model call streams, including the Supervisor's structured JSON and the
Critic's verdict. Those are not the answer, and interleaving them with it would
be noise. `token_agents` widens it.
"""

from collections.abc import AsyncIterator, Iterable

from langchain_core.runnables import Runnable

from research_copilot.models import ModelNotAvailable
from research_copilot.resilience import INTERVENTION_EVENT, QuotaExhausted, classify

# The top-level agents a `node` event is reported for. The other nodes
# (begin_turn, prune_history, plan_question, start_revision, review_draft,
# finalize_answer) are reported too; this set only decides what counts as an
# agent for the `agent` field.
AGENT_NODES = frozenset({"researcher", "writer", "critic", "supervisor"})

# The one identifying argument per tool, allow-listed.
TOOL_ARGS = {"search_arxiv": "query", "verify_citation": "arxiv_id"}


def _where(ev: dict) -> tuple[str, str, int]:
    """(agent, node, depth) for an event. Top-level nodes report their own name
    as the agent. Nodes inside a subgraph report the subgraph (the first
    segment of checkpoint_ns) as the agent and themselves as the node."""
    md = ev.get("metadata") or {}
    node = md.get("langgraph_node", "")
    ns = md.get("checkpoint_ns") or ""
    agent = ns.split(":")[0] if ns else node
    depth = ns.count("|") + 1 if ns else 0
    return agent, node, depth


def _decision(update: dict) -> dict | None:
    log = (update or {}).get("supervisor_log") or []
    if not log:
        return None
    e = log[-1]
    return {"type": "decision", "step": e.get("step"), "proposed": e.get("proposed"),
            "routed_to": e.get("routed_to"), "override": e.get("override", ""),
            "rationale": e.get("rationale", ""), "brief": e.get("brief", "")}


def _done(state: dict) -> dict:
    from research_copilot.graph import final_answer

    return {
        "type": "done",
        "answer": final_answer(state),
        "dispatches": dict(state.get("dispatches") or {}),
        "revisions": state.get("revisions", 0),
        "verdict": state.get("verdict", ""),
        "unsupported_citations": list(state.get("unsupported_citations") or []),
        "interventions": sum(len(state.get(f"{a}_interventions") or []) for a in AGENT_NODES),
    }


async def astream_run(
    graph: Runnable,
    graph_input,
    config: dict | None = None,
    *,
    token_agents: Iterable[str] = ("writer",),
) -> AsyncIterator[dict]:
    """Run the graph and yield normalised progress events (see the module doc).

    `graph_input` is a turn's input or a `Command(resume=...)`, as for
    `.invoke()`. With a checkpointer, `config` names the thread, and a run that
    parks at the human gate ends with a `paused` event.
    """
    token_agents = frozenset(token_agents)
    final_state: dict | None = None
    try:
        async for ev in graph.astream_events(graph_input, config, version="v2"):
            kind = ev["event"]
            agent, node, depth = _where(ev)

            if kind == "on_chain_end" and not ev.get("parent_ids"):
                # The graph's own end event: its output is the final state.
                output = ev["data"].get("output")
                if isinstance(output, dict):
                    final_state = output
                continue

            if kind in ("on_chain_start", "on_chain_end") and ev.get("name") == node and node:
                if node == "__start__" or (depth > 0 and node == agent):
                    # LangGraph internals: a subgraph's entry step, and the
                    # subgraph's own run inside its namespace (the parent-level
                    # event for the same agent is already reported).
                    continue
                yield {"type": "node", "phase": "start" if kind == "on_chain_start" else "end",
                       "agent": agent, "node": node, "depth": depth}
                if kind == "on_chain_end" and depth == 0:
                    output = ev["data"].get("output") or {}
                    if node == "supervisor" and isinstance(output, dict):
                        decision = _decision(output)
                        if decision:
                            yield decision
                    if node == "critic" and isinstance(output, dict) and output.get("verdict"):
                        yield {"type": "verdict", "verdict": output["verdict"],
                               "critique": output.get("critique", ""),
                               "citation_checks": list(output.get("citation_checks") or [])}
                continue

            if kind == "on_custom_event" and ev.get("name") == INTERVENTION_EVENT:
                data = ev.get("data") or {}
                yield {"type": "intervention", **{k: data[k] for k in (
                    "node", "kind", "part", "tokens_before", "tokens_after", "limit", "detail") if k in data}}
                continue

            if kind == "on_chat_model_stream" and agent in token_agents:
                chunk = ev["data"].get("chunk")
                text = getattr(chunk, "text", "") if chunk is not None else ""
                if text:
                    yield {"type": "token", "agent": agent, "text": text}
                continue

            if kind in ("on_tool_start", "on_tool_end"):
                name = ev.get("name", "")
                arg = TOOL_ARGS.get(name)
                inputs = (ev["data"].get("input") or {}) if kind == "on_tool_start" else {}
                yield {"type": "tool", "phase": "start" if kind == "on_tool_start" else "end",
                       "agent": agent, "name": name,
                       **({"arg": str(inputs.get(arg, ""))} if arg and kind == "on_tool_start" else {})}
                continue
    except QuotaExhausted as exc:
        yield {"type": "stopped", "reason": "quota_exhausted", "detail": _limit_line(exc)}
        return
    except ModelNotAvailable as exc:
        yield {"type": "stopped", "reason": "model_not_available", "detail": str(exc)}
        return
    except Exception as exc:
        # Report, then re-raise: a consumer sees *something*, and our own bugs
        # stay loud (module doc: "two kinds of ending").
        yield {"type": "error", "kind": classify(exc), "detail": f"{type(exc).__name__}: {str(exc)[:300]}"}
        raise

    if config is not None and getattr(graph, "checkpointer", None) is not None:
        snapshot = await graph.aget_state(config)
        if snapshot.interrupts:
            yield {"type": "paused", "payload": snapshot.interrupts[0].value}
            return
    if final_state is not None:
        yield _done(final_state)


def _limit_line(exc: BaseException) -> str:
    from research_copilot.live_check.runner import describe_rate_limit

    return describe_rate_limit(str(exc)) or str(exc)[:300]
