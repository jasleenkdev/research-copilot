"""Run one example once, and return what an evaluator may read. (Phase 7, E2)

CONCEPT: one sample = one execution, whatever happened
`execute()` never raises for something that went wrong *inside* the run. It
returns the EvalRecord together with the exception, and the sampler decides:
a rate limit is recorded and the session stops, and a bug in our own code is
recorded and then re-raised (evaluation.samples). The record is the same shape
either way, so no failure can end a sample without a trace.

CONCEPT: two execution paths, and the one that has no stream
  whole-graph (e2e)   through `streaming.astream_run`, with a checkpointer so
                      the final state can be read back. The stream reports
                      how the run ended.
  one agent (unit)    invoked directly, as Part A did, so samples stay
                      comparable with its results. There is no stream, so
                      nothing reports the ending. `_unit()` catches the
                      exception itself and records it with the same
                      `ending_for()` the stream uses. Without that, a unit
                      run cut short by a rate limit would look like any other
                      failure, and the infrastructure/behaviour split would
                      have nothing to go on for that half of the dataset.

CONCEPT: the tool recorder is an observer, so it keeps only derived facts
Evaluators need two things the stream does not carry: whether a tool reported
its own failure (arXiv down, lookup timed out), and which arXiv ids a search
returned (to check the Researcher invented none). `ToolRecorder` sees the raw
tool output to work those out, and keeps only a status and a list of ids. The
result text itself is the private channel's content and never leaves this
module. That is tested against a canary, under the observer rule.
"""

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler, get_usage_metadata_callback
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from research_copilot.evaluation.dataset import Example
from research_copilot.evaluation.record import EvalRecord, build_eval_record, ending_for
from research_copilot.live_check.runner import CallRecorder, _SdkRetryCounter, arxiv_ids
from research_copilot.streaming import TOOL_ARGS, astream_run

# How our own tools report a failure, copied from their code (tools/arxiv.py,
# tools/citations.py) and pinned by a test against them. A failed tool is
# infrastructure (arXiv was down), not the agent's behaviour.
TOOL_FAILURE_PREFIXES = ("arXiv search failed", "ERROR: lookup failed")


class ToolRecorder(BaseCallbackHandler):
    """Every tool call: name, identifying argument, status, result ids."""

    def __init__(self, default_agent: str, *, agent_from_namespace: bool = False):
        # In a whole-graph run the namespace's first segment is the agent. In a
        # unit run the agent *is* the graph, and that segment is an inner node.
        self.default_agent = default_agent
        self.agent_from_namespace = agent_from_namespace
        self.calls: list[dict] = []
        self._by_run: dict[str, dict] = {}

    def on_tool_start(self, serialized, input_str, *, run_id, metadata=None, inputs=None, **kwargs):
        name = (serialized or {}).get("name") or kwargs.get("name") or "?"
        ns = ((metadata or {}).get("checkpoint_ns") or "") if self.agent_from_namespace else ""
        call = {"agent": ns.split(":")[0] or self.default_agent, "name": name, "status": "started"}
        arg = TOOL_ARGS.get(name)
        if arg and isinstance(inputs, dict) and arg in inputs:
            call["arg"] = str(inputs[arg])
        self.calls.append(call)
        self._by_run[str(run_id)] = call

    def on_tool_end(self, output, *, run_id, **kwargs):
        call = self._by_run.get(str(run_id))
        if call is None:
            return
        text = getattr(output, "content", output)
        text = text if isinstance(text, str) else str(text)
        call["status"] = "failed" if text.startswith(TOOL_FAILURE_PREFIXES) else "ok"
        call["result_ids"] = sorted(arxiv_ids(text))

    def on_tool_error(self, error, *, run_id, **kwargs):
        call = self._by_run.get(str(run_id))
        if call is not None:
            call["status"] = "error"


@dataclass
class ExecConfig:
    model_factory: Callable | None = None   # () -> chat model; default get_chat_model
    search_tool: Any = None                  # default: the real search_arxiv
    verify_tool: Any = None                  # default: the real verify_citation
    extra: dict = field(default_factory=dict)


def _model(cfg: ExecConfig):
    from research_copilot.models import get_chat_model

    return (cfg.model_factory or get_chat_model)()


# --- the unit runs: each returns a state-like mapping for build_eval_record ---------------------


def _probe_answer(e: Example, cfg):
    from research_copilot.chains import build_answer_chain

    return {"probe": {"answer": build_answer_chain(model=_model(cfg)).invoke({"question": e.inputs["question"]})}}


def _probe_structured(e: Example, cfg):
    from langchain_core.exceptions import OutputParserException
    from pydantic import ValidationError

    from research_copilot.chains import build_structured_chain

    try:
        return {"probe": {"parsed": build_structured_chain(model=_model(cfg)).invoke(e.inputs).model_dump()}}
    except (OutputParserException, ValidationError) as exc:
        # A parse failure is what this probe measures - a result, not a crash.
        # Only these two: anything else is not "the output did not parse".
        return {"probe": {"parsing_error": f"{type(exc).__name__}: {exc}"[:300]}}


def _structured_smoke(e: Example, cfg):
    import json

    from research_copilot.agents.supervisor import SupervisorDecision
    from research_copilot.models import structured_output_kwargs

    model = _model(cfg)
    kwargs = structured_output_kwargs(model)
    out = model.with_structured_output(SupervisorDecision, include_raw=True, **kwargs).invoke([
        SystemMessage(content="Decide who acts next on a research team."),
        HumanMessage(content="Nothing has been researched yet. Decide who acts next."),
    ])
    raw, parsed = out.get("raw"), out.get("parsed")
    order = [k for call in getattr(raw, "tool_calls", None) or [] for k in call.get("args", {})]
    if not order and isinstance(getattr(raw, "content", None), str):
        try:
            order = list(json.loads(raw.content).keys())
        except ValueError:
            pass
    return {"probe": {"method": kwargs["method"], "strict": kwargs.get("strict"),
                      "parsed": parsed.model_dump() if parsed else None,
                      "parsing_error": str(out.get("parsing_error") or ""), "field_order": order}}


def _supervisor(e: Example, cfg):
    from research_copilot.agents.supervisor import make_supervisor

    return make_supervisor(model=_model(cfg), enable_critic=True, max_revisions=2)(e.inputs["state"])


def _critic(e: Example, cfg):
    from research_copilot.agents.critic import build_critic
    from research_copilot.tools.citations import verify_citation

    if e.inputs.get("lookup") == "error":
        @tool("verify_citation")
        def verify_citation_down(arxiv_id: str) -> str:
            """Check that an arXiv paper with this id exists, and get its title."""
            return "ERROR: lookup failed (network error: timed out). This says nothing about the citation."

        tools = [verify_citation_down]
    else:
        tools = [cfg.verify_tool or verify_citation]
    return build_critic(model=_model(cfg), tools=tools).invoke(
        {k: e.inputs[k] for k in ("question", "draft", "research_notes", "unsupported_citations")})


def _researcher(e: Example, cfg):
    from research_copilot.agents.researcher import build_researcher
    from research_copilot.tools import search_arxiv

    return build_researcher(model=_model(cfg), tools=[cfg.search_tool or search_arxiv]).invoke({
        "question": e.inputs["question"], "mode": "live-search",
        "messages": [HumanMessage(content=e.inputs["question"])],
        "research_notes": e.inputs.get("research_notes", ""),
        "researcher_brief": e.inputs.get("brief", ""),
    })


def _writer(e: Example, cfg):
    from research_copilot.agents.writer import make_writer

    state = {"question": e.inputs["question"], "mode": e.inputs.get("mode", "live-search"),
             "messages": [HumanMessage(content=e.inputs["question"])],
             **{k: v for k, v in e.inputs.items() if k not in ("question", "mode")}}
    return make_writer(model=_model(cfg), max_revisions=2)(state)


UNIT_RUNNERS = {
    "probe_answer": _probe_answer, "probe_structured": _probe_structured,
    "structured_smoke": _structured_smoke, "supervisor": _supervisor, "critic": _critic,
    "researcher": _researcher, "writer": _writer,
}

# The agent a unit run's tool calls belong to (no subgraph namespace to say so).
_UNIT_AGENT = {"critic": "critic", "researcher": "researcher"}


def _unit(e: Example, cfg, callbacks) -> tuple[dict, list[dict], dict | None, BaseException | None]:
    """A single agent, invoked directly. The explicit error path: there is no
    stream to report the ending, so it is recorded here."""
    from langchain_core.runnables import RunnableLambda

    runner = UNIT_RUNNERS[e.kind]
    try:
        # A RunnableLambda so the callbacks (tool and call recorders) reach
        # every model and tool call the agent makes.
        state = RunnableLambda(lambda _: runner(e, cfg)).invoke(None, {"callbacks": callbacks})
        return dict(state or {}), [], None, None
    except Exception as exc:  # noqa: BLE001 - recorded here, re-raised by the sampler if it is ours
        return {}, [], ending_for(exc), exc


def _e2e(e: Example, cfg, callbacks) -> tuple[dict, list[dict], dict | None, BaseException | None]:
    from langgraph.checkpoint.memory import MemorySaver

    from research_copilot.multi_agent_graph import build_multi_agent_graph, multi_agent_turn_input

    graph = build_multi_agent_graph(
        model=cfg.model_factory() if cfg.model_factory else None,
        tools=[cfg.search_tool] if cfg.search_tool else None,
        critic_tools=[cfg.verify_tool] if cfg.verify_tool else None,
        enable_critic=e.inputs["critic"], max_revisions=1, memory_strategy="none",
        checkpointer=MemorySaver(),
    )
    config = {"configurable": {"thread_id": f"eval-{e.id}"}, "recursion_limit": 80, "callbacks": callbacks}
    events: list[dict] = []

    async def run():
        async for event in astream_run(graph, multi_agent_turn_input(e.inputs["question"]), config):
            events.append(event)

    error = None
    try:
        asyncio.run(run())
    except Exception as exc:  # noqa: BLE001 - astream_run has already yielded the error event
        error = exc
    # The state so far, even after a failure: dispatches, the log, the notes.
    state = dict(graph.get_state({"configurable": {"thread_id": f"eval-{e.id}"}}).values or {})
    return state, events, (ending_for(error) if error else None), error


def execute(e: Example, cfg: ExecConfig | None = None) -> tuple[EvalRecord, BaseException | None]:
    """One sample of `e`: its EvalRecord, and the exception if the run died."""
    cfg = cfg or ExecConfig()
    tools = ToolRecorder(default_agent=_UNIT_AGENT.get(e.kind, e.kind), agent_from_namespace=e.kind == "e2e")
    calls = CallRecorder()
    started = time.monotonic()
    with _SdkRetryCounter() as retries, get_usage_metadata_callback() as usage:
        run = _e2e if e.kind == "e2e" else _unit
        state, events, ending, error = run(e, cfg, [tools, calls])
    tokens = sum(u.get("total_tokens", 0) for u in usage.usage_metadata.values())
    record = build_eval_record(
        state, events=events, calls=calls.calls, tool_calls=tools.calls, ending=ending,
        tokens=tokens, sdk_retries=retries.count, seconds=round(time.monotonic() - started, 1),
    )
    return record, error
