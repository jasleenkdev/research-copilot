"""EvalRecord: the only thing an evaluator reads. (Phase 7, E1)

CONCEPT: the evaluator's input is an observer, so it gets the observer rule
The README's standing rule: any new way of observing the graph is assumed to
leak private-channel content until a test proves otherwise. An evaluator is
one. It reads a run's results, and whatever it reads also reaches the results
file, the report, and a LangSmith export. So an evaluator never sees the
graph's state or raw `astream_events`. It sees an EvalRecord, built here from
named fields only:

  from the final state    question, answer, draft, research notes and outcome,
                          the Supervisor's log, the Critic's verdict, critique
                          and citation checks, unsupported citations,
                          dispatches, revisions, interventions
  from astream_run        tool calls (name and one identifying argument), and
                          how the run ended
  from the runner         per-call sizes (CallRecorder), tokens, SDK retries,
                          seconds

Nested entries are allow-listed too (a supervisor_log entry, a citation
check, an intervention), because a list of dicts is where an extra field
would slip through.

CONCEPT: private, derived from the agents' own schemas
`private_keys()` reads each subgraph's state type and returns every key that
is in neither its input nor its output contract: `research_messages`,
`critic_messages`, `lookups` today, and whatever is added tomorrow. The leak
test uses it, so a new private key is covered without anyone having to
remember to add it.

CONCEPT: the ending is part of the record
E2 separates "the model got it wrong" from "the run never finished" (a spent
quota, a rate limit that outlasted the retries). That split needs the ending
recorded as data. It is taken from the stream's last event: `done`,
`stopped` (with its reason), `error` (with its classification), or `paused`.
"""

import typing
from collections.abc import Iterable, Mapping
from typing import Any, TypedDict

AGENTS = ("researcher", "writer", "critic", "supervisor")

LOG_ENTRY_KEYS = ("step", "proposed", "routed_to", "override", "rationale", "brief", "revision")
CITATION_CHECK_KEYS = ("arxiv_id", "status")
INTERVENTION_KEYS = ("node", "kind", "part", "tokens_before", "tokens_after", "limit", "detail")
TOOL_KEYS = ("agent", "name", "arg")
CALL_KEYS = ("agent", "node", "est_tokens", "status")
ENDING_KEYS = ("type", "reason", "kind", "detail")
RUNNER_KEYS = ("tokens", "sdk_retries", "seconds")

ENDINGS = frozenset({"done", "stopped", "error", "paused"})


class EvalRecord(TypedDict, total=False):
    question: str
    answer: str
    draft: str
    research_notes: str
    research_outcome: str
    research_iterations: int
    supervisor_log: list[dict]
    verdict: str
    critique: str
    citation_checks: list[dict]
    unsupported_citations: list[str]
    dispatches: dict[str, int]
    revisions: int
    interventions: list[dict]
    tools: list[dict]
    calls: list[dict]
    ending: dict
    tokens: int
    sdk_retries: int
    seconds: float


EVAL_FIELDS = frozenset(typing.get_type_hints(EvalRecord))

_STATE_TEXT = ("question", "draft", "research_notes", "research_outcome", "verdict", "critique")
_STATE_NUMBERS = ("research_iterations", "revisions")


def private_keys() -> frozenset[str]:
    """Keys that exist inside a subgraph and are in neither of its contracts."""
    from research_copilot.agents import critic, researcher

    private: set[str] = set()
    for state, contract_in, contract_out in (
        (researcher.ResearcherState, researcher.ResearcherInput, researcher.ResearcherOutput),
        (critic.CriticState, critic.CriticInput, critic.CriticOutput),
    ):
        private |= set(typing.get_type_hints(state)) - set(typing.get_type_hints(contract_in)) \
            - set(typing.get_type_hints(contract_out))
    return frozenset(private)


def _pick(entry: Mapping, keys: tuple[str, ...]) -> dict:
    return {k: entry[k] for k in keys if k in entry}


def _answer(state: Mapping) -> str:
    """The final answer on a whole-graph run, the draft on a Writer-only one."""
    from langchain_core.messages import AIMessage

    for message in reversed(state.get("messages") or []):
        if isinstance(message, AIMessage) and message.text:
            return message.text
    return state.get("draft") or ""


def _ending(events: list[dict], ending: Mapping | None) -> dict:
    if ending is not None:
        picked = _pick(ending, ENDING_KEYS)
    else:
        last = next((e for e in reversed(events) if e.get("type") in ENDINGS), None)
        # A unit run (one agent, invoked directly) has no stream; returning
        # at all means it finished.
        picked = _pick(last, ENDING_KEYS) if last else {"type": "done"}
    if picked.get("type") not in ENDINGS:
        raise ValueError(f"unknown ending: {picked!r}")
    return picked


def build_eval_record(
    state: Mapping[str, Any],
    *,
    events: Iterable[dict] = (),
    calls: Iterable[Mapping] = (),
    ending: Mapping | None = None,
    **runner: Any,
) -> EvalRecord:
    """An EvalRecord from a run's final state (or one agent's output), the
    normalised events of `streaming.astream_run`, and the runner's counters.

    Raw `astream_events` are refused: they carry the private channels (Part C).
    """
    events = list(events)
    raw = [e for e in events if "event" in e or "type" not in e]
    if raw:
        raise TypeError("build_eval_record takes streaming.astream_run events, never raw astream_events")
    unknown_runner = set(runner) - set(RUNNER_KEYS)
    if unknown_runner:
        raise TypeError(f"unknown runner fields: {sorted(unknown_runner)}")

    record: EvalRecord = {}
    for key in _STATE_TEXT + _STATE_NUMBERS:
        if state.get(key) not in (None, ""):
            record[key] = state[key]
    answer = _answer(state)
    if answer:
        record["answer"] = answer
    if state.get("supervisor_log"):
        record["supervisor_log"] = [_pick(e, LOG_ENTRY_KEYS) for e in state["supervisor_log"]]
    if state.get("citation_checks"):
        record["citation_checks"] = [_pick(c, CITATION_CHECK_KEYS) for c in state["citation_checks"]]
    if state.get("unsupported_citations"):
        record["unsupported_citations"] = [str(c) for c in state["unsupported_citations"]]
    if state.get("dispatches"):
        record["dispatches"] = {str(k): int(v) for k, v in state["dispatches"].items()}
    interventions = [_pick(i, INTERVENTION_KEYS)
                     for agent in AGENTS for i in (state.get(f"{agent}_interventions") or [])]
    if interventions:
        record["interventions"] = interventions
    tools = [_pick(e, TOOL_KEYS) for e in events if e["type"] == "tool" and e.get("phase") == "start"]
    if tools:
        record["tools"] = tools
    calls = [_pick(c, CALL_KEYS) for c in calls]
    if calls:
        record["calls"] = calls
    record["ending"] = _ending(events, ending)
    for key, value in runner.items():
        record[key] = value
    return record
