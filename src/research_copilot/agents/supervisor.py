"""The Supervisor: decide which agent acts next, and say why.

6.1's graph had a fixed order: research, then write, then finish. 6.2 replaces
the edges between agents with a hub. Every agent reports back to the Supervisor,
and the Supervisor decides where the run goes next:

                       ┌──────────────→ researcher ──┐
    plan_question ─→ supervisor ←───────────────────┘
                       │  ↑
                       │  └──────── writer ←──┐
                       ├──────────────────────┘
                       └─→ finalize_answer ─→ END

--------------------------------------------------------------------------
CONCEPT: why routing becomes a model's judgement here
--------------------------------------------------------------------------
Every routing function until now has been a lookup. `route_by_mode` reads
`mode`, `should_continue` reads `tool_calls`, `should_revise` reads `status`.
Each branches on a value that some earlier step wrote down.

"Are these notes good enough to write from?" is not a value anyone wrote down.
To answer it you have to read the notes and the question side by side and form
a view. So does "this draft skips the second half of the question, send it
back to research". That is a judgement about text, and judging text is the one
job in this system that needs a model. So the Supervisor is a model call, and
its output is a routing decision.

--------------------------------------------------------------------------
CONCEPT: structured output, and why routing needs it
--------------------------------------------------------------------------
Phase 5's critic replied in free text, with "APPROVE"/"REJECT" on the first
line, and a parser read it. That was acceptable there, because the verdict had
two values and every parse failure failed closed into a rejection.

A routing decision is less forgiving. It has several values, and the wrong
value sends the run somewhere else entirely. It also carries a field the next
agent will read (the brief). So the Supervisor's reply is constrained by a
schema, `SupervisorDecision` below, through
`model.with_structured_output(SupervisorDecision, method="json_schema")`.

What that buys you:
  - `next` is a Literal. A reply naming an agent that does not exist fails
    validation, which is a far clearer failure than being routed to a node
    that does not exist.
  - with `method="json_schema"`, the Anthropic API *constrains generation* to
    the schema (`output_config.format`), rather than asking nicely and parsing
    afterwards. The alternative, `method="function_calling"`, forces a tool
    call instead. The Anthropic API rejects a forced tool call when thinking is
    on, and some newer models reject forced tool calls outright. Native
    structured output has neither problem.
  - the result is a validated Pydantic object, not a string to parse.

What it does *not* buy you: a *sensible* decision. The schema guarantees the
shape of the answer, not its wisdom. That is why the guards below exist.

--------------------------------------------------------------------------
CONCEPT: the rationale, and why it comes first
--------------------------------------------------------------------------
`rationale` is declared before `next` in the schema, on purpose. With
constrained decoding, fields are generated in schema order. Rationale-first
means the model writes its reasons and *then* picks a route, so the route is
conditioned on the reasoning. With the order reversed, the model would pick
first and the rationale would be a justification written afterwards. That is
exactly the "rationale doesn't match the decision" failure, built in by
design.

The rationale is also for you. It goes into `supervisor_log`, next to where the
run actually went. It is the only record of *why* a route was taken, and
reading it beside `routed_to` is how you debug the Supervisor.

--------------------------------------------------------------------------
CONCEPT: code guards overrule the model (the `should_revise` rule, again)
--------------------------------------------------------------------------
Phase 5 made the point that a loop must not put its own termination condition
inside the thing being judged. The Supervisor is that case exactly. A hub where
a model decides whether to go round again is a loop whose exit is a model's
opinion. So every proposed route passes through `apply_guards`, which is plain
code and has the last word:

  1. a capped agent is never dispatched. Each agent has a per-turn dispatch
     cap, and the Supervisor's own `dispatches` count is checked against it.
  2. no writing before any research. A "writer" proposed before the Researcher
     has run once is sent to the Researcher instead.
  3. no finishing without a current draft. "finish" with no draft, or with a
     draft written *before* the latest research changed the notes, is sent to
     the Writer (budget permitting).

Every override is recorded, with its reason, beside what the model proposed.
Guard 3 is also where 6.1's "stale draft" question landed. It is answered from
the Supervisor's own log, the order in which agents were dispatched, so no new
field was needed.

These guards give a termination bound that does not depend on the model at
all. Every route except "finish" increments a capped count, and a guard never
routes to an agent that is capped. So a turn makes at most
`sum(caps) + 1` Supervisor decisions, whatever the model says.

--------------------------------------------------------------------------
CONCEPT: failing towards the known-good hand-off
--------------------------------------------------------------------------
If the model call raises, or its output fails validation, the Supervisor does
not guess. It falls back to `fixed_policy`, which is 6.1's hand-off rewritten as
a routing rule (research if nobody has, write if there is no current draft,
otherwise finish). An unusable Supervisor therefore degrades the system to
exactly the behaviour 6.1 verified, and not to something new. The same function
powers `routing="fixed"`, which runs the hub with no model at all.
"""

from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from research_copilot.models import get_chat_model
from research_copilot.multi_agent_state import MultiAgentState, SupervisorLogEntry

AGENTS = ("researcher", "writer")

# Per-turn dispatch caps. 2 each means one first pass and one follow-up.
DEFAULT_DISPATCH_CAPS = {"researcher": 2, "writer": 2}

# How much of the notes and the draft the Supervisor is shown. It needs enough
# to judge coverage, not every word. The cut is marked, so the model knows it
# is looking at an excerpt and not at a suspiciously short document.
VIEW_CHARS = 4000


class SupervisorDecision(BaseModel):
    """The Supervisor's structured reply. Field order matters - see above."""

    rationale: str = Field(
        description=(
            "One to three sentences on what in the current state decides the "
            "next step: what the notes or draft cover, what they miss, and why "
            "that points to the agent you are about to choose. Write this "
            "before choosing."
        )
    )
    next: Literal["researcher", "writer", "finish"] = Field(
        description=(
            "researcher: evidence is missing or thin. writer: the evidence is "
            "sufficient and there is no current draft, or the draft needs "
            "rewriting from the notes. finish: the current draft answers the "
            "question."
        )
    )
    researcher_brief: str = Field(
        default="",
        description=(
            "Only when next is researcher: the specific thing to look for that "
            "the current notes lack, as a short search-oriented instruction. "
            "Empty otherwise."
        ),
    )


SUPERVISOR_SYSTEM_PROMPT = (
    "You coordinate a small research team and decide who acts next. You never "
    "research or write yourself.\n\n"
    "The team:\n"
    "- researcher: gathers evidence (searches, or retrieves from the user's "
    "documents) and hands over notes. Send it back with a brief when the notes "
    "miss part of the question. It merges new findings into its existing notes.\n"
    "- writer: writes the user-facing answer from the notes. It cannot search.\n"
    "- finish: commit the current draft as the answer.\n\n"
    "Guidance:\n"
    "- The normal path is researcher, then writer, then finish. Deviate only "
    "for a concrete reason you can name.\n"
    "- A follow-up research pass is worth it only if you can say what it should "
    "find. If the last pass found nothing new, another pass with the same brief "
    "will not either.\n"
    "- Each agent has a limited number of dispatches this turn, shown below. "
    "Plan within them.\n"
    "- Finish when the draft answers the question well enough. Perfect is not "
    "required."
)


def _excerpt(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return "(empty)"
    if len(text) <= VIEW_CHARS:
        return text
    return text[:VIEW_CHARS] + f"\n[... truncated; {len(text) - VIEW_CHARS} more characters]"


def last_dispatched(state: MultiAgentState) -> str:
    """The last agent the Supervisor sent the run to this turn, or ""."""
    for entry in reversed(state.get("supervisor_log") or []):
        if entry["routed_to"] in AGENTS:
            return entry["routed_to"]
    return ""


def draft_is_current(state: MultiAgentState) -> bool:
    """Is there a draft, and was it written from the notes as they stand now?

    Answered from the Supervisor's own log, with no extra field. If the last
    agent dispatched was the Researcher and its pass changed the notes, any
    draft predates those notes. `budget_exhausted` counts as changed, which is
    conservative: it may have appended raw results.
    """
    if not (state.get("draft") or "").strip():
        return False
    if last_dispatched(state) == "researcher" and state.get("research_outcome") in (
        "findings",
        "budget_exhausted",
    ):
        return False
    return True


def render_supervisor_view(state: MultiAgentState, caps: dict[str, int]) -> str:
    """Everything the Supervisor sees, as one message. It is a pure function,
    so tests can assert on exactly what the model is shown.

    CONCEPT: what information the routing decision is made from.
      question, plan      what "done" means
      research notes      what evidence exists, excerpted
      research outcome    how the *latest* pass went, as a code-level fact
                          rather than something to infer from the notes
      draft + currency    what the Writer produced, and whether it predates
                          the latest notes
      dispatch counts     what budget is left, so the model can plan within it
      decisions so far    its own earlier routes and reasons, so it does not
                          send the Researcher off on the same brief twice

    What it does *not* see: the Researcher's private tool loop (it is private,
    see agents/researcher.py) and the shared transcript's earlier turns. The
    Supervisor routes *this* turn, and earlier turns matter only through the
    question the user asked.
    """
    dispatches = state.get("dispatches") or {}
    budget = "\n".join(
        f"- {agent}: {dispatches.get(agent, 0)} of {caps[agent]} used"
        for agent in AGENTS
    )
    sub_questions = state.get("sub_questions") or []
    plan = "\n".join(f"- {q}" for q in sub_questions) or "(none)"
    outcome = state.get("research_outcome") or "(researcher has not run this turn)"

    draft = (state.get("draft") or "").strip()
    if not draft:
        draft_note = "(no draft yet)"
    elif draft_is_current(state):
        draft_note = "(written from the current notes)"
    else:
        draft_note = "(STALE: written before the latest research changed the notes)"

    log = state.get("supervisor_log") or []
    history = "\n".join(
        f"{e['step']}. -> {e['routed_to']}"
        + (f" (brief: {e['brief']})" if e.get("brief") else "")
        + (f" [overridden: {e['override']}]" if e.get("override") else "")
        + f": {e['rationale']}"
        for e in log
    ) or "(none - this is the first decision)"

    return (
        f"Question: {state.get('question', '')}\n\n"
        f"Plan (sub-questions):\n{plan}\n\n"
        f"Research notes:\n{_excerpt(state.get('research_notes', ''))}\n\n"
        f"Latest research pass: {outcome}\n\n"
        f"Draft {draft_note}:\n{_excerpt(draft)}\n\n"
        f"Dispatches this turn:\n{budget}\n\n"
        f"Your decisions so far:\n{history}\n\n"
        "Decide who acts next."
    )


def fixed_policy(state: MultiAgentState, caps: dict[str, int]) -> str:
    """6.1's fixed hand-off, as a routing rule. Used for `routing="fixed"` and
    as the fallback when the Supervisor's output is unusable."""
    dispatches = state.get("dispatches") or {}

    def available(agent: str) -> bool:
        return dispatches.get(agent, 0) < caps[agent]

    if dispatches.get("researcher", 0) == 0 and available("researcher"):
        return "researcher"
    if not draft_is_current(state) and available("writer"):
        return "writer"
    return "finish"


def apply_guards(
    proposed: str, state: MultiAgentState, caps: dict[str, int]
) -> tuple[str, str]:
    """Return (where the run goes, why the proposal was overridden or "").

    Pure and deterministic, so each rule can be tested on its own. The rules
    are explained at the top of this module.
    """
    dispatches = state.get("dispatches") or {}

    def available(agent: str) -> bool:
        return dispatches.get(agent, 0) < caps[agent]

    if proposed in AGENTS and not available(proposed):
        route = fixed_policy(state, caps)
        return route, f"{proposed} dispatch cap ({caps[proposed]}) reached"

    if proposed == "writer" and dispatches.get("researcher", 0) == 0 and available("researcher"):
        return "researcher", "writer proposed before any research"

    if proposed == "finish" and not draft_is_current(state) and available("writer"):
        reason = (
            "finish proposed with no draft"
            if not (state.get("draft") or "").strip()
            else "finish proposed with a draft that predates the latest research"
        )
        return "writer", reason

    return proposed, ""


def make_supervisor(
    *,
    model: BaseChatModel | None = None,
    caps: dict[str, int] | None = None,
    routing: Literal["supervisor", "fixed"] = "supervisor",
):
    """Build the Supervisor node.

    `routing="fixed"` keeps the node and the hub, and replaces the model with
    `fixed_policy`. That gives 6.1's behaviour on 6.2's graph shape. It is useful
    as a baseline, and as the only mode that runs without an API key.
    """
    caps = {**DEFAULT_DISPATCH_CAPS, **(caps or {})}
    _cache: dict = {}

    def structured():
        if "runnable" not in _cache:
            base = model or get_chat_model()
            _cache["runnable"] = base.with_structured_output(
                SupervisorDecision, method="json_schema"
            )
        return _cache["runnable"]

    def supervisor(state: MultiAgentState) -> dict:
        log = list(state.get("supervisor_log") or [])
        step = len(log) + 1
        brief = ""

        if routing == "fixed":
            proposed = fixed_policy(state, caps)
            rationale = "fixed policy (no model): research, then write, then finish"
            route, override = apply_guards(proposed, state, caps)
        else:
            try:
                decision = structured().invoke(
                    [
                        SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
                        HumanMessage(content=render_supervisor_view(state, caps)),
                    ]
                )
                # Some fakes and some providers return a dict rather than the
                # model instance. Validating either way means a bad value fails
                # here, inside the try, and not later at the edge.
                if not isinstance(decision, SupervisorDecision):
                    decision = SupervisorDecision.model_validate(decision)
            except Exception as exc:  # noqa: BLE001 - any failure takes the same path
                proposed = None
                rationale = f"(supervisor output unusable: {type(exc).__name__}: {exc})"
                route = fixed_policy(state, caps)
                override = "fallback to fixed policy"
            else:
                proposed = decision.next
                rationale = decision.rationale.strip()
                route, override = apply_guards(proposed, state, caps)
                # The brief only travels with a route the model actually chose.
                # A guard that redirects "writer" to the Researcher has no
                # brief to give, and inventing one would put words in the
                # model's mouth.
                if route == "researcher" and proposed == "researcher":
                    brief = decision.researcher_brief.strip()

        entry: SupervisorLogEntry = {
            "step": step,
            "proposed": proposed,
            "rationale": rationale,
            "routed_to": route,
            "override": override,
            "brief": brief,
        }
        update: dict = {"next_agent": route, "supervisor_log": [*log, entry]}
        if route in AGENTS:
            dispatches = dict(state.get("dispatches") or {})
            dispatches[route] = dispatches.get(route, 0) + 1
            update["dispatches"] = dispatches
        if route == "researcher":
            # Always written on a research dispatch, even when empty. A brief
            # left over from an earlier dispatch must not steer this one.
            update["researcher_brief"] = brief
        return update

    return supervisor
