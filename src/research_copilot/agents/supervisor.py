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

--------------------------------------------------------------------------
6.3: the Critic joins, and rejections come here
--------------------------------------------------------------------------
Three changes.

CONCEPT: classifying a rejection
When the Critic (or a human) rejects a draft, `start_revision` counts the round
and the run comes back *here*. It does not go to a fixed "revise" node. The
Supervisor reads the critique and decides who fixes it:

    "cites 2401.99999, which does not exist"        evidence  -> researcher
    "claims X outperforms Y; notes don't say so"    evidence  -> researcher
    "never answers the second sub-question,
     although the notes cover it"                    writing   -> writer
    "cites a paper that is not in the notes"         writing   -> writer
                                                     (the Writer invented it)

The last two rows are why this is a judgement and not a lookup. "A citation is
wrong" can be either agent's fault, and telling which means reading the
critique against the notes. `citation_checks` gives the Supervisor the
verification results as facts, so it does not have to trust the critique's
prose about them.

The *decision to revise at all* stays in code. `after_critique` and
`after_review` in multi_agent_graph.py check `revisions` against
`max_revisions`, exactly as Phase 5's `should_revise` did. The Supervisor
decides *who*, and the cap decides *whether*.

CONCEPT: staleness, second case - a critique about an older draft
6.2 answered "is the draft stale?" from the order of dispatches in the log. The
same ordering answers "is the critique stale?". A critique describes the draft
that existed when the Critic ran. If the Writer has been dispatched since, the
critique is about a draft that has since been replaced - once, or twice, the
count does not matter. The rule: **a critique is current only if the Critic
was dispatched after the last Writer dispatch.** No `critiqued_draft` field,
no hash of the draft. The log's order already contains the answer, as it did
for the draft.

What a stale critique means depends on its verdict:
  - a stale *rejection* is normal: it is the reason the Writer was sent back,
    and the Writer reads it as its revision instruction.
  - a stale *approval* approves nothing. `finish` on a draft whose only
    approval predates it is sent back to the Critic.

Each log entry now carries its revision number, so "the Writer ran this round"
can be told apart from "the Writer ran before the last rejection". A rejected
draft is not a finishable draft, even though nothing has been dispatched since.

CONCEPT: the roster guard
"critic" is in the decision schema whether or not `--critic` is on. With
`json_schema` decoding, the model *can* propose it when it is off. It is not
offered in the prompt, so it rarely will. The guard refuses it and logs
`proposed: critic` next to `override: critic is not on the roster`.

The alternative, a per-run schema whose Literal omits the Critic, would make
the proposal impossible, and therefore invisible. The log could never show that
the model wanted a reviewer, which is a thing worth knowing about a
Supervisor. 6.2's version had exactly that blind spot. An invalid "critic"
from a fake failed validation and was logged as `proposed: None`. 6.3 also
keeps the attempted route when validation fails, where it can be recovered.
"""

from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from research_copilot.models import get_chat_model, structured_output_method
from research_copilot.multi_agent_state import (
    AGENTS,
    MultiAgentState,
    SupervisorLogEntry,
    budget_of,
)

# Per-turn dispatch caps (see the scope note on `dispatches` in
# multi_agent_state.py). These are the defaults with no reviewer. With a
# reviewer, `default_dispatch_caps` scales the Writer's and Critic's caps with
# `max_revisions`, because every revision round needs one more draft and one
# more critique. A per-turn cap that ignored this would silently shorten the
# revision loop below what `--max-revisions` promised.
DEFAULT_DISPATCH_CAPS = {"researcher": 2, "writer": 2, "critic": 0}


def default_dispatch_caps(*, enable_critic: bool, max_revisions: int) -> dict[str, int]:
    """Per-turn dispatch caps that leave room for every allowed revision.

      researcher  2: a first pass and one follow-up, per turn
      writer      a first draft, one per revision, and one spare for a rewrite
                  after mid-round research
      critic      one per draft that can be reviewed: 1 + max_revisions. 0 when
                  the Critic is off, so the dispatch guard refuses it too
    """
    return {
        "researcher": 2,
        "writer": 2 + max_revisions,
        "critic": (1 + max_revisions) if enable_critic else 0,
    }


# How much of the notes and the draft the Supervisor is shown. It needs enough
# to judge coverage, not every word. The cut is marked, so the model knows it
# is looking at an excerpt and not at a suspiciously short document.
VIEW_CHARS = 4000


class SupervisorDecision(BaseModel):
    """The Supervisor's structured reply. Field order matters - see above."""

    rationale: str = Field(
        description=(
            "One to three sentences on what in the current state decides the "
            "next step: what the notes, draft, or critique cover or miss, and "
            "why that points to the agent you are about to choose. Write this "
            "before choosing."
        )
    )
    next: Literal["researcher", "writer", "critic", "finish"] = Field(
        description=(
            "researcher: evidence is missing, thin, or wrong. writer: the "
            "evidence is sufficient and the draft needs writing or rewriting. "
            "critic: a current draft needs reviewing (only if the critic is on "
            "the team). finish: the current draft is ready."
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


_TEAM = {
    "researcher": (
        "- researcher: gathers evidence (searches, or retrieves from the user's "
        "documents) and hands over notes. Send it back with a brief when the "
        "notes miss part of the question, or when a critique shows a source is "
        "missing, fake, or does not support a claim. It merges new findings into "
        "its existing notes."
    ),
    "writer": (
        "- writer: writes the user-facing answer from the notes. It cannot "
        "search. Send it when a draft is needed, or when a critique is about the "
        "writing - structure, an unanswered part the notes already cover, a "
        "claim or citation the notes do not contain."
    ),
    "critic": (
        "- critic: reviews the current draft against the notes and verifies its "
        "arXiv citations. A draft must pass the critic before it can finish."
    ),
}


def supervisor_system_prompt(roster: tuple[str, ...]) -> str:
    """The prompt names only the agents on this run's roster (see the roster
    guard above). The Critic appears only when it is on."""
    team = "\n".join(_TEAM[a] for a in roster)
    normal = " -> ".join([*roster, "finish"]) if "critic" in roster else "researcher -> writer -> finish"
    return (
        "You coordinate a small research team and decide who acts next. You never "
        "research, write, or review yourself.\n\n"
        f"The team:\n{team}\n- finish: end the turn with the current draft.\n\n"
        "Guidance:\n"
        f"- The normal path is {normal}. Deviate only for a concrete reason you "
        "can name.\n"
        "- After a rejection, decide whose problem it is: evidence problems go to "
        "the researcher (with a brief naming what to find), writing problems go "
        "to the writer.\n"
        "- A follow-up research pass is worth it only if you can say what it "
        "should find. If the last pass found nothing new, another pass with the "
        "same brief will not either.\n"
        "- Each agent has a limited number of dispatches this turn and a limited "
        "budget this round, shown below. Plan within them.\n"
        "- Finish when the draft answers the question well enough. Perfect is "
        "not required."
    )


def _excerpt(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return "(empty)"
    if len(text) <= VIEW_CHARS:
        return text
    return text[:VIEW_CHARS] + f"\n[... truncated; {len(text) - VIEW_CHARS} more characters]"


# --------------------------------------------------------------------------
# Staleness, from the dispatch order alone
# --------------------------------------------------------------------------


def _dispatch_order(state: MultiAgentState) -> list[dict]:
    return [e for e in (state.get("supervisor_log") or []) if e["routed_to"] in AGENTS]


def last_dispatched(state: MultiAgentState) -> str:
    """The last agent the Supervisor sent the run to this turn, or ""."""
    order = _dispatch_order(state)
    return order[-1]["routed_to"] if order else ""


def _last_index(state: MultiAgentState, agent: str) -> int:
    order = _dispatch_order(state)
    for i in range(len(order) - 1, -1, -1):
        if order[i]["routed_to"] == agent:
            return i
    return -1


def draft_is_current(state: MultiAgentState) -> bool:
    """Is there a draft, and was it written from the notes as they stand now?

    6.2's rule: if the last agent dispatched was the Researcher and its pass
    changed the notes, the draft predates them. `budget_exhausted` counts as
    changed, which is conservative: it may have appended raw results.
    """
    if not (state.get("draft") or "").strip():
        return False
    if last_dispatched(state) == "researcher" and state.get("research_outcome") in (
        "findings",
        "budget_exhausted",
    ):
        return False
    return True


def draft_was_rejected(state: MultiAgentState) -> bool:
    """Was the draft rejected, with no rewrite since? (6.3)

    A rejection ends in `start_revision`, which bumps `revisions`. So "the Writer
    has run since the last rejection" is "some Writer dispatch in the log is
    stamped with the current revision number". Again, it is derived from the
    log.
    """
    revisions = state.get("revisions", 0)
    if revisions <= 0:
        return False
    return not any(
        e["routed_to"] == "writer" and e.get("revision", 0) == revisions
        for e in _dispatch_order(state)
    )


def draft_needs_writing(state: MultiAgentState) -> bool:
    """No draft, a stale draft, or a rejected one. Each is a reason to send the
    Writer, and a reason not to finish."""
    return not draft_is_current(state) or draft_was_rejected(state)


def critique_is_current(state: MultiAgentState) -> bool:
    """Staleness, second case (6.3): is the critique about the draft that exists now?

    Current only if the Critic was dispatched after the last Writer dispatch.
    One rewrite since or three, the answer is the same, because it is an
    ordering question.
    """
    if not state.get("verdict"):
        return False
    critic = _last_index(state, "critic")
    return critic >= 0 and critic > _last_index(state, "writer")


def approved_by_current_critique(state: MultiAgentState) -> bool:
    return critique_is_current(state) and state.get("verdict") == "approve"


# --------------------------------------------------------------------------
# What the Supervisor sees
# --------------------------------------------------------------------------


def render_supervisor_view(
    state: MultiAgentState,
    caps: dict[str, int],
    *,
    round_caps: dict[str, int] | None = None,
    roster: tuple[str, ...] = ("researcher", "writer"),
    max_revisions: int = 0,
) -> str:
    """Everything the Supervisor sees, as one message. It is a pure function,
    so tests can assert on exactly what the model is shown.

    CONCEPT: what information the routing decision is made from.
      question, plan      what "done" means
      research notes      what evidence exists, excerpted
      research outcome    how the *latest* pass went, as a code-level fact
      draft + currency    what the Writer produced, and whether it predates
                          the latest notes or was rejected (6.3)
      critique (6.3)      verdict, text, whether it is about the current draft,
                          and the citation checks as facts
      human feedback      if a person rejected the draft (6.3)
      revisions (6.3)     how many rejection rounds are left
      budgets             per agent: dispatches this turn, model calls this round
      decisions so far    its own earlier routes and reasons

    What it does *not* see: any agent's private tool loop, and earlier turns of
    the conversation. The Supervisor routes *this* turn.
    """
    round_caps = round_caps or {}
    dispatches = state.get("dispatches") or {}
    budget_lines = []
    for agent in roster:
        line = f"- {agent}: dispatched {dispatches.get(agent, 0)} of {caps.get(agent, 0)} this turn"
        if agent in round_caps:
            b = budget_of(state, agent, round_caps[agent])
            line += f"; {b['used']} of {b['cap']} model calls used this round"
        budget_lines.append(line)
    sub_questions = state.get("sub_questions") or []
    plan = "\n".join(f"- {q}" for q in sub_questions) or "(none)"
    outcome = state.get("research_outcome") or "(researcher has not run this turn)"

    draft = (state.get("draft") or "").strip()
    if not draft:
        draft_note = "(no draft yet)"
    elif draft_was_rejected(state):
        draft_note = "(REJECTED: not rewritten since the last rejection)"
    elif draft_is_current(state):
        draft_note = "(written from the current notes)"
    else:
        draft_note = "(STALE: written before the latest research changed the notes)"

    sections = [
        f"Question: {state.get('question', '')}",
        f"Plan (sub-questions):\n{plan}",
        f"Research notes:\n{_excerpt(state.get('research_notes', ''))}",
        f"Latest research pass: {outcome}",
        f"Draft {draft_note}:\n{_excerpt(draft)}",
    ]

    if "critic" in roster:
        if not state.get("verdict"):
            sections.append("Critique: (the critic has not reviewed a draft yet)")
        else:
            when = (
                "about the CURRENT draft"
                if critique_is_current(state)
                else "about an EARLIER draft, since rewritten"
            )
            checks = state.get("citation_checks") or []
            checked = "\n".join(f"- {c['arxiv_id']}: {c['status']}" for c in checks) or "(none checked)"
            sections.append(
                f"Critique ({state['verdict'].upper()}, {when}):\n{_excerpt(state.get('critique', ''))}\n\n"
                f"Citation checks by the critic:\n{checked}"
            )

    if state.get("human_verdict") == "reject":
        sections.append(f"Human reviewer REJECTED the draft:\n{state.get('human_feedback', '') or '(no reason given)'}")

    if max_revisions or state.get("revisions"):
        sections.append(f"Revision rounds used: {state.get('revisions', 0)} of {max_revisions}")

    log = state.get("supervisor_log") or []
    history = "\n".join(
        f"{e['step']}. -> {e['routed_to']}"
        + (f" (brief: {e['brief']})" if e.get("brief") else "")
        + (f" [overridden: {e['override']}]" if e.get("override") else "")
        + f": {e['rationale']}"
        for e in log
    ) or "(none - this is the first decision)"

    sections += [
        "Budget:\n" + "\n".join(budget_lines),
        f"Your decisions so far:\n{history}",
        "Decide who acts next.",
    ]
    return "\n\n".join(sections)


# --------------------------------------------------------------------------
# Policy and guards
# --------------------------------------------------------------------------


def _availability(state, caps, round_caps, roster):
    """Why each agent cannot be dispatched right now, or "" if it can."""
    dispatches = state.get("dispatches") or {}

    def reason(agent: str) -> str:
        if agent not in roster:
            return f"{agent} is not on the roster"
        if dispatches.get(agent, 0) >= caps.get(agent, 0):
            return f"{agent} dispatch cap ({caps.get(agent, 0)}) reached this turn"
        if agent in round_caps:
            # Knowledge-base research makes no model calls, so a spent round
            # budget does not stop it. Only a live-search pass needs budget.
            needs_budget = not (agent == "researcher" and state.get("mode") == "knowledge-base")
            b = budget_of(state, agent, round_caps[agent])
            if needs_budget and b["used"] >= round_caps[agent]:
                return f"{agent} round budget ({round_caps[agent]} model calls) spent"
        return ""

    return reason


def fixed_policy(
    state: MultiAgentState,
    caps: dict[str, int],
    *,
    round_caps: dict[str, int] | None = None,
    roster: tuple[str, ...] = ("researcher", "writer"),
) -> str:
    """6.1's fixed hand-off as a routing rule, with the Critic slotted in (6.3).

    research if nobody has -> write if the draft needs writing -> critique if
    the Critic is on and has not approved this draft -> finish. Used for
    `routing="fixed"` and as the fallback when the Supervisor's output is
    unusable.

    Note what it does after a rejection: the draft "needs writing", so it
    sends the Writer. A fixed policy cannot classify a critique, so every
    problem is a writing problem to it. That is the gap the model-driven
    Supervisor exists to close.
    """
    round_caps = round_caps or {}
    blocked = _availability(state, caps, round_caps, roster)
    dispatches = state.get("dispatches") or {}

    if dispatches.get("researcher", 0) == 0 and not blocked("researcher"):
        return "researcher"
    if draft_needs_writing(state) and not blocked("writer"):
        return "writer"
    if "critic" in roster and not approved_by_current_critique(state) and not blocked("critic"):
        return "critic"
    return "finish"


def apply_guards(
    proposed: str,
    state: MultiAgentState,
    caps: dict[str, int],
    *,
    round_caps: dict[str, int] | None = None,
    roster: tuple[str, ...] = ("researcher", "writer"),
) -> tuple[str, str]:
    """Return (where the run goes, why the proposal was overridden or "").

    In order:
      1. roster      - an agent not on this run's roster          -> fixed_policy
      2. capacity    - dispatch cap reached, or round budget spent -> fixed_policy
      3. writer before any research                               -> researcher
      4. critic on a draft that needs (re)writing                  -> writer
      5. finish on a draft that needs (re)writing                  -> writer
      6. finish without the Critic's approval of *this* draft      -> critic

    Rules 4-6 bend only when the agent they would send is itself unavailable.
    A rule that sent the run to a capped agent would break the termination
    bound, and the bound beats every preference.
    """
    round_caps = round_caps or {}
    blocked = _availability(state, caps, round_caps, roster)
    dispatches = state.get("dispatches") or {}

    def fallback():
        return fixed_policy(state, caps, round_caps=round_caps, roster=roster)

    if proposed in AGENTS and blocked(proposed):
        return fallback(), blocked(proposed)

    if proposed == "writer" and dispatches.get("researcher", 0) == 0 and not blocked("researcher"):
        return "researcher", "writer proposed before any research"

    if proposed == "critic" and draft_needs_writing(state) and not blocked("writer"):
        return "writer", "critic proposed on a draft that needs (re)writing first"

    if proposed == "finish" and draft_needs_writing(state) and not blocked("writer"):
        if not (state.get("draft") or "").strip():
            reason = "finish proposed with no draft"
        elif draft_was_rejected(state):
            reason = "finish proposed on a rejected draft"
        else:
            reason = "finish proposed with a draft that predates the latest research"
        return "writer", reason

    if proposed == "finish" and "critic" in roster and not approved_by_current_critique(state) and not blocked("critic"):
        reason = (
            "finish proposed on a draft whose approval is stale"
            if state.get("verdict") == "approve"
            else "finish proposed before the critic approved this draft"
        )
        return "critic", reason

    return proposed, ""


def _attempted_route(raw) -> str | None:
    """Recover what the model *tried* to say when validation failed, if possible."""
    if isinstance(raw, dict):
        value = raw.get("next")
        return str(value) if value is not None else None
    return None


def make_supervisor(
    *,
    model: BaseChatModel | None = None,
    caps: dict[str, int] | None = None,
    round_caps: dict[str, int] | None = None,
    enable_critic: bool = False,
    max_revisions: int = 0,
    routing: Literal["supervisor", "fixed"] = "supervisor",
):
    """Build the Supervisor node.

    `routing="fixed"` keeps the node and the hub, and replaces the model with
    `fixed_policy`. It is the baseline, and the only mode that runs without an
    API key.
    """
    roster: tuple[str, ...] = ("researcher", "writer", "critic") if enable_critic else ("researcher", "writer")
    caps = caps or default_dispatch_caps(enable_critic=enable_critic, max_revisions=max_revisions)
    round_caps = round_caps or {}
    system_prompt = supervisor_system_prompt(roster)
    _cache: dict = {}

    def structured():
        if "runnable" not in _cache:
            base = model or get_chat_model()
            # Phase 7: the method is the provider's to decide, not the
            # Supervisor's - json_schema on Anthropic, function_calling on
            # Groq's Llama. See models.structured_output_method.
            _cache["method"] = structured_output_method(base)
            _cache["runnable"] = base.with_structured_output(
                SupervisorDecision, method=_cache["method"]
            )
        return _cache["runnable"]

    def supervisor(state: MultiAgentState) -> dict:
        log = list(state.get("supervisor_log") or [])
        step = len(log) + 1
        brief = ""
        guard_kwargs = {"round_caps": round_caps, "roster": roster}

        if routing == "fixed":
            proposed = fixed_policy(state, caps, **guard_kwargs)
            rationale = "fixed policy (no model)"
            route, override = apply_guards(proposed, state, caps, **guard_kwargs)
        else:
            raw = None
            try:
                raw = structured().invoke(
                    [
                        SystemMessage(content=system_prompt),
                        HumanMessage(
                            content=render_supervisor_view(
                                state, caps, max_revisions=max_revisions, **guard_kwargs
                            )
                        ),
                    ]
                )
                decision = (
                    raw if isinstance(raw, SupervisorDecision) else SupervisorDecision.model_validate(raw)
                )
            except Exception as exc:  # noqa: BLE001 - any failure takes the same path
                # 6.3: keep what the model tried to say, when it can be
                # recovered. 6.2 logged None here, which hid a proposal like
                # "critic" behind a generic validation error.
                proposed = _attempted_route(raw)
                rationale = f"(supervisor output unusable: {type(exc).__name__}: {exc})"
                route = fixed_policy(state, caps, **guard_kwargs)
                override = "fallback to fixed policy"
            else:
                proposed = decision.next
                rationale = decision.rationale.strip()
                route, override = apply_guards(proposed, state, caps, **guard_kwargs)
                # The brief only travels with a route the model actually chose.
                if route == "researcher" and proposed == "researcher":
                    brief = decision.researcher_brief.strip()

        entry: SupervisorLogEntry = {
            "step": step,
            "proposed": proposed,
            "rationale": rationale,
            "routed_to": route,
            "override": override,
            "brief": brief,
            "revision": state.get("revisions", 0),
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
