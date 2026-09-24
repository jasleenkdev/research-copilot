"""Phase 6's State: the same shared dict, now with an owner for every field.

Read `state.py` first. Everything it says about State, reducers, and what may
and may not live in a checkpoint still holds. This module changes one thing -
*who is allowed to write each key* - and that one change is the subject of the
whole file.

--------------------------------------------------------------------------
CONCEPT: why one `draft` field was fine through Phase 5, and breaks now
--------------------------------------------------------------------------
In Phase 5, `State["draft"]` was written by exactly one node, `call_model`. The
same node gathered the evidence (by calling tools), wrote the answer, and wrote
every revision of it. "The draft" never had to answer the question "whose
draft?", because there was only one author. The reviewers *read* it and wrote
their verdicts elsewhere (`critique`, `human_feedback`). One writer per field
held by accident: there was only one writer.

Split `call_model` into agents and that accident stops holding. There are now
three different kinds of output, produced by different nodes at different
times:

    research findings   the Researcher, before any writing happens
    the written answer  the Writer, from the findings
    a judgement of it   the Critic (6.3), after the writing

If those shared one field, each agent would overwrite the others' work. Worse,
the overwrite would not look like a bug. The Writer would read "the draft" and
find research notes; the Critic would review notes as if they were an answer.
Every node would still run and every run would still end. The data flowing
between them would just be quietly wrong, and nothing would raise.

Phase 5's reviewers already showed the fix in miniature. `critique` and
`human_feedback` are separate keys *because* they have separate authors (see
the note on `critique` in state.py). Phase 6 applies that rule to everything:

    ONE FIELD, ONE OWNER. During a turn, a key is written only by the agent
    that owns it. Everyone else may read it.

The payoff is that a state dump explains itself. `research_notes` holds the
Researcher's work and nothing else. If it is empty, the Researcher produced
nothing, and it cannot be because the Writer clobbered it. When the Supervisor
arrives in 6.2 and has to decide who acts next, it reads fields whose meaning
does not depend on which agent happened to run last.

--------------------------------------------------------------------------
CONCEPT: the one exception - the turn boundary
--------------------------------------------------------------------------
`multi_agent_turn_input` (in multi_agent_graph.py) writes almost every key at
once: it clears last turn's notes, draft, and plan. That does not break the
rule. It is the rule's lifecycle. Ownership is *within* a turn. Between turns,
the key's previous value is stale and the turn boundary resets it, exactly as
`turn_input` has reset `iterations` since Phase 4. So the Writer never clears
the Researcher's notes. The next turn does.

`finalize_answer` is the other name worth noticing in `OWNERS` below. It owns
`messages`, which means that during a turn it is the *only* node that appends
to the shared transcript. That is also why it does not clear `draft` after
committing it, the way Phase 4's `finalize_answer` did. `draft` belongs to the
Writer, so the turn boundary clears it instead.

--------------------------------------------------------------------------
CONCEPT: how the rule is enforced, not just documented
--------------------------------------------------------------------------
A convention that lives only in comments erodes the first time somebody is in
a hurry. So ownership is checked in two different ways, and which one applies
depends on whether the agent is a plain node or a subgraph:

  plain nodes   are wrapped in `owns(...)` below. The wrapper looks at the
                keys the node returned and raises `OwnershipError` if any of
                them is not in that node's `OWNERS` entry. The failure is loud
                and immediate, and it names the node and the key.

  subgraphs     (the Researcher) cannot be wrapped without hiding their
                structure from the graph drawing (see multi_agent_graph.py). They are held
                to the rule structurally instead: a compiled subgraph built with
                `output_schema=ResearcherOutput` can only *return* the keys that
                schema declares, whatever its internal nodes wrote.
                `ResearcherOutput` below and `OWNERS["researcher"]` are the same
                set, and a test checks that the two do not drift apart.

The check costs a set difference per node call, which is nothing next to a
model call. It is not a security boundary: a node could still mutate a list in
place. It is a guard against the ordinary mistake, which is returning a key you
should not have.
"""

from collections.abc import Callable
from functools import wraps
from typing import Annotated, Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from research_copilot.state import Mode

# 6.2. How the Researcher's *latest pass* went, as a value code can branch on.
#   findings          this pass produced new evidence, and the notes include it
#   nothing_found     this pass found nothing new. Any notes from an earlier
#                     pass are still there, unchanged.
#   budget_exhausted  the tool loop hit its cap while still searching. Whatever
#                     came back was handed over raw.
# The empty string means the Researcher has not run this turn.
#
# Before 6.2 this distinction existed only as a header inside the notes text.
# That is fine for the Writer, a model that reads prose. It is not fine for the
# Supervisor's *guards*, which are code: "was the budget the problem?" should
# not be answered by searching a string for "budget ran out".
ResearchOutcome = Literal["findings", "nothing_found", "budget_exhausted"]

# 6.2. The places the Supervisor can send the run. "finish" is not an agent. It
# ends the Supervisor's part of the turn: `finalize_answer` commits the draft,
# or `review_draft` asks a human first (6.3, when approval is on).
#
# 6.3: "critic" joins. It is in the Literal - and therefore in the structured-
# output schema - whether or not `--critic` is on. Whether the Critic is on the
# *roster* for a given run is a code guard (agents/supervisor.py), not a
# schema variant. That way a model that proposes the Critic when it is off gets
# its proposal logged verbatim beside the override that refused it, rather than
# being made unable to say it. See "the roster guard" in agents/supervisor.py.
Route = Literal["researcher", "writer", "critic", "finish"]

# Every agent that has a budget. Three since 6.3.
AGENTS: tuple[str, ...] = ("researcher", "writer", "critic")

# 6.3. The Critic's verdict on the draft it was shown. It has two values plus
# "not yet", and deliberately no "edit". Phase 5's reasoning holds: rewriting is
# the Writer's job, and a Critic that rewrites is a second Writer nobody reviews.
Verdict = Literal["approve", "reject"]

# 6.3. The human reviewer's verdict: Phase 4's three decisions. "edit" is the
# one power a person has that the Critic does not.
HumanVerdict = Literal["approve", "edit", "reject"]


class CitationCheck(TypedDict):
    """One `verify_citation` result, lifted out of the Critic's private channel.

    status:
      found       arXiv has a paper with this id
      not_found   the id is well-formed and arXiv has no such paper - a
                  fabricated or mistyped citation
      invalid     the id is not a well-formed arXiv id at all
      error       the lookup itself failed (network, rate limit). This says
                  nothing about the citation, and the Critic is told not to
                  count it against the draft.
    """

    arxiv_id: str
    status: Literal["found", "not_found", "invalid", "error"]


# --------------------------------------------------------------------------
# 6.3: per-agent budgets
# --------------------------------------------------------------------------
class AgentBudget(TypedDict, total=False):
    """One agent's model-call budget for the current revision round.

    `used`  model calls this agent has made in the current round. The Researcher
            and Critic count every step of their tool loops. The Writer counts
            one per draft.
    `cap`   the configured cap, recorded beside `used` so that a reader (the
            Supervisor's view, Studio, Phase 7's eval tooling) sees both halves
            without the build config. Enforcement reads the config, the same
            split Phase 5 made between checkpointed `revisions` and the
            `max_revisions` closure. `cap` in state is a record, not the rule.

    A TypedDict rather than a dataclass. It is a plain dict at runtime, so
    it goes through the checkpoint serializer and shows up in Studio's state
    panel exactly like every other key, and an old checkpoint that lacks a
    field reads back as a dict missing that key, not as a failed object
    construction. `total=False` plus `budget_of()` below is the migration path.
    """

    used: int
    cap: int


def merge_budgets(
    existing: dict[str, AgentBudget] | None, update: dict[str, AgentBudget] | None
) -> dict[str, AgentBudget]:
    """The per-agent reducer for `budgets`.

    CONCEPT: a reducer that merges per agent, and per field
    Three agents and one reset site all write `budgets`. With the default
    overwrite reducer, the Writer returning `{"writer": {...}}` would replace
    the whole dict and erase the Researcher's and Critic's entries. This is the
    6.1 two-agents-one-field bug, one level down. So the merge goes one level
    deep:

        {"researcher": {"used": 3, "cap": 6}, "writer": {"used": 1, "cap": 3}}
      + {"writer": {"used": 2, "cap": 3}}
      = {"researcher": {"used": 3, "cap": 6}, "writer": {"used": 2, "cap": 3}}

    and one level deeper still. Fields inside an entry merge too, so a reset
    can write `{"writer": {"used": 0}}` without knowing the cap, and the
    recorded cap survives.

    What the reducer cannot do is know *who* wrote an entry. That is the job of
    `owns()` (below) for plain nodes. For the subgraph agents, see the note on
    passthrough in `owns`.
    """
    merged: dict[str, AgentBudget] = {k: dict(v) for k, v in (existing or {}).items()}
    for agent, entry in (update or {}).items():
        merged[agent] = {**merged.get(agent, {}), **(entry or {})}
    return merged


def budget_of(state: "MultiAgentState", agent: str, default_cap: int) -> AgentBudget:
    """An agent's budget entry, with defaults for anything missing.

    This is how every reader gets a budget, and it is the migration path. A
    thread checkpointed before 6.3 has no `budgets` key at all, and a thread
    from partway through a turn may lack one agent's entry. Both read as
    "nothing used, configured cap", which is exactly what an agent that has not
    run this round has spent.
    """
    entry = (state.get("budgets") or {}).get(agent) or {}
    return {"used": entry.get("used", 0), "cap": entry.get("cap", default_cap)}


class SupervisorLogEntry(TypedDict):
    """One Supervisor decision, kept for debugging.

    `proposed` and `routed_to` are separate on purpose. `proposed` is what the
    model asked for, and `routed_to` is where the run actually went after the
    code guards had their say. When the two differ, `override` says why. When
    the model's output could not be used at all, `proposed` is None and
    `rationale` records the error.
    """

    step: int
    proposed: str | None
    rationale: str
    routed_to: str
    override: str
    brief: str
    # 6.3. The revision round this decision was made in. It is what lets
    # staleness checks tell "the Writer ran this round" from "the Writer ran
    # before the last rejection", using the log's ordering alone.
    revision: int


class MultiAgentState(TypedDict, total=False):
    """The shared state of the multi-agent graph. Every key has one owner.

    Grouped by owner rather than by phase, because in this graph "who writes
    it" is the first question to ask about any key. The owner of each group is
    in its heading, and `OWNERS` below is the machine-checked version of the
    same thing.
    """

    # --- the request (written only by the turn boundary) ------------------------
    # Read by everyone, written by nobody inside the turn. Same meaning as in
    # state.py.
    question: str
    mode: Mode
    # Phase 4's compressed history. Nothing in 6.1 writes it yet (there is no
    # prune_history in this graph - see the build notes in
    # multi_agent_graph.py), but the Writer already reads it. That way a
    # checkpointed thread that does have a summary is answered correctly.
    summary: str

    # --- the shared transcript (owner: finalize_answer) -------------------------
    # Only human turns and committed answers. The Researcher's tool calls and
    # tool results are NOT here. They live in the Researcher's private
    # `research_messages` channel, which never crosses the subgraph boundary.
    # See agents/researcher.py for why that matters to every agent downstream.
    messages: Annotated[list[BaseMessage], add_messages]

    # --- the plan (owner: plan_question) ----------------------------------------
    # Phase 5's advisory sub-questions, unchanged. Read by the Researcher (it
    # researches each one) and by the Writer (it treats them as a checklist).
    sub_questions: list[str]

    # --- the Researcher's output (owner: researcher) ----------------------------
    # The findings, as text: what was found, and where it came from. In
    # live-search mode this is the Researcher model's own write-up of its
    # searches. In knowledge-base mode it is the retrieved excerpts, numbered
    # for citation. It is the *only* channel from Researcher to Writer. The
    # Writer never sees a tool call, a raw search result, or the Researcher's
    # conversation with itself.
    #
    # Empty string means "the Researcher produced nothing". That is a real
    # outcome - no documents matched, or every search failed - and the Writer
    # is told to say so rather than fill the gap from memory.
    research_notes: str
    # The raw retrieved chunks, knowledge-base mode only. Kept for the same
    # reason `documents` was in Phase 3: so the CLI can show which excerpts a
    # given [n] citation refers to.
    documents: list[Document]
    # How many model calls the Researcher's tool loop spent. It is output (not
    # private) because a state dump should show what the research cost. It is
    # also the first per-agent counter, and 6.3's per-agent budgets start from
    # it. See the note on budgets at the bottom of multi_agent_graph.py.
    research_iterations: int

    # 6.2: how the latest research pass went. See ResearchOutcome above.
    research_outcome: ResearchOutcome | Literal[""]

    # --- the Writer's output (owner: writer) ------------------------------------
    # The answer as written, before it is committed to `messages`. Same reason
    # as Phase 4's `draft`: a proposed answer is a thing the transcript cannot
    # hold. What changed is that the key now has exactly one author, *by rule*
    # rather than by accident.
    draft: str

    # --- the Supervisor's decisions (owner: supervisor) -------------------------
    # CONCEPT (6.2): the Supervisor owns the *routing* fields, and no content.
    # It never writes notes or drafts. It decides who acts next, and records
    # enough to let you check that decision afterwards.

    # Where the routing function sends the run next. The Supervisor's decision
    # is stored in state, and a plain routing function reads it. The model is
    # not asked to name a node directly. See `route_from_supervisor` in
    # multi_agent_graph.py for why the decision and the edge are two steps.
    next_agent: Route | Literal[""]

    # What the Supervisor wants the Researcher to look for on this dispatch.
    # Empty means "research the question as asked". It is how a second research
    # pass differs from the first: without a brief, a re-dispatched Researcher
    # is shown the same inputs and runs the same searches again. The Supervisor
    # writes it and the Researcher reads it, which is ownership working as
    # intended. One agent's output is another agent's input, through a field
    # with a single author.
    researcher_brief: str

    # How many times each agent has been dispatched this turn, e.g.
    # {"researcher": 2, "writer": 1}. The Supervisor owns the count because
    # dispatching is the Supervisor's action: an agent cannot count how often
    # it is *called*. The Researcher in particular starts its private state
    # from scratch on every call (see agents/researcher.py). These counts are
    # checked against per-agent dispatch caps by the Supervisor's guards, and
    # they are what bounds the hub.
    #
    # CONCEPT (6.3): the scope of this number, stated once so nobody has to
    # re-derive it
    #
    #     dispatches[agent]         per TURN. Reset only by the turn boundary.
    #                               NEVER reset by start_revision.
    #     budgets[agent]["used"]    per REVISION ROUND. Reset by start_revision.
    #     revisions                 per TURN. The outer cap on rounds.
    #
    # Same argument as Phase 5's iterations/revisions split, one level down.
    # A count is interpretable on its own only if it is reset on exactly one
    # schedule, and that schedule is written next to it. If `dispatches` reset
    # every revision, "researcher: 2" at the end of a turn would mean "twice in
    # whichever round happened to be last", and total research effort would
    # be unrecoverable from the final state. As a per-turn count it means one
    # thing: how many times this turn's answer sent work to that agent. That is
    # the number Phase 7's evaluation should read as "dispatches used", without
    # consulting this code.
    #
    # The two scopes also bound different things. `dispatches` bounds the *hub*:
    # every non-finish route spends one, so no Supervisor can loop. `budgets`
    # bounds the *work inside a round*: a Researcher sent back twice in one
    # round shares one tool budget across both passes, instead of getting a
    # fresh one each time (the 6.1 per-invocation gap).
    dispatches: dict[str, int]

    # Every decision this turn, oldest first. See SupervisorLogEntry.
    #
    # Deliberately *not* `Annotated[list, operator.add]`, although it is an
    # append-only log. An accumulating reducer cannot be reset: the turn
    # boundary writing `[]` would mean "append nothing", and the log would
    # carry every earlier turn's decisions forever. The Supervisor is the only
    # writer, so it can do the append itself (read the list, return it one
    # longer) under the default overwrite reducer. Then the turn boundary's
    # `[]` really does reset it. Single ownership is what makes that safe: with
    # two writers, read-then-overwrite would lose entries.
    supervisor_log: list[SupervisorLogEntry]

    # --- the Critic's output (owner: critic) -----------------------------------
    # Phase 5's `critique` field, now with one author, the Critic agent. The
    # human's notes moved to their own fields below for the same reason
    # Phase 5 kept them apart: the audit trail must say which reviewer objected.
    critique: str
    verdict: Verdict | Literal[""]
    # What `verify_citation` said, per id, in the critic's latest pass. Lifted
    # out of the private channel as structured results, the same move as
    # `research_outcome`. The Supervisor reads "2401.00001: not_found" as a
    # fact, instead of inferring it from the critique's prose.
    citation_checks: list[CitationCheck]

    # --- the human reviewer's output (owner: review_draft) ---------------------
    # Phase 4's `human_feedback`, plus the two things a human verdict carries
    # that a critic's does not.
    human_verdict: HumanVerdict | Literal[""]
    human_feedback: str
    # CONCEPT (6.3): a human edit gets its own field, because `draft` is the
    # Writer's. In Phase 4 the reviewer's edited text overwrote `draft` - one
    # field, two authors, which the ownership rule now forbids. So the edit
    # lands here, and `finalize_answer` commits `human_edit` if there is one,
    # else `draft`. The final state then shows what the Writer wrote *and*
    # what the human changed it to. 6.1's note on `finalize_answer` predicted
    # the two would stop being the same text; this is where.
    human_edit: str

    # --- the revision loop (owner: start_revision) -----------------------------
    # Phase 5's outer counter, unchanged in meaning: rejection rounds this turn,
    # capped by `max_revisions`. Incremented only by `start_revision`, which is
    # also the single reset site for every agent's `budgets[...]["used"]`.
    revisions: int
    # See AgentBudget and merge_budgets above. Each agent writes only its own
    # entry. `start_revision` writes all of them (the reset).
    budgets: Annotated[dict[str, AgentBudget], merge_budgets]


class ResearcherInput(TypedDict, total=False):
    """What the Researcher subgraph is allowed to *read* from the shared state.

    CONCEPT: a subgraph's input schema is its read contract.
    When a compiled subgraph is added as a node, LangGraph hands it the
    parent's state and keeps only the keys that appear in the subgraph's input
    channels. Declaring them here makes the Researcher's dependencies explicit
    and short. A reader can see it needs the question, the mode, the plan, and
    the conversation so far, and nothing about drafts.

    `messages` is readable so that a follow-up ("and what came after it?") can
    be researched in context. It is deliberately *not* in `ResearcherOutput`,
    so the Researcher can read the transcript and can never write to it.

    6.2 adds three reads:
      researcher_brief  the Supervisor's instruction for this pass
      research_notes,   the Researcher's *own* previous output. It reads these
      documents         so it can merge into them rather than replace them
                        (the merge-on-rerun rule, in agents/researcher.py).
                        Reading your own field back is still single ownership:
                        the owner is the only writer, not the only reader.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    question: str
    mode: Mode
    summary: str
    sub_questions: list[str]
    # --- 6.2 ---
    researcher_brief: str
    research_notes: str
    documents: list[Document]
    # --- 6.3 --- its own budget entry, to know what is left this round
    budgets: Annotated[dict[str, AgentBudget], merge_budgets]


class ResearcherOutput(TypedDict, total=False):
    """What the Researcher subgraph is allowed to *write* to the shared state.

    CONCEPT: a subgraph's output schema is its write contract.
    Whatever the internal nodes put into the subgraph's own state, only these
    keys come back out to the parent. That is the structural half of the
    ownership rule, and it is what keeps the private channel private (see
    agents/researcher.py).
    """

    research_notes: str
    documents: list[Document]
    research_iterations: int
    research_outcome: ResearchOutcome | Literal[""]
    budgets: Annotated[dict[str, AgentBudget], merge_budgets]


class CriticInput(TypedDict, total=False):
    """What the Critic subgraph may read. (6.3)

    The draft, the question it answers, and - new since Phase 5 - the research
    notes it was written from. Phase 5's critic saw only the draft, so "is this
    claim supported?" meant "does it sound supported?". With the notes it
    means "is it in the evidence?", and with `verify_citation` "does the source
    exist?" becomes a lookup rather than a suspicion.

    Not readable: the Supervisor's log, the Researcher's private channel, the
    human's notes. The Critic judges the draft against the evidence. It does not
    judge the process that produced them.
    """

    question: str
    mode: Mode
    sub_questions: list[str]
    draft: str
    research_notes: str
    budgets: Annotated[dict[str, AgentBudget], merge_budgets]


class CriticOutput(TypedDict, total=False):
    """What the Critic subgraph may write back. (6.3)"""

    critique: str
    verdict: Verdict | Literal[""]
    citation_checks: list[CitationCheck]
    budgets: Annotated[dict[str, AgentBudget], merge_budgets]


# --------------------------------------------------------------------------
# The ownership table
# --------------------------------------------------------------------------
# One entry per node that writes shared state. It is the single place to look
# to answer "who is allowed to write this key?", and `owns` below checks the
# plain nodes against it on every call.
#
# There is no entry for the turn boundary because `multi_agent_turn_input` is
# the graph's *input*, not a node. It is merged in before any node runs. See
# "the one exception" at the top of this file.
OWNERS: dict[str, frozenset[str]] = {
    "plan_question": frozenset({"sub_questions"}),
    # Must match ResearcherOutput exactly. tests/test_multi_agent.py checks it.
    "researcher": frozenset(ResearcherOutput.__annotations__),
    # 6.3: and its own `budgets` entry - see BUDGET_ENTRY_OWNERS.
    "writer": frozenset({"draft", "budgets"}),
    # Must match CriticOutput exactly, same check as the Researcher.
    "critic": frozenset(CriticOutput.__annotations__),
    "review_draft": frozenset({"human_verdict", "human_feedback", "human_edit"}),
    "start_revision": frozenset({"revisions", "budgets"}),
    "finalize_answer": frozenset({"messages"}),
    # 6.2. Routing fields only. The Supervisor can read everything and write
    # nothing that an agent produces.
    "supervisor": frozenset(
        {"next_agent", "researcher_brief", "dispatches", "supervisor_log"}
    ),
}


# 6.3: ownership one level down. For the shared `budgets` dict, which *entries*
# each writer may touch. `start_revision` is the reset site, so it writes all
# of them. Everyone else writes only their own.
BUDGET_ENTRY_OWNERS: dict[str, frozenset[str]] = {
    "researcher": frozenset({"researcher"}),
    "writer": frozenset({"writer"}),
    "critic": frozenset({"critic"}),
    "start_revision": frozenset(AGENTS),
}


class OwnershipError(RuntimeError):
    """A node returned a key it does not own.

    Raised rather than logged. The quiet version of this bug - two agents
    taking turns overwriting one field - is exactly what the ownership rule
    exists to make impossible, and a warning nobody reads would leave it
    possible.
    """


def owns(node_name: str) -> Callable[[Callable], Callable]:
    """Wrap a node so that it may only write the keys `OWNERS[node_name]` lists.

    Used as `builder.add_node("writer", owns("writer")(write_draft))`. The node
    name is passed explicitly, rather than taken from the function, so that the
    ownership entry and the graph's node name are visibly the same string at
    the one place they meet.

    6.3 extends the check into `budgets`: a plain node may write only the
    entries BUDGET_ENTRY_OWNERS gives it.

    CONCEPT (6.3): where sub-key ownership stops being enforceable - subgraphs
    The Researcher and Critic subgraphs are not wrapped (see 6.1), and their
    output_schema works at *key* granularity. A subgraph that reads `budgets`
    and updates its own entry returns the *whole* dict as its output: the
    other agents' entries come back as passthrough, unchanged. Under
    `merge_budgets` that passthrough rewrites those entries with the values
    they already had, a no-op, because this graph runs one node at a time. It
    would stop being a no-op the day two branches run in parallel (Phase 7's
    fan-out). Then a stale passthrough copy could overwrite a concurrent
    update. The fix at that point is a per-agent key or a delta reducer, not a
    cleverer merge. `test_subgraph_budget_passthrough_leaves_other_entries_alone`
    pins down the current, sequential behaviour.
    """
    allowed = OWNERS[node_name]

    def decorate(fn: Callable) -> Callable:
        @wraps(fn)
        def checked(state: MultiAgentState) -> dict:
            update = fn(state) or {}
            trespass = set(update) - allowed
            if trespass:
                raise OwnershipError(
                    f"node {node_name!r} wrote {sorted(trespass)}, which it does "
                    f"not own. It may write only {sorted(allowed)}. If another "
                    "agent needs this value, it belongs in a field this node "
                    "owns and the other agent should read it from there."
                )
            # 6.3: the same rule inside `budgets`, per entry.
            if "budgets" in update:
                entries = BUDGET_ENTRY_OWNERS.get(node_name, frozenset())
                foreign = set(update["budgets"] or {}) - entries
                if foreign:
                    raise OwnershipError(
                        f"node {node_name!r} wrote budget entries {sorted(foreign)}, "
                        f"which it does not own. It may write only {sorted(entries)}."
                    )
            return update

        return checked

    return decorate
