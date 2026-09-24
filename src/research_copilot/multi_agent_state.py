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
from typing import Annotated, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from research_copilot.state import Mode


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

    # --- the Writer's output (owner: writer) ------------------------------------
    # The answer as written, before it is committed to `messages`. Same reason
    # as Phase 4's `draft`: a proposed answer is a thing the transcript cannot
    # hold. What changed is that the key now has exactly one author, *by rule*
    # rather than by accident.
    draft: str


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
    """

    messages: Annotated[list[BaseMessage], add_messages]
    question: str
    mode: Mode
    summary: str
    sub_questions: list[str]


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
    "writer": frozenset({"draft"}),
    "finalize_answer": frozenset({"messages"}),
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
            return update

        return checked

    return decorate
