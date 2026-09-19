"""The graph's State: one shared, typed dict that every node reads and writes.

CONCEPT: State
A LangGraph node is a plain function `State -> partial State update`. It takes
the whole state, and returns a dict holding *only the keys it changed*. LangGraph
merges that partial update into the running state and hands the result to the
next node. Nothing is passed node-to-node directly; the state is the only
channel between them. That is what makes a graph inspectable (you can print the
state between any two steps), resumable (Phase 4 snapshots it), and visual
(Studio renders the state after each node).

CONCEPT: reducers
A reducer answers one question: when a node returns a value for key K, how does
that combine with the value K already has?

    reducer(existing_value, update_from_node) -> new_value

The default reducer is *overwrite*: the node's value replaces whatever was
there. That is exactly right for `mode`, `question`, `context`, and
`iterations` - a scalar has one current value, and the newest write wins. If
`call_model` sets `iterations` to 3, the old 2 is meaningless.

It is exactly wrong for `messages`. A conversation *accumulates*: `call_tool`
returning a ToolMessage means "add this to the transcript", not "the transcript
is now this one message". With the default reducer, every node would wipe the
history and the agent would forget everything each step.

So `messages` is annotated with `add_messages`:

    messages: Annotated[list[BaseMessage], add_messages]

`Annotated[T, reducer]` is how LangGraph attaches a reducer to a key - the type
stays `list[BaseMessage]`, and the second argument is metadata LangGraph reads
when it builds the graph. `add_messages` does more than `list.__add__`:
  - appends new messages to the existing list
  - coerces raw dicts / (role, text) tuples into real Message objects
  - assigns an `id` to any message that lacks one
  - *replaces* an existing message when an update carries the same `id`, which
    is how you edit or redact history instead of only appending
  - handles RemoveMessage, the explicit "delete this one" marker that Phase 4's
    trimming will use

The rule of thumb: if a key is a running collection that several nodes
contribute to, it needs a reducer. If it is a single current value, it doesn't.

CONCEPT: one home for conversation history
Phase 1's `agent_loop.py` kept a local `messages` list inside its while-loop.
Phase 2's `ConversationMemory` kept its own `messages` list on a dataclass.
Those were two parallel copies of the same idea, and combining them would have
meant keeping them in sync by hand.

`State["messages"]` replaces both. The graph's state is now the single home for
the transcript:
  - the tool loop appends to it (AIMessage with tool_calls, then ToolMessage)
  - the RAG path appends to it (HumanMessage, then the grounded AIMessage)
  - Phase 4's checkpointer will persist exactly this list, which is what makes a
    conversation survive across process restarts

What is deliberately *not* in `messages`: the system prompt. `call_model` builds
it per-call from `mode`, the same way Phase 2 kept the persona in `CHAT_PROMPT`
rather than in `ConversationMemory.messages`. Pruning, replaying, and
checkpointing a transcript are all simpler when the instructions aren't mixed
into the turns.

CONCEPT (Phase 4): what is in State and what is emphatically not
Phase 4 attaches a checkpointer, which means every key below is now *written to
disk after every super-step*. That turns State's contents into a design
decision with consequences:

  - it is serialized, so everything in it must be serializable. `documents` is
    fine (LangChain `Document` objects round-trip); an open file handle, a
    database connection, or a compiled model object would not be.
  - it is durable, so anything you put here you are choosing to keep. A secret
    dropped into State is a secret written to `data/checkpoints.sqlite3`.
  - it is per-conversation, so it must not hold anything that identifies *which*
    conversation this is. The thread_id is not in State and cannot be - see the
    long note at the top of checkpointing.py. State is the content; `config` is
    the address.

CONCEPT (Phase 4): State keys as a schema you have to migrate
Because old snapshots on disk were written against the State you had *then*,
adding a key is safe (it reads back missing, and `total=False` plus
`state.get(...)` already handles that) while renaming or repurposing one is not.
An existing thread resumed after a rename carries the old key, which no node
reads any more, and the new key is absent. That is why every key here is read
with `state.get(key, default)` and never `state[key]`: the defaults *are* the
migration path for threads written before the key existed.
"""

from typing import Annotated, Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

# The two research paths, named exactly as Phase 2's `ask --mode` names them.
# Phase 6's Supervisor will choose this value from the question itself instead of
# taking it from a CLI flag.
Mode = Literal["knowledge-base", "live-search"]

# Phase 4. The positions a draft can be in, named rather than numbered so that a
# state dump reads like English and an unexpected value is obvious.
#   drafting           nothing is pending; the model is still working
#   awaiting_approval  the model has drafted an answer; a human owes a decision
#   approved           the draft (possibly edited) is going into the transcript
#   rejected           the draft is discarded; `human_feedback` says why
#
# "drafting" is the one that is easy to leave out and shouldn't be. Without it
# there is no value meaning "no decision is pending", so the field can only ever
# be read as stale: a thread whose previous turn ended on "awaiting_approval"
# (because the iteration cap tripped before the reviewer was asked) would send
# the *next* turn straight to review, before the model had drafted anything.
# `run_graph` resets the field to "drafting" at the start of every turn and
# `call_model` sets it back whenever it emits a tool call, so the value always
# describes the current turn rather than some earlier one.
#
# Phase 5 will add a fifth position for the critique loop ("needs_revision"),
# which is why this is a Literal that can grow and not a bool.
ReviewStatus = Literal["drafting", "awaiting_approval", "approved", "rejected"]


class State(TypedDict, total=False):
    """Everything the graph knows, at any point in a run.

    `total=False` means no key is required to be present. Callers seed the run
    with `question`, `mode`, and the opening `messages`; the rest appear as nodes
    produce them. Nodes therefore read optional keys with `state.get(...)`, never
    `state[...]`.
    """

    # --- the transcript -------------------------------------------------------
    # The one accumulating key, and so the one key with a reducer. Grows as:
    #   HumanMessage(question)
    #   AIMessage(tool_calls=[...])   <- call_model
    #   ToolMessage(result)           <- call_tool
    #   AIMessage("final answer")     <- call_model
    messages: Annotated[list[BaseMessage], add_messages]

    # --- the request ----------------------------------------------------------
    # Kept separately from `messages` even though the opening HumanMessage holds
    # the same text: nodes that need the question (the retriever) shouldn't have
    # to dig through a transcript to find it, and after several turns the "first
    # human message" is no longer the current question anyway.
    question: str

    # Which path the router sends the run down. A plain scalar: one value at a
    # time, newest write wins, no reducer.
    mode: Mode

    # --- produced by retrieve_docs (knowledge-base mode only) -----------------
    # The raw chunks, kept so the CLI can show which excerpts produced the
    # answer, exactly as Phase 2's RAG chain returned `docs` alongside `answer`.
    documents: list[Document]
    # The same chunks rendered as numbered text for the prompt. Derived from
    # `documents`, stored because two nodes would otherwise re-derive it.
    context: str

    # --- loop accounting ------------------------------------------------------
    # How many times call_model has run. This is Phase 1's `max_iterations`
    # moved into state, and it's what stops a model that asks for tools forever.
    # Overwrite is the right reducer here: the node computes the new total
    # itself. (`Annotated[int, operator.add]` would be the accumulating
    # alternative - then nodes would return the *delta*, `{"iterations": 1}`.)
    iterations: int

    # --- Phase 4: compressed history -----------------------------------------
    # Phase 2's summarize strategy, as a state key. When `prune_history` folds
    # old turns away under the "summarize" strategy, the gist lands here and
    # `call_model` prepends it to the request as a SystemMessage.
    #
    # Why a separate key rather than a SystemMessage inside `messages`: order.
    # `add_messages` appends, so a summary returned by a node would land at the
    # *end* of the transcript - after the turns it summarizes, which reads as
    # nonsense to the model. Keeping it out of `messages` also preserves the rule
    # set out above: `messages` holds conversation turns, instructions are built
    # per call. Overwrite is the right reducer: each summarization produces the
    # new complete summary (it is given the previous one as input).
    summary: str

    # --- Phase 4: the human-in-the-loop handshake -----------------------------
    # The answer the model has proposed but that has *not* been committed to
    # `messages` yet. This key is the whole reason human-in-the-loop needs richer
    # state: "an answer awaiting approval" is a real thing the system can be in
    # the middle of, and a transcript alone cannot express it. `messages` has
    # room for a turn that happened; it has no room for a turn that is pending.
    #
    # The lifecycle, all of it visible in graph.py:
    #   call_model      writes `draft` + status "awaiting_approval" instead of
    #                   appending to `messages`
    #   review_draft    interrupt()s, then records the verdict
    #   finalize_answer commits the approved (possibly edited) text to `messages`
    #                   and clears the draft
    draft: str

    # Where the draft is in that handshake. The values are deliberately explicit
    # rather than a bare bool: "not yet asked", "asked and refused", and "asked
    # and accepted" are three different states, and a bool can only hold two.
    #
    # This is also what makes a *resumed* run legible. A process that starts up
    # and loads a thread has no memory of what happened before; `status ==
    # "awaiting_approval"` plus `graph.get_state(config).next` is how the CLI
    # knows, from the snapshot alone, that somebody is owed a decision.
    status: ReviewStatus

    # What the reviewer said when they rejected or edited the draft. Kept because
    # a rejection with no reason is not actionable - and because Phase 5's revise
    # loop will feed exactly this text back to the model as the instruction for
    # the next attempt. Writing it down now means Phase 5 adds an edge, not a
    # field.
    human_feedback: str

    # --- loop accounting, part two: why one counter will not be enough --------
    # PHASE 5 NOTE (Part D): where the second iteration counter goes.
    #
    # `iterations` above counts *call_model invocations within one turn* - it is
    # the guard on the call_model <-> call_tool cycle, and `should_continue`
    # compares it against `max_iterations`. That is the tool loop's budget and
    # nothing else's.
    #
    # Phase 5 adds a second cycle: draft -> critique -> revise -> critique,
    # looping until the critic is satisfied or a cap is hit. It is tempting to
    # reuse `iterations` for that cap, and it is wrong, because the two cycles
    # are nested and measure different things:
    #
    #   one revision attempt may itself run the tool loop several times
    #     revise -> call_model -> call_tool -> call_model  (iterations 1, 2)
    #   and the whole thing may then be revised again
    #     critique -> revise -> call_model -> ...          (iterations 3, 4)
    #
    # Share the field and three things break at once. The tool-loop cap trips
    # during revision 2 for work that revision 1 did, so later revisions get less
    # tool budget than earlier ones - the agent silently gets worse the harder it
    # tries. The revise cap trips on tool calls, so a single tool-heavy answer
    # looks like a model that cannot take criticism. And neither number means
    # anything when you read the final state, because you cannot tell which cycle
    # spent it.
    #
    # So Phase 5 adds its own key alongside, not instead:
    #
    #     revisions: int          # how many critique -> revise rounds have run
    #
    # with its own cap (`max_revisions`, separate from `max_iterations`) and its
    # own routing function (`should_revise`, separate from `should_continue`).
    # One more rule comes with it: whichever node begins a revision must reset
    # `iterations` to 0, exactly as `run_graph` resets it at the start of every
    # turn (see the note there about a checkpointed counter that never resets).
    # A per-turn budget that is never reset is a budget that only ever runs out.
    #
    # Deliberately not declared yet - Phase 5 owns it. The note is here so the
    # collision is a decision already made rather than a bug to be found.
