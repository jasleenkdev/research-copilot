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
"""

from typing import Annotated, Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

# The two research paths, named exactly as Phase 2's `ask --mode` names them.
# Phase 6's Supervisor will choose this value from the question itself instead of
# taking it from a CLI flag.
Mode = Literal["knowledge-base", "live-search"]


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
