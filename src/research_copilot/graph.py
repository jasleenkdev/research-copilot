"""Phase 1's tool loop and Phase 2's RAG chain, rebuilt as one StateGraph.

Read `agent_loop.py` beside this file. The two do the same work; only the
control flow moved.

    agent_loop.py (Phase 1)              graph.py (Phase 3)
    ------------------------------------ --------------------------------------
    a local `messages` list               State["messages"] + add_messages
    `for iteration in range(...)`         an edge from call_tool back to call_model
    `if not ai.tool_calls: return`        the should_continue conditional edge
    `model_with_tools.invoke(...)`        the call_model node
    the inner `for call in tool_calls`    the call_tool node
    `if mode == ...` in cli.py            the route_by_mode conditional edge

CONCEPT: StateGraph
A StateGraph is a graph whose nodes are functions over a shared state object.
You declare it in three steps:

    builder = StateGraph(State)      # 1. the state schema (and its reducers)
    builder.add_node("name", fn)     # 2. the work
    builder.add_edge("a", "b")       # 3. the control flow
    graph = builder.compile()        # -> a Runnable

CONCEPT: nodes
A node is `State -> dict`. The dict is a *partial* update: only the keys that
changed. Returning `{"messages": [ai_message]}` means "add this message";
`add_messages` (see state.py) turns that into an append. A node that returns
`{}` or `None` changes nothing, which is legal and occasionally useful.

CONCEPT: edges
`add_edge("a", "b")` is unconditional: after a, always b. `START` and `END` are
the two sentinel nodes - `add_edge(START, "x")` names the entry point, and an
edge to `END` finishes the run. Edges are declared up front, which is why the
graph can be drawn before it is ever executed.

CONCEPT: conditional edges
`add_conditional_edges(source, path_fn, path_map)` is the branch. After `source`
runs, `path_fn(state)` returns a key, and `path_map` translates that key into
the next node's name. The routing function is ordinary Python: it reads state
and returns a string. It is not a node - it does not appear in the trace as a
step and it must not have side effects, because LangGraph may call it while
figuring out the graph's shape.

This file has two of them, and they are the first real branching in the project:
  route_by_mode   which research strategy to use   (chosen from data: `mode`)
  should_continue whether the tool loop keeps going (chosen from the last message)

The shape:

    START ──route_by_mode──┬─ "knowledge-base" ─→ retrieve_docs ─→ call_model
                           └─ "live-search" ────────────────────→ call_model
                                                                      │
                                       ┌──────────────────────────────┘
                                       │
                            should_continue
                                 ├─ "call_tool" ─→ call_tool ─→ call_model  (loop)
                                 └─ END

CONCEPT: compile()
`builder.compile()` validates the graph (every edge points at a node that
exists) and returns a `CompiledStateGraph`, which is a Runnable like everything
else in the project: `.invoke()`, `.stream()`, `.batch()`. Building and
compiling are separate so the structure can be checked - and drawn - before any
model is called.

Worth knowing what compile does *not* check: reachability. A node with no edge
into it compiles happily and simply never runs. Phase 4 relies on that - the
review nodes are registered on every graph, even when approval is switched off,
so that toggling the flag does not change the graph's shape (see the note on
`require_approval` in build_graph).

--------------------------------------------------------------------------
PHASE 4: persistence and the human in the loop
--------------------------------------------------------------------------

Three things arrive, and each one is a single argument or a single call:

    builder.compile(checkpointer=saver)   state survives past .invoke()
    interrupt(payload)                    a node stops the graph and waits
    Command(resume=verdict)               the caller starts it again

CONCEPT: the checkpointer (Part A)
Phase 3's compile() took no checkpointer, so state lived for one `.invoke()`.
Passing one makes LangGraph snapshot the whole State after every super-step,
keyed by the thread_id in `config`. The long explanation - what a checkpoint
contains, why thread_id lives in `config` and not in `State`, and what
MemorySaver and SqliteSaver each buy you - is in checkpointing.py. What matters
*here* is that the graph code is unchanged by it. The nodes below do not know
whether they are being persisted. That is the point: persistence is a
deployment decision, not an application rewrite.

CONCEPT: what persistence breaks, and how this file fixes it
Turning on a checkpointer is not free, and two Phase 3 assumptions stop holding
the moment state survives a turn:

  1. `messages` now grows without bound. Phase 3 started every run from an empty
     transcript, so the token budget Phase 2 worried about quietly stopped
     mattering. With a checkpointer, turn 40 resends turns 1-39. `prune_history`
     (Part B) is the answer.
  2. `iterations` now *persists*. It is the tool loop's per-turn budget, but a
     checkpointed counter is never zero again: turn 2 starts at 2, turn 4 starts
     at 6, and somewhere around turn 4 `should_continue` refuses to let the model
     call a single tool because it thinks it has already run six times. The fix
     is one line in `run_graph` - seed `iterations: 0` in each turn's input - and
     the general rule is in state.py's Phase 5 note: a per-turn budget that is
     never reset is a budget that only ever runs out.

CONCEPT: pruning node vs RemoveMessage - the choice, and why it is not the
choice it looks like (Part B)
These are usually presented as two options. They are really two *questions*,
and a node answers both:

    what the model sees this turn   <- what the node passes to .invoke()
    what is still in the state      <- what the node returns

A node that filters - builds a trimmed list, hands it to the model, returns
`{}` - changes only the first. The model gets a short history; the checkpoint
written right after still holds every message. Cheap, perfectly reversible,
keeps a complete audit trail, and the persisted transcript grows forever.

A node that returns `{"messages": [RemoveMessage(id=...), ...]}` changes both.
`add_messages` (see state.py) understands RemoveMessage as "drop the message
with this id", so the next snapshot genuinely lacks those messages, and every
later turn loads the shorter list.

`prune_history` below does the second. Two reasons:

  - Filtering solves the token bill and leaves the durability problem. Phase 4's
    whole subject is the thing on disk, and if pruning only ever affects one
    request, the object you reload next session is still unbounded. You would
    have fixed the symptom you can see and kept the one you cannot.
  - Filtering has to be repeated, identically, by every future reader of that
    state - Studio, a Phase 7 FastAPI handler, Phase 6's other agents. Pruning
    into the state makes the shorter history the *actual* history, so there is
    one transcript rather than one transcript plus a convention about how to
    read it.

The honest cost, and the thing to keep straight: "deleted from state" is not
"deleted from disk". A checkpointer writes a new row per super-step and never
rewrites old ones, so the messages RemoveMessage drops are gone from the
*latest* snapshot and still sitting in every earlier row of that thread. That is
what makes time travel work, and it is why pruning is a context-window and
token-cost mechanism, not a deletion mechanism. If you need a message gone for
real - a leaked secret, a deletion request - you delete the thread
(`checkpointer.delete_thread(thread_id)`), because that is the only operation
that touches the history rows.

CONCEPT: interrupt() and Command (Part C)
`interrupt(payload)` is called *inside a node*. It stops the graph, writes a
checkpoint, and surfaces `payload` to whoever called `.invoke()`, which returns
with an `__interrupt__` key instead of running to completion. The graph is now
parked: `graph.get_state(config).next` names the node that was interrupted, and
nothing else happens until someone resumes.

Resuming is `graph.invoke(Command(resume=verdict), config)`. `Command` is a
control-flow object rather than state - it is how a caller says "continue" the
way a routing function says "go to that node". The value in `resume` becomes the
return value of the `interrupt()` call that paused, so the node picks up with
the human's verdict in hand.

The one mechanic that surprises everyone, and that this file is arranged
around: **a resumed node re-runs from its first line.** LangGraph does not
suspend a Python frame; it replays the node and feeds the stored resume value to
`interrupt()` when execution reaches it again. So everything above the
`interrupt()` call happens twice. If a model call sat there, you would pay for
it twice - and that is exactly why `call_model` drafts the answer and
`review_draft` only reviews it. The expensive work happens in a node that never
interrupts; the node that interrupts does nothing before the call but read
state. Keep pre-interrupt work cheap and idempotent, or move it upstream.

CONCEPT: a draft is state the transcript cannot hold
`review_draft` sits between "the model produced an answer" and "the answer is
part of the conversation", and Phase 3's state had nowhere to put that. A
`messages` list can represent a turn that happened; it cannot represent a turn
that is proposed. So the answer goes to `State["draft"]` with
`status="awaiting_approval"`, and only `finalize_answer` commits it to
`messages` - which is what makes "reject" possible at all. Had the answer been
appended first, rejecting it would mean editing history instead of declining to
write it.

--------------------------------------------------------------------------
PHASE 5: reflection and planning
--------------------------------------------------------------------------

Phase 4 ended with a draft, a reviewer, and a dead end: a rejection recorded a
withheld answer and the run stopped. Phase 5 closes that into a loop and puts a
model in the reviewer's seat beside the human.

Three things arrive:

    revisions / max_revisions / should_revise   the reflection loop's own budget
    critique_draft                              a model where the human sat
    plan_question                               decompose before researching

CONCEPT: two loops, two counters (Part A)
The graph now has two cycles, nested:

    the tool loop        call_model <-> call_tool      counted by `iterations`
    the reflection loop  call_model -> review -> revise counted by `revisions`

They get separate counters, separate caps, and separate routing functions. The
full argument is in state.py beside the `revisions` declaration; the short
version is that one revision attempt can itself run the tool loop several
times, so a shared counter would make the tool cap trip during revision 2 for
work revision 1 did, and neither number would mean anything afterwards.

The rule the split forces: `start_revision` resets `iterations` to 0 when it
begins a round, exactly as `turn_input` resets it when it begins a turn. That is
why `revision_input` exists beside `turn_input` at the bottom of this file -
same pattern, one level down.

CONCEPT: should_revise, and why a cap must be able to overrule a verdict
(Part B)
`should_revise` reads the reviewer's verdict and decides what happens next.
Approve finalizes; reject revises - *unless* `revisions` has hit its cap, in
which case the run ends even though the verdict still says reject.

That override is the point of the function, not a corner of it. The verdict is
produced by the thing being guarded: a human who keeps saying no, or a model
that is very willing to keep saying no. "Loop until the reviewer is satisfied"
puts the termination condition inside the reviewer, and a reviewer that is never
satisfied is not an unusual case - it is the default behaviour of a critic
prompt asked to find fault, which will always find something. There is no
verdict a reviewer can return that means "and I promise to stop eventually".

So the loop counts its own attempts and stops on its own authority. The verdict
decides *quality*; the cap decides *when we are done spending*. Anything that
can loop on a judgement it did not make needs a bound it controls itself.

CONCEPT: the critic is the human, structurally (Part C)
`critique_draft` and `review_draft` are deliberately the same node twice:

    review_draft     reads `draft` -> asks a human    -> verdict -> should_revise
    critique_draft   reads `draft` -> asks a model    -> verdict -> should_revise

Same input, same output shape, same outgoing edge, same fail-closed parsing
(`_parse_critique` hands its result to the very `_parse_verdict` the human path
uses). The only difference is who supplies the verdict, and that difference is
one node body - not a different graph.

This is why human-in-the-loop was worth building first even though it is the
less automatic feature: the pattern generalizes. "Something outside this node
judges the draft and the graph routes on the judgement" covers a human at a
terminal, a model, a test suite, a schema validator, a second graph. Phase 6's
Critic agent is this node again with tools and a system prompt of its own.

One thing the symmetry does *not* cover, and it is the reason `--critic` needs
no checkpointer while `--approve` does: only the human pauses. `interrupt()`
needs somewhere to park a run; a model call just blocks and returns.

CONCEPT: ordering, when both reviewers are on
With `--critic --approve` the critic goes first and the human is asked only
about drafts the critic passed. The trade-off is written out at the top of
`critique_draft`; the consequence to keep in mind is that a critic which never
approves means the human is never asked at all.

CONCEPT: planning (Part D)
`plan_question` runs before the mode branch and writes `sub_questions`. It is
advisory: `retrieve_docs` retrieves for each sub-question as well as the whole
question, and `call_model` gets the list as a checklist. There is no fan-out and
no per-sub-question orchestration - that is Phase 6.

The shape, with Phase 4's nodes marked (*) and Phase 5's (**):

    START ─→ prune_history(*) ─→ plan_question(**) ──route_by_mode──┬─ "knowledge-base" ─→ retrieve_docs ─┐
                                                                    └─ "live-search" ───────────────────┐ │
                                            ┌───────────────────────────────────────────────────────────┴─┘
                                            ↓
     ┌───────────────────────────────→  call_model ──should_continue──┬─ "call_tool" ─→ call_tool ┐
     │                                      ↑                          │                          │
     │                                      └──────────────────────────┼──────────────────────────┘
     │                                                                 ├─ "critique_draft" ─→ critique_draft(**)
     │                                                                 ├─ "review_draft" ──→ review_draft(*)
     │                                                                 └─ "end" ──────────────────→ END
     │                                                                          │        │
     │                                          both route through should_revise ────────┘
     │                                                    │
     │  start_revision(**) ←── "start_revision" ──────────┤
     └────────┘                                           ├── "review_draft" ─→ review_draft   (critic passed,
                                                          │                                     human still owes
                                                          │                                     a verdict)
                                                          └── "finalize_answer" ─→ finalize_answer(*) ─→ END
"""

from collections.abc import Sequence
from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    trim_messages,
)
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

# CONCEPT: the two Phase 4 primitives, both from langgraph.types.
#   interrupt(payload) - called inside a node; stops the graph and hands
#                        `payload` to whoever called .invoke()
#   Command(resume=v)  - passed to .invoke() instead of a state dict; restarts
#                        the parked graph, and `v` becomes interrupt()'s return
# They are control flow, not state, which is why neither appears in state.py.
from langgraph.types import Command, interrupt

# Reused verbatim from Phase 1 so that the manual loop and the graph execute
# tools through identical code. Every behavioural difference between
# `agent --structured` and `graph-agent` is then a difference in control flow,
# not in tool handling. (LangGraph ships `langgraph.prebuilt.ToolNode`, which is
# this node already written - see prebuilt.py for what leaning on it costs you.)
from research_copilot.agent_loop import _execute_tool_call
from research_copilot.checkpointing import thread_config
from research_copilot.config import get_settings
# Phase 2's summarizer, reused rather than rewritten: `prune_history` under the
# "summarize" strategy is Phase 2's `ConversationMemory._summarize` expressed as
# a node. Same prompt, same rendering, different home.
from research_copilot.memory import render_messages
from research_copilot.models import get_chat_model
from research_copilot.prompts import (
    AGENT_SYSTEM_PROMPT,
    CRITIC_PROMPT,
    PLAN_PROMPT,
    RAG_PROMPT,
    REVISION_INSTRUCTIONS,
    SUMMARY_PROMPT,
)
from research_copilot.retrieval import format_docs, get_retriever
from research_copilot.state import Mode, ReviewStatus, State
from research_copilot.tools import search_arxiv

# The same default as Phase 1's `run_tool_loop`.
DEFAULT_MAX_ITERATIONS = 6

# How many recent messages `prune_history` keeps verbatim before it will consider
# summarizing. Mirrors ConversationMemory.keep_last_messages.
DEFAULT_KEEP_LAST_MESSAGES = 4

# PHASE 5 (Part A): the reflection loop's cap, and pointedly *not* the same
# number as DEFAULT_MAX_ITERATIONS above. Two budgets, two constants.
#
# The default is 0, which means "a rejection ends the run" - Phase 4's exact
# behaviour. Every Phase 4 argument defaulted to Phase 3's behaviour for the same
# reason: a new phase should be something you switch on, so that an existing
# caller's output does not change under it and the difference stays legible.
# `cli.py` resolves its own default from settings (2), because on the command
# line `--critic` with no revisions allowed would be a critic that can only ever
# veto.
DEFAULT_MAX_REVISIONS = 0

# PHASE 5 (Part D): the ceiling on decomposition. A planner with no limit is a
# cost multiplier - each sub-question is another retrieval or another tool call.
DEFAULT_MAX_SUB_QUESTIONS = 4


def build_graph(
    *,
    model: BaseChatModel | None = None,
    tools: Sequence[BaseTool] | None = None,
    retriever: BaseRetriever | None = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    # --- Phase 4 -------------------------------------------------------------
    checkpointer: BaseCheckpointSaver | None = None,
    require_approval: bool = False,
    memory_strategy: str | None = None,
    max_history_tokens: int | None = None,
    keep_last_messages: int = DEFAULT_KEEP_LAST_MESSAGES,
    summary_model: BaseChatModel | None = None,
    # --- Phase 5 -------------------------------------------------------------
    enable_critic: bool = False,
    max_revisions: int = DEFAULT_MAX_REVISIONS,
    critic_model: BaseChatModel | None = None,
    enable_planning: bool = False,
    planner_model: BaseChatModel | None = None,
    max_sub_questions: int = DEFAULT_MAX_SUB_QUESTIONS,
) -> Runnable:
    """Wire up and compile the graph.

    Everything injectable is an argument, for the same reason Phase 2's
    `build_rag_chain` took a model and a retriever: the tests swap in fakes and
    run the whole graph offline.

    The nodes are defined inside this function as closures. A node's signature
    is fixed - LangGraph calls it with the state - so anything else it needs
    (the model, the tool table, the retriever) has to be captured, not passed.

    Phase 4 arguments, all defaulting to Phase 3's behaviour so that every
    Phase 3 caller and test keeps working unchanged:

      checkpointer        the saver passed to compile(). None reproduces Phase 3
                          exactly: state lives for one .invoke() and is dropped.
      require_approval    whether a finished answer pauses for human review
                          before it is committed to `messages`.
      memory_strategy     "trim", "summarize", or "none" - the same three-way
                          choice as Phase 2's --memory flag, now applied to the
                          persisted transcript by the prune_history node.
      max_history_tokens  the budget prune_history enforces. None means "read it
                          from .env"; 0 disables pruning entirely.
      summary_model       the model that writes the summary, separated from
                          `model` so a test can script them independently and so
                          production can use a cheaper model for compression.

    Phase 5 arguments, all defaulting to Phase 4's behaviour for the same
    reason - switching a phase on should be a visible act:

      enable_critic       whether `critique_draft` reviews the answer before
                          (optionally) the human does. Needs no checkpointer:
                          a model call blocks and returns, it does not park.
      max_revisions       how many critique -> revise rounds one turn may spend.
                          0 reproduces Phase 4: a rejection ends the run.
      critic_model        the model in the reviewer's seat, separate from
                          `model` for the same two reasons `summary_model` is -
                          a test scripts them independently, and in production
                          you may well want a different (often stronger, or at
                          least differently-prompted) model judging than
                          writing. A critic that *is* the writer grades its own
                          homework: same blind spots, same hallucinations,
                          reliably generous.
      enable_planning     whether `plan_question` decomposes the question into
                          sub-questions before research.
      planner_model       likewise separable, and a good candidate for a cheap
                          model - decomposition is a much easier job than the
                          answer.
      max_sub_questions   the ceiling on that decomposition.
    """
    tools = list(tools) if tools is not None else [search_arxiv]
    tools_by_name = {tool.name: tool for tool in tools}

    settings = get_settings()
    strategy = (memory_strategy or settings.memory_strategy).lower()
    budget = (
        settings.max_history_tokens
        if max_history_tokens is None
        else max_history_tokens
    )

    # PHASE 5: "is anything going to review this draft?" - true if either
    # reviewer is switched on.
    #
    # This is the flag `call_model` and `should_continue` actually need, and
    # collapsing the two switches into it here is what keeps the rest of the
    # file from repeating `require_approval or enable_critic` at every branch.
    # It also names the real precondition: the draft/status handshake exists so
    # that *someone* can look at the answer before it is committed, and the
    # graph does not care whether that someone has a pulse.
    review_enabled = require_approval or enable_critic

    # The model is built on first use rather than here, for the same reason the
    # retriever is: the graph's *structure* doesn't depend on either, so
    # compiling it, drawing it, or opening it in Studio shouldn't require an API
    # key or load a 90 MB embedding model.
    _responders: dict[str, Runnable] = {}

    def responder(*, with_tools: bool) -> Runnable:
        key = "with_tools" if with_tools else "plain"
        if key not in _responders:
            base = model or get_chat_model()
            # CONCEPT: bind_tools returns a *new* Runnable with the tool schemas
            # attached; the original is untouched. Both are kept, because only
            # live-search mode should be able to call tools. In knowledge-base
            # mode the model is handed retrieved text and asked to answer from it
            # alone - offering it a live search there would let it quietly escape
            # the knowledge base, which is the one thing that mode promises not
            # to do.
            _responders[key] = base.bind_tools(tools) if with_tools else base
        return _responders[key]

    # PHASE 5: the two auxiliary models, built lazily for the same reason as
    # `responder` - compiling or drawing the graph must not need an API key.
    #
    # Both fall back to `model` and then to the factory, so a caller who passes
    # nothing gets one model doing all three jobs. That is the convenient
    # default and the weaker configuration: see `critic_model` in the docstring
    # for why a critic that is the writer grades its own homework.
    _auxiliaries: dict[str, BaseChatModel] = {}

    def auxiliary(key: str, override: BaseChatModel | None) -> BaseChatModel:
        if key not in _auxiliaries:
            # No bind_tools: neither the critic nor the planner may call tools.
            # The critic judges the draft it was handed - letting it go and
            # research the question itself would make it a second writer, and a
            # reviewer that does its own research reviews its own findings. The
            # planner's whole job is one cheap structural call; a planner that
            # can search is a research loop hiding inside a planning node, and
            # it would spend `iterations` that the tool loop has not begun yet.
            _auxiliaries[key] = override or model or get_chat_model()
        return _auxiliaries[key]

    # ------------------------------------------------------------------ nodes

    def _summary_messages(summary: str | None) -> list[BaseMessage]:
        """Phase 2's `ConversationMemory._summary_messages`, unchanged in spirit.

        The summary is rendered as a SystemMessage at request time rather than
        stored in `messages` - see the note on `summary` in state.py for why
        order makes that necessary.
        """
        if not summary:
            return []
        return [
            SystemMessage(content=f"Summary of earlier conversation:\n{summary}")
        ]

    def prune_history(state: State) -> dict:
        """Keep the persisted transcript inside a token budget. (Part B)

        Phase 2 solved this on a dataclass that lived for one `while` loop, so
        "the history" and "what we send the model" were the same list and the
        distinction never came up. With a checkpointer they come apart, and this
        node has to choose. It chooses to delete: the RemoveMessage objects it
        returns go through `add_messages`, which drops those ids, so the next
        checkpoint genuinely holds the shorter list. The reasoning, and the
        exact sense in which deleted-from-state is *not* deleted-from-disk, is in
        the Part B section of this module's docstring.

        Where it runs is a design decision too. This node sits at the entry, so
        it prunes once per turn, before the model is first called - not between
        every call_model and call_tool. Pruning inside the tool loop risks
        orphaning a ToolMessage from the AIMessage that requested it, which the
        Anthropic API rejects outright (every tool_use block needs its matching
        tool_result in the same request). Within one turn the transcript can only
        grow by `max_iterations` model calls plus their tool results, which is
        bounded and small; across turns it is unbounded, which is the actual
        problem. Prune where the growth is unbounded.
        """
        messages = list(state.get("messages", []))
        summary = state.get("summary", "")

        # budget <= 0 or strategy "none" turns the node into a no-op, and a
        # node returning {} changes nothing. Keeping the node in the graph even
        # when it does nothing is deliberate - see the note on graph shape in
        # the wiring section below.
        if strategy == "none" or budget <= 0 or not messages:
            return {}

        # The summary counts against the budget. It is part of what gets sent,
        # so leaving it out would make the budget a number about the wrong thing.
        if count_tokens_approximately(_summary_messages(summary) + messages) <= budget:
            return {}

        # CONCEPT: reuse the trimmer to decide, then express the decision as
        # deletions. `trim_messages` already knows the rules that make a kept
        # history *valid* - start on a human turn, never cut a message in half,
        # keep the newest - so it picks the survivors; this node only has to
        # turn "not a survivor" into a RemoveMessage. Rewriting that selection by
        # hand is how you end up with a transcript the API refuses.
        kept = trim_messages(
            messages,
            max_tokens=budget,
            token_counter=count_tokens_approximately,
            strategy="last",
            start_on="human",
            include_system=False,
            allow_partial=False,
        )

        # A budget smaller than the current question leaves `kept` empty, which
        # would delete the turn we are about to answer. Fall back to the tail
        # from the last human turn onwards: over budget, but coherent. Answering
        # a question you just deleted is not a cheaper failure, it is a worse one.
        if not kept:
            starts = [
                i for i, m in enumerate(messages) if isinstance(m, HumanMessage)
            ]
            kept = messages[starts[-1]:] if starts else messages[-1:]

        kept_ids = {m.id for m in kept}
        dropped = [m for m in messages if m.id not in kept_ids]
        if not dropped:
            return {}

        # CONCEPT: RemoveMessage
        # A marker message, not a real turn. `add_messages` sees it, finds the
        # message with that id, and removes it instead of appending. It is the
        # only way a node can *subtract* from a reduced key - a node returns a
        # partial update, and "here is a shorter list" would be read by
        # `add_messages` as "append all of these again".
        update: dict = {"messages": [RemoveMessage(id=m.id) for m in dropped]}

        if strategy == "summarize":
            # The expensive strategy: one model call per compression, buying a
            # compact record of what was deleted. `trim` just loses it.
            compressor = summary_model or model or get_chat_model(max_tokens=1024)
            chain = SUMMARY_PROMPT | compressor | StrOutputParser()
            update["summary"] = chain.invoke(
                {
                    "previous_summary": summary or "(none)",
                    "conversation": render_messages(dropped),
                }
            ).strip()

        return update

    # ------------------------------------------------- Phase 5: planning

    def plan_question(state: State) -> dict:
        """Decompose the question into sub-questions, when that helps. (Part D)

        CONCEPT: planning as a node that usually declines to plan
        This runs before the mode branch, so it is the first thing that happens
        to a question after pruning. Its output is `sub_questions`: a list, very
        often empty, and empty is a *result* rather than a failure. The prompt
        spends most of its words on when not to decompose, because a model asked
        to split a question will split it - "who wrote the BERT paper?" comes
        back as four sub-questions and one cheap lookup becomes four.

        So the node treats "NONE" as a first-class answer and a caller who gets
        `sub_questions == []` should carry on exactly as Phase 4 did. The
        expensive failure here is not the planner refusing to plan; it is the
        planner planning when it should have refused.

        CONCEPT: where this sits relative to `mode` (noted, not solved)
        `mode` is in state before this node runs, so the plan is mode-aware in
        principle, and the prompt is told which mode it is planning for. What
        the two modes then *do* with a plan differs in a way this phase does not
        try to unify:

          knowledge-base  `retrieve_docs` runs the retriever once per
                          sub-question and merges. n sub-questions means up to
                          (1 + n) * k chunks, so decomposition buys precision
                          and spends context window. `prune_history`'s budget
                          does not cover `context` - it is built per call and
                          never enters `messages` - so nothing currently stops a
                          wide plan from producing a very large prompt.

          live-search     the sub-questions are advice to the tool loop, and the
                          loop pays for them out of `max_iterations`. A plan
                          with 4 sub-questions against a cap of 6 leaves little
                          room to iterate on any of them, and nothing here
                          couples the two numbers. That coupling is real and
                          deliberately left alone: it is a budgeting question,
                          not a loop-mechanics one, and Parts A-C do not depend
                          on it. `max_sub_questions` is the blunt instrument in
                          the meantime.

        Both are worth knowing before turning `--plan` on with a low
        `--max-iterations`; neither is load-bearing for the reflection loop.

        CONCEPT: a no-op node still belongs in the graph
        With planning off this returns `{}` and changes nothing, exactly as
        `prune_history` does under `--memory none`. Same reasoning as the review
        nodes: one graph shape whatever the flags say.
        """
        if not enable_planning:
            return {}

        question = state.get("question", "")
        if not question:
            return {"sub_questions": []}

        chain = PLAN_PROMPT | auxiliary("planner", planner_model) | StrOutputParser()
        raw = chain.invoke(
            {
                "question": question,
                "mode": state.get("mode", "live-search"),
                "max_sub_questions": max_sub_questions,
            }
        )
        return {"sub_questions": _parse_plan(raw, max_sub_questions)}

    def retrieve_docs(state: State) -> dict:
        """Knowledge-base path: fetch the chunks that call_model will answer from.

        Phase 2 did this inside an LCEL chain (`RunnableParallel(question=...,
        docs=retriever)`). As a node it is the same retriever call; the
        difference is that its output lands in shared state, where every later
        node - and you, in Studio - can see it.
        """
        # Built lazily rather than in build_graph, because constructing a
        # retriever opens the Chroma collection and loads a ~90 MB embedding
        # model. Nothing should pay that cost just to compile the graph or draw
        # it in Studio.
        active_retriever = retriever if retriever is not None else get_retriever()

        # PHASE 5 (Part D): retrieve for the plan as well as the question.
        #
        # This is the whole of what decomposition buys on this path, and it is
        # deliberately the simplest possible use of `sub_questions`: one
        # retrieval per query, results merged, order preserved, duplicates
        # dropped. The question itself always goes first, so a bad plan can
        # only ever *add* chunks - it cannot displace the ones a Phase 4 run
        # would have found. An empty plan makes this identical to Phase 4.
        #
        # What it is not: fan-out. Nothing answers the sub-questions separately
        # and nothing merges per-sub-question answers. One retrieval set, one
        # model call, one answer - see the note on `sub_questions` in state.py.
        queries = [state["question"], *state.get("sub_questions", [])]

        documents: list[Document] = []
        seen: set[str] = set()
        for query in queries:
            for document in active_retriever.invoke(query):
                # Sub-questions overlap by construction - they are parts of one
                # question - so the same chunk comes back for several of them.
                # Deduplicating on content rather than on identity because a
                # retriever may well hand back equal-but-distinct objects, and
                # a prompt that lists the same excerpt as [2] and [5] invites
                # the model to cite it twice as if it were two sources.
                if document.page_content in seen:
                    continue
                seen.add(document.page_content)
                documents.append(document)

        return {"documents": documents, "context": format_docs(documents)}

    def _revision_instruction(state: State) -> list[BaseMessage]:
        """The "you are re-drafting, here is why" preamble, or nothing. (Part B)

        CONCEPT: how a rejection reaches the model
        `should_revise` routes a rejection back to `call_model`, but an edge
        carries no data - everything a node knows, it reads from state. So the
        revision instruction is assembled here, from the fields the reviewers
        wrote: the draft that was rejected, plus `critique` and
        `human_feedback`.

        CONCEPT: `revisions > 0` is the signal, and why it has to be
        There is no "is this a revision?" flag, and adding one would be a third
        thing to keep in sync with the counter that already knows. `revisions`
        is incremented by `start_revision` and reset to 0 by `turn_input`, so
        "we are mid-revision" is exactly "the counter is not zero" - which is
        also why `turn_input` has to clear `critique` and `human_feedback`
        along with it. A turn that inherited last turn's rejection note would
        open by apologising for an answer the user never saw.

        Note this fires on *every* `call_model` inside a revision round, not
        only the first. That is intended: if the revision calls a tool, the
        model that reads the tool result still needs to know what it is fixing.
        `iterations` counts within the round; `revisions` counts the rounds.

        CONCEPT: both reviewers' notes, labelled by source
        When `--critic` and `--approve` are both on they can disagree, and the
        honest thing to hand the writer is both objections with their authors
        attached. Merging them would lose which one came from a person - and if
        they contradict each other, that is precisely what the writer needs to
        see in order to say so. (`_parse_critique` and `REVISION_INSTRUCTIONS`
        both tell the model it may push back on a criticism rather than
        thrashing between two.)
        """
        if state.get("revisions", 0) <= 0:
            return []

        feedback = []
        if state.get("critique"):
            feedback.append(f"- Machine critic: {state['critique']}")
        if state.get("human_feedback"):
            feedback.append(f"- Human reviewer: {state['human_feedback']}")
        if not feedback:
            # A rejection with no reason. It happens - `_parse_verdict` invents
            # one for a garbled verdict, and a human can reject with an empty
            # note - and it is worth saying out loud rather than sending an
            # empty bullet list, which a model reads as "nothing was wrong".
            feedback.append(
                "- The draft was rejected without a reason being recorded. "
                "Re-read the question and write the strongest answer you can."
            )

        return [
            SystemMessage(
                content=REVISION_INSTRUCTIONS.format(
                    attempt=state.get("revisions", 0),
                    cap=max_revisions,
                    draft=state.get("draft", "") or "(the previous draft was empty)",
                    feedback="\n".join(feedback),
                )
            )
        ]

    def _plan_checklist(state: State) -> list[BaseMessage]:
        """The sub-questions, as a checklist for the writer. (Part D)

        Built per call and never appended to `messages`, for the same reason the
        persona and the summary are: it is an instruction about this request,
        not a turn in the conversation. Empty plan, empty list, and `call_model`
        behaves exactly as it did in Phase 4.
        """
        sub_questions = state.get("sub_questions") or []
        if not sub_questions:
            return []
        listed = "\n".join(f"- {q}" for q in sub_questions)
        return [
            SystemMessage(
                content=(
                    "The question was broken down into these sub-questions. "
                    "Work through them and make sure the final answer covers "
                    "each one, but write a single connected answer - not a list "
                    "of separate replies.\n" + listed
                )
            )
        ]

    def call_model(state: State) -> dict:
        """Ask the model for the next step: a tool call, or a final answer.

        This is Phase 1's `model_with_tools.invoke(messages)` line, plus the
        choice of which system prompt and which model variant the mode calls for.

        PHASE 5: this node now does double duty - it drafts *and* it revises.
        The split that Phase 4 made for `interrupt()`'s sake (the expensive work
        happens in a node that never pauses) is what makes that free: a
        revision is just another call to the drafting node with a different
        preamble, so `should_revise` can route back here without a second
        model-calling node existing. That stops being true in Phase 6, where a
        dedicated Writer and a dedicated Reviser would be separate agents with
        separate prompts; see the Phase 6 note at the bottom of this file.
        """
        mode = state.get("mode", "live-search")

        if mode == "knowledge-base":
            # Phase 2's RAG_PROMPT renders to [SystemMessage(rules),
            # HumanMessage(context + question)]. The retrieved context goes into
            # the *request* but is deliberately never appended to
            # State["messages"]: context is re-retrieved for each question, so
            # storing it in the transcript would bloat the history (and later,
            # the checkpoint) with excerpts that are already stale by the next
            # turn. `documents` and `context` keep it in state for inspection.
            request = RAG_PROMPT.format_messages(
                context=state.get("context", "(no excerpts retrieved)"),
                question=state["question"],
            )
            # PHASE 5: the plan and the revision note are spliced in *after* the
            # RAG system message and before the human turn carrying the context.
            # Order matters on this path: RAG_PROMPT's first message is the
            # "answer only from the context" rule, and the revision instruction
            # must not land above it, or a reviewer's "add more detail" starts
            # reading like permission to go beyond the excerpts - which is the
            # one thing knowledge-base mode promises not to do.
            request = [
                request[0],
                *_plan_checklist(state),
                *_revision_instruction(state),
                *request[1:],
            ]
            # No tools: answer from the excerpts or admit the gap.
            next_step = responder(with_tools=False)
        else:
            # The persona and tool-use policy are rebuilt per call rather than
            # stored in `messages` - see the note at the end of state.py.
            # Phase 4 inserts one more built-per-call SystemMessage: whatever
            # prune_history has summarized away. It goes before the surviving
            # turns, which is the position `add_messages` could never have given
            # it (it appends), and is why `summary` is its own state key.
            request = [
                SystemMessage(content=AGENT_SYSTEM_PROMPT),
                *_summary_messages(state.get("summary")),
                *_plan_checklist(state),
                # PHASE 5: last of the built-per-call instructions, so it sits
                # closest to the transcript it is asking to be redone. Note it
                # goes *before* `messages` rather than after: a revision note
                # appended at the end would be a system message following the
                # last AI turn, which reads as a new user request rather than
                # as an instruction about the whole turn.
                *_revision_instruction(state),
                *state.get("messages", []),
            ]
            next_step = responder(with_tools=True)

        ai_message = next_step.invoke(request)
        update: dict = {"iterations": state.get("iterations", 0) + 1}

        # PHASE 4 (Part C): a final answer becomes a *draft*, not a transcript
        # entry, when approval is required.
        #
        # An AIMessage with no tool_calls is this turn's answer. Without
        # approval it is appended and the run ends. With approval it must not be
        # appended yet - `messages` is the record of what was said, and nothing
        # has been said to the user until a human agrees to it. So it lands in
        # `draft`, and only finalize_answer commits it.
        #
        # Note this node keeps the draft as *text*. Safe here precisely because
        # there are no tool calls to preserve: an AIMessage carrying tool_calls
        # or Claude thinking blocks has to be replayed intact, but a plain final
        # answer is fully described by its text, and text is what a reviewer
        # edits.
        # PHASE 5: `review_enabled`, not `require_approval`. The draft handshake
        # is now needed whenever *anyone* is going to look at the answer - the
        # critic needs something to review just as much as the human does, and
        # an answer committed straight to `messages` is one no reviewer can
        # decline. Same mechanism, one more reason to use it.
        if review_enabled and not getattr(ai_message, "tool_calls", None):
            update["draft"] = ai_message.text
            update["status"] = "awaiting_approval"
            return update

        # The whole AIMessage is returned, not just its text: it carries the
        # tool-call blocks the next turn needs, and (with Claude) thinking blocks
        # that must be replayed unchanged. `add_messages` appends it.
        update["messages"] = [ai_message]
        if review_enabled:
            # The model asked for a tool, so nothing is pending review. Saying so
            # explicitly keeps `status` describing the current turn - see the
            # ReviewStatus note in state.py for the stale-status bug this avoids.
            update["status"] = "drafting"
        return update

    def call_tool(state: State) -> dict:
        """Run every tool the last AIMessage asked for.

        One AIMessage can request several tools at once, so this returns a list
        of ToolMessages and `add_messages` appends them all. Phase 1's inner
        `for call in ai_message.tool_calls` loop, moved into a node.
        """
        last = state["messages"][-1]

        # Defensive, and worth understanding: should_continue is the only edge
        # into this node and it only routes here when tool calls exist. But a
        # node is a plain function - a later refactor, a wrong path_map entry, or
        # a Studio session where you jump straight to this node can all reach it
        # with nothing to run. Returning an empty update is the safe answer;
        # indexing into `last.tool_calls` blindly would raise inside the node and
        # abort the whole run.
        if not isinstance(last, AIMessage) or not last.tool_calls:
            return {}

        return {
            "messages": [
                _execute_tool_call(call, tools_by_name) for call in last.tool_calls
            ]
        }

    # ------------------------------------------- Phase 4: human in the loop

    def review_draft(state: State) -> dict:
        """Pause the graph and wait for a verdict on the draft. (Part C)

        CONCEPT: interrupt()
        The call below stops the graph mid-run. LangGraph writes a checkpoint,
        and the `.invoke()` that is currently executing returns immediately with
        an `__interrupt__` key holding the payload passed here. Nothing raises,
        nothing blocks, and the process is free to exit - the parked run is a row
        in the checkpointer, not a suspended thread. Somebody can resume it an
        hour later from a different process.

        The payload is the *question to the human*, so it should carry
        everything a decision needs and nothing else. It is serialized into the
        checkpoint, so it follows State's rules: small, JSON-ish, no secrets.

        CONCEPT: the node re-runs on resume
        `interrupt()` does not suspend this function. On resume, LangGraph calls
        `review_draft` again from the top and, when execution reaches the
        `interrupt()` call, returns the stored resume value instead of pausing
        again. Everything above that line therefore executes twice.

        That constraint is the reason this node is as thin as it is. The draft
        was produced upstream by `call_model`; if it were produced here, resuming
        would re-call the model, pay for it again, and - because the model is
        not deterministic - hand the reviewer a verdict on text that no longer
        matches what they approved. Pre-interrupt work must be cheap and
        idempotent. Reading state is both.

        CONCEPT: interrupt() requires a checkpointer
        There is nowhere to park a run without one. Compiling with
        `checkpointer=None` and then reaching this node raises at runtime, not at
        compile time - build_graph refuses the combination up front instead.
        """
        draft = state.get("draft", "")

        verdict = interrupt(
            {
                "question": state.get("question", ""),
                "mode": state.get("mode", "live-search"),
                "draft": draft,
                "prompt": (
                    "Approve, reject, or edit this draft. Resume with "
                    "Command(resume={'decision': 'approve'|'reject'|'edit', "
                    "'text': '<edited answer, for edit>', "
                    "'note': '<why, for reject>'})"
                ),
            }
        )

        decision, text, note = _parse_verdict(verdict)

        if decision == "approve":
            return {"status": "approved"}

        if decision == "edit":
            # An edit is an approval of different words. The reviewer's text
            # replaces the draft and goes into the transcript as the answer -
            # the conversation records what was actually said to the user, not
            # what the model wanted to say.
            return {
                "status": "approved",
                "draft": text or draft,
                "human_feedback": note or "(edited by the reviewer)",
            }

        return {
            "status": "rejected",
            "human_feedback": note or "(rejected without a reason given)",
        }

    # ------------------------------------------- Phase 5: the machine reviewer

    def critique_draft(state: State) -> dict:
        """Ask a model whether the draft is good enough. (Part C)

        CONCEPT: this node is `review_draft` with a different reviewer.
        Put them side by side - that is the point of the phase:

            review_draft      critique_draft
            ----------------  --------------------------
            reads `draft`     reads `draft`
            asks a human      asks a model
            interrupt()       chain.invoke()
            _parse_verdict    _parse_critique -> _parse_verdict
            writes status +   writes status +
              human_feedback    critique
            -> should_revise  -> should_revise

        Everything that makes the graph work is in the identical rows. A
        reviewer is, structurally, just something that turns a draft into an
        approve/reject verdict; the graph routes on the verdict and is
        indifferent to where it came from. That indifference is what lets you
        swap a human for a model, add a second reviewer, or (Phase 6) hand the
        seat to an agent with its own tools - without touching an edge.

        The one asymmetry, and it is worth naming because it looks like it
        should matter more than it does: only the human pauses. `interrupt()`
        needs a checkpointer to park the run in, which is why
        `require_approval=True` without one is refused at build time.
        `enable_critic=True` needs nothing - a model call blocks and returns
        like any other function call, so the critic is just a slow node.

        CONCEPT: ordering, when both reviewers are on
        With `--critic --approve` this node runs first and the human is asked
        only about drafts the critic already passed. That choice is worth making
        explicitly rather than falling into:

          for critic-first  The critic is cheap, automatic and available at
                            3am; the human is none of those. Filtering with the
                            expendable reviewer before spending the scarce one
                            is the whole reason to have two. And it keeps the
                            human's word final: a human approves, the answer
                            ships. Human-first would mean a model overturning a
                            person's approval, which is not a review process
                            anybody wants to explain.

          against           The human never sees the drafts the critic rejected,
                            so a critic with bad taste silently narrows what
                            gets shown to a person - and there is no record in
                            the final transcript of the drafts that never made
                            it. Worse, with `--approve --critic` and a critic
                            that never approves, the revision cap is reached
                            before the interrupt is ever hit: the run ends with
                            a withheld draft and `--approve` looks like it did
                            nothing. `cli.py` prints a hint for exactly that.

        Human-first is a one-line change (swap the branch in `should_continue`
        and the `"awaiting_approval"` case in `should_revise`), so if the
        trade-off ever goes the other way it is cheap to move.
        """
        draft = state.get("draft", "")

        # CONCEPT: pre-interrupt discipline does not apply here, and that is a
        # real difference in kind.
        # `review_draft` has to keep everything above `interrupt()` cheap and
        # idempotent, because a resume re-runs the node from the top. This node
        # never interrupts, so its model call runs exactly once. The symmetry
        # between the two reviewers is in their interface, not in their
        # constraints - which is the sort of thing that only shows up when you
        # try to write them as one node and discover you cannot.
        chain = CRITIC_PROMPT | auxiliary("critic", critic_model) | StrOutputParser()
        raw = chain.invoke(
            {
                "question": state.get("question", ""),
                "draft": draft or "(the model produced an empty draft)",
                "sub_questions": "\n".join(
                    f"- {q}" for q in (state.get("sub_questions") or [])
                )
                or "(none)",
            }
        )

        decision, _text, note = _parse_verdict(_parse_critique(raw))

        if decision == "approve":
            # CONCEPT: an approval that is not the last word.
            # If a human is also in the loop, the critic passing the draft means
            # the draft is now ready for *them* - so the status goes back to
            # "awaiting_approval" and `should_revise` routes to `review_draft`.
            # No extra state and no "who reviewed last" field: the status
            # already says whether a decision is still owed, and that is exactly
            # the question the next hop needs answered.
            return {
                "status": "awaiting_approval" if require_approval else "approved",
                "critique": note or "(approved by the critic)",
            }

        # A critic has no "edit" verdict, deliberately. `_parse_critique` maps
        # anything that is not an approval to a rejection, so a critic that
        # tries to rewrite the answer is treated as a critic that rejected it
        # with a long note - and the note goes back to the writer, which is the
        # thing that is allowed to write. Letting a reviewer's text become the
        # answer is a power the human path grants a *person*, on purpose.
        return {
            "status": "rejected",
            "critique": note or "(rejected by the critic without a reason given)",
        }

    def start_revision(state: State) -> dict:
        """Begin one revision round. (Part A)

        CONCEPT: the node that resets the nested budget.
        This exists so that exactly one place is responsible for "a new revision
        is starting", and so that `iterations` gets reset there. `should_revise`
        could have routed straight back to `call_model` - and Part B describes
        the edge that way - but then `call_model` would have to work out for
        itself whether it was being re-entered for a new round, which it cannot
        do reliably: it is also re-entered by the tool loop, many times, within
        one round.

        A routing function must not have side effects (LangGraph may call it
        while working out the graph's shape), so the reset cannot live there
        either. A one-line node is the honest home for it.

        The reset itself is `revision_input`, defined beside `turn_input` at the
        bottom of this file, because it is the same pattern one level down:
        a nested loop gets a fresh budget when it starts a round, exactly as a
        turn gets a fresh budget when it starts a turn.
        """
        return revision_input(state)

    def finalize_answer(state: State) -> dict:
        """Commit the reviewed draft - or record that it was withheld.

        Kept separate from review_draft so that "the human decided" and "the
        transcript changed" are two steps you can see apart in a trace, and so
        that the only node that writes an answer into `messages` is one that
        never interrupts.
        """
        draft = state.get("draft", "")

        if state.get("status") == "approved":
            # Clearing `draft` matters once state persists: a draft left behind
            # is a draft the next turn's state dump shows as if it were pending.
            return {"messages": [AIMessage(content=draft)], "draft": ""}

        # Rejected. The answer is discarded, but *something* has to go into the
        # transcript, because the turn opened with a HumanMessage and leaving it
        # unanswered puts two human turns back to back - which the Anthropic API
        # rejects, since it requires alternating roles. Recording the refusal is
        # also simply true: the system did respond to that turn, by declining.
        #
        # PHASE 5: reaching here with status "rejected" now means something
        # narrower than it did in Phase 4. A rejection is no longer the end of
        # the road - `should_revise` sends it back to be revised - so the only
        # way a rejected draft arrives at this node is that the revision cap ran
        # out while the verdict was still "no". This is the cap overruling the
        # reviewer, and the message says so, because "withheld" with no number
        # beside it invites the wrong diagnosis: you would go looking for a
        # reviewer who hated the answer when what actually happened is that you
        # gave the loop two attempts and it needed three.
        #
        # Both reviewers' notes go in, labelled, for the same reason
        # `_revision_instruction` carries both: with `--critic --approve` the
        # transcript should record which of them objected.
        notes = []
        if state.get("critique"):
            notes.append(f"critic: {state['critique']}")
        if state.get("human_feedback"):
            notes.append(f"human: {state['human_feedback']}")
        reason = "; ".join(notes) or "(no reason recorded)"

        revisions = state.get("revisions", 0)
        spent = (
            f" after {revisions} revision{'s' if revisions != 1 else ''}"
            if revisions
            else ""
        )
        return {
            "messages": [
                AIMessage(content=f"(draft withheld{spent} - {reason})")
            ],
            "draft": "",
        }

    # -------------------------------------------------- routing (edge logic)

    def route_by_mode(state: State) -> str:
        """Entry branch: which research strategy does this run use?

        Phase 2 made this choice with an `if` in `cli.py`. Here it is a routing
        function, which is the same decision expressed as graph structure - so it
        shows up as a drawn branch in Studio, and as a recorded step in the trace.

        The branch is at the *entry*, not after call_model, because retrieval has
        to happen before the model speaks: the whole point of the knowledge-base
        path is that the model never answers ungrounded. `call_tool` is the
        mirror image - the model asks for a search, so that node necessarily runs
        after it. Same fork in the road, opposite sides of the model call.

        Phase 6's Supervisor generalizes exactly this function: same signature
        (state in, node name out), same `path_map` wiring, except the return
        value comes from a structured LLM call over `question` instead of from a
        CLI flag. The routing is already in the right place; only the decision
        maker changes.
        """
        mode = state.get("mode", "live-search")
        if mode == "knowledge-base":
            return "retrieve_docs"
        if mode == "live-search":
            return "call_model"
        # CONCEPT: cover every case, explicitly.
        # A routing function returning a key that isn't in the path_map fails at
        # runtime, mid-run, with LangGraph's own error - after you have already
        # paid for whatever ran before it. Raising here instead turns a typo like
        # mode="knowledge base" into an immediate, readable failure. The
        # alternative is a deliberate default (`return "call_model"`); what you
        # must not do is let an unlisted value fall through silently.
        raise ValueError(
            f"route_by_mode got an unknown mode {mode!r}; "
            f"expected 'knowledge-base' or 'live-search'"
        )

    def should_continue(state: State) -> str:
        """Loop branch: did the model ask for a tool, or is it done?

        This is Phase 1's `if not ai_message.tool_calls: return ...`, with one
        addition - the iteration cap, which Phase 1 kept in its `for` statement
        and the graph keeps in state.
        """
        messages = state.get("messages", [])
        last = messages[-1] if messages else None

        # PHASE 4 (Part C): the review branch, checked first and deliberately so.
        #
        # If call_model has drafted an answer, that draft is the run's entire
        # output and it is not in `messages` yet. Every other exit from here
        # goes to END, which would drop it on the floor - including the
        # iteration-cap exit below. A drafted answer always gets reviewed, even
        # if the tool budget ran out on the same step.
        #
        # PHASE 5: the same branch, now with a choice of reviewer. This one line
        # is where the critic-first ordering lives - swap the two branches and
        # the human goes first instead. The reasoning is at the top of
        # `critique_draft`.
        if review_enabled and state.get("status") == "awaiting_approval":
            return "critique_draft" if enable_critic else "review_draft"

        # The termination guard. Without it, a model that asks for a tool on
        # every turn makes call_model -> call_tool -> call_model cycle forever.
        # LangGraph has a backstop of its own (`recursion_limit`, 25 steps by
        # default, which raises GraphRecursionError), but that is a crash, not an
        # answer. Stopping here ends the run cleanly with whatever the model has
        # said so far - the same trade Phase 1 made with `stopped_early`.
        if state.get("iterations", 0) >= max_iterations:
            return "end"

        # Mode is checked here too, not only at the entry branch. It is tempting
        # to skip this: knowledge-base mode uses the model *without* tools bound,
        # so it should never produce tool_calls. But "should never" is a property
        # of the data, and this is an edge - the edge from call_model to call_tool
        # exists for every run that reaches call_model, whichever branch got it
        # there. A prompt-injected document, a model that hallucinates a
        # tool-call block, or a future change that binds tools in both modes
        # would all send a knowledge-base run into the live-search tool loop,
        # silently defeating the one guarantee that mode makes: answers come from
        # the ingested documents and nowhere else.
        #
        # This is the general lesson for Phase 6: a routing function is the only
        # thing standing between two paths. It has to enforce the separation
        # itself, not assume an earlier branch already did.
        if state.get("mode") == "knowledge-base":
            return "end"

        # The normal exit: an AIMessage with no tool_calls is a final answer.
        if isinstance(last, AIMessage) and last.tool_calls:
            return "call_tool"
        return "end"

    def should_revise(state: State) -> str:
        """Verdict branch: revise, hand on to the next reviewer, or finish.
        (Part B)

        This is the function both reviewers share - `review_draft` and
        `critique_draft` each route through it, with the same `path_map`. That
        is the Part C symmetry made structural: two nodes, one set of
        consequences, because a verdict is a verdict.

        The three outcomes:

          rejected, budget left   -> start_revision (which resets `iterations`
                                     and bumps `revisions`, then re-enters
                                     call_model with the feedback in state)
          awaiting_approval       -> review_draft. Only reachable from
                                     `critique_draft`: the critic approved and a
                                     human still owes a verdict. `review_draft`
                                     itself only ever writes "approved" or
                                     "rejected", so it can never produce this
                                     status - which is why its path_map omits
                                     the key. Same function, different declared
                                     destinations; see the wiring for why that
                                     is not a second copy of the logic.
          anything else           -> finalize_answer

        CONCEPT: why the cap has to be able to overrule the verdict
        The rejection branch checks `revisions` *before* it trusts "reject", and
        that ordering is the substance of this function rather than a guard
        bolted onto it.

        A reflection loop's exit condition is supplied by the thing the loop is
        meant to be checking. "Keep revising until the reviewer approves" is
        only a terminating condition if the reviewer is guaranteed to approve
        eventually, and nothing guarantees that. A critic prompted to find fault
        will find fault - there is always another caveat to want - and a human
        reviewer can be unavailable, unreasonable, or simply wrong about what
        the model is capable of. Neither can emit a verdict meaning "and I
        promise to stop asking", because that promise is not the kind of thing a
        judgement contains.

        So the loop counts its own attempts and stops on its own authority. The
        division of labour is: the verdict decides whether the answer is *good*;
        the cap decides when we are done *spending*. A loop that delegates both
        to the same judge has no termination condition, only a hope.

        The cost of the override is real and is paid in `finalize_answer`: the
        run ends with a draft nobody approved, recorded as withheld with the
        revision count attached. That is the right failure - it is visible, it
        is bounded, and it says which of the two limits was hit. LangGraph's
        `recursion_limit` would also eventually stop this loop, but as a
        `GraphRecursionError` with no answer and no explanation, which is the
        same trade `should_continue` already refuses for the tool loop.

        CONCEPT: "route to END" means routing to the node that ends cleanly
        The exhausted branch goes to `finalize_answer`, not to `END` directly.
        `finalize_answer` is the only node that writes the turn's outcome into
        `messages`, and the turn opened with a HumanMessage: ending without it
        leaves two human turns adjacent, which the Anthropic API rejects
        outright. "Stop looping" and "leave the transcript valid" are two
        requirements, and the second one has a node.
        """
        status = state.get("status")

        if status == "rejected":
            if state.get("revisions", 0) >= max_revisions:
                return "finalize_answer"
            return "start_revision"

        if status == "awaiting_approval":
            return "review_draft"

        # "approved", and - failing closed - anything unexpected. An unknown
        # status must not silently start a revision loop; finalizing is the
        # bounded choice, and `finalize_answer` only commits text when the
        # status is exactly "approved", so an unrecognized value withholds.
        return "finalize_answer"

    # ------------------------------------------------------------------ wiring

    if require_approval and checkpointer is None:
        # Fail here rather than at runtime. `interrupt()` has nowhere to park a
        # run without a checkpointer, so this combination is always a mistake -
        # and catching it at build time means the error arrives before any model
        # call has been paid for, with a message that says what to do.
        raise RuntimeError(
            "require_approval=True needs a checkpointer: interrupt() parks the "
            "run in one. Pass checkpointer=... (see checkpointing.py) or use "
            "--checkpointer memory/sqlite."
        )

    builder = StateGraph(State)

    builder.add_node("prune_history", prune_history)
    builder.add_node("retrieve_docs", retrieve_docs)
    builder.add_node("call_model", call_model)
    builder.add_node("call_tool", call_tool)
    # CONCEPT: one graph shape, whatever the flags say.
    # These two are registered even when require_approval is False, in which
    # case should_continue never routes to them and they simply never run
    # (compile checks edge targets, not reachability - see the compile() note
    # above). The alternative - building a different graph per flag - means the
    # structure of a thread changes when you toggle `--approve`, and a
    # checkpoint written by one shape is being resumed by another. Same nodes,
    # same names, every time: the flag changes which paths are taken, not which
    # paths exist.
    builder.add_node("review_draft", review_draft)
    builder.add_node("finalize_answer", finalize_answer)
    # PHASE 5, and registered unconditionally for the same reason the review
    # nodes are: `--critic` and `--plan` change which paths are taken, not which
    # paths exist. `plan_question` is on the unconditional path and returns {}
    # when planning is off; the other two are simply unreachable.
    builder.add_node("plan_question", plan_question)
    builder.add_node("critique_draft", critique_draft)
    builder.add_node("start_revision", start_revision)

    # PHASE 4: pruning is the entry point, so the transcript is brought inside
    # its budget before anything reads it - including retrieve_docs, and
    # including the first call_model of the turn. In Phase 3 this edge ran
    # straight from START into the mode branch.
    builder.add_edge(START, "prune_history")

    # PHASE 5: planning sits between pruning and the mode branch. Before the
    # branch because both modes want the plan; after pruning because a planner
    # that reads a transcript should read the one that is going to be sent.
    builder.add_edge("prune_history", "plan_question")

    # The entry branch. Passing a path_map (the dict) rather than letting the
    # function's return value name the node directly is what lets LangGraph know
    # the full set of destinations *without running anything* - which is how
    # Studio can draw both arrows before the first token. Without it, the drawn
    # graph shows a branch into the unknown.
    builder.add_conditional_edges(
        "plan_question",
        route_by_mode,
        {"retrieve_docs": "retrieve_docs", "call_model": "call_model"},
    )

    # Unconditional: retrieval always feeds the model.
    builder.add_edge("retrieve_docs", "call_model")

    # The loop branch, now three-way: keep looping, stop for review, or finish.
    builder.add_conditional_edges(
        "call_model",
        should_continue,
        {
            "call_tool": "call_tool",
            "critique_draft": "critique_draft",
            "review_draft": "review_draft",
            "end": END,
        },
    )

    # The edge that closes the cycle. A graph is allowed to contain cycles -
    # that is the main thing a StateGraph gives you that an LCEL chain cannot.
    # The cycle is safe only because should_continue can leave it.
    builder.add_edge("call_tool", "call_model")

    # PHASE 5 (Part B): the review path is no longer a straight line to
    # `finalize_answer`. In Phase 4 a verdict had exactly one consequence, so an
    # unconditional edge was right; now a rejection can mean "go round again"
    # and the edge has to branch.
    #
    # Both reviewers get the *same* routing function. That is the Part C
    # symmetry in the wiring rather than only in a comment: nothing downstream
    # of a verdict knows or cares which node produced it, so one function
    # decides the consequences of both.
    #
    # Their `path_map`s differ, and the difference is worth understanding because
    # it is exactly what a path_map is for. It is not a second copy of the
    # routing logic - it is the declared set of destinations reachable *from
    # this source*, and LangGraph uses it to draw the graph before anything
    # runs. `should_revise` can return "review_draft" only for a draft the
    # critic just approved while a human still owes a verdict, which
    # `review_draft` itself can never produce (it only ever writes "approved" or
    # "rejected"). Listing it there anyway would draw a review_draft -> itself
    # self-loop in Studio that no run can ever take, and a drawing that shows
    # impossible paths is worth less than one that does not.
    #
    # The rule: share the routing function, declare each source's real
    # destinations. Adding a third reviewer is one `add_node` and one of these
    # calls with whatever that node can actually reach.
    builder.add_conditional_edges(
        "critique_draft",
        should_revise,
        {
            "start_revision": "start_revision",
            "review_draft": "review_draft",
            "finalize_answer": "finalize_answer",
        },
    )
    builder.add_conditional_edges(
        "review_draft",
        should_revise,
        {
            "start_revision": "start_revision",
            "finalize_answer": "finalize_answer",
        },
    )

    # The reflection loop's back edge, and the second cycle in this graph. Same
    # bargain as `call_tool -> call_model`: a cycle is safe only because the
    # routing function that enters it can also leave it, and here that is
    # `should_revise` checking `revisions` against `max_revisions`.
    builder.add_edge("start_revision", "call_model")

    builder.add_edge("finalize_answer", END)

    # PHASE 4: the one argument that makes state outlive the call.
    # None reproduces Phase 3 exactly. A saver makes every super-step write a
    # snapshot under the thread_id in `config`, which is what turns `messages`
    # into a conversation, and what gives `interrupt()` somewhere to park.
    return builder.compile(checkpointer=checkpointer, name="research-copilot")


VERDICTS = ("approve", "reject", "edit")


def _parse_verdict(value: Any) -> tuple[str, str, str]:
    """Normalize whatever came back through `Command(resume=...)`.

    The resume value is entirely the caller's choice - LangGraph passes it
    through untouched, so `interrupt()` can return a string, a dict, or anything
    else that survived being written to the checkpoint. That flexibility is
    convenient and it means a node cannot trust the shape, so the parsing lives
    in one place with one set of rules:

        "approve"                                  -> ("approve", "", "")
        {"decision": "edit", "text": "..."}        -> ("edit", "...", "")
        {"decision": "reject", "note": "too thin"} -> ("reject", "", "too thin")

    Anything unrecognized is treated as a rejection. Failing closed is the only
    safe default for an approval gate: an unparseable verdict must never be read
    as consent, and a mistaken rejection costs a retry while a mistaken approval
    publishes something nobody agreed to.
    """
    if isinstance(value, str):
        decision, text, note = value.strip().lower(), "", ""
    elif isinstance(value, dict):
        decision = str(value.get("decision", "")).strip().lower()
        text = str(value.get("text", "") or "")
        note = str(value.get("note", "") or "")
    else:
        decision, text, note = "", "", ""

    if decision not in VERDICTS:
        return (
            "reject",
            "",
            note or f"unrecognized verdict {value!r}; treated as a rejection",
        )
    return decision, text, note


def _parse_critique(raw: str) -> dict:
    """Turn the critic's reply into the verdict dict the human path produces.

    CONCEPT: one verdict shape, two reviewers.
    The output of this function is fed straight to `_parse_verdict` - the same
    parser that reads what a human typed at `research-copilot review`. That is
    not tidiness for its own sake: it means the critic inherits the fail-closed
    rule for free, and it means there is exactly one definition of what a
    verdict is. A second parser would be a second place for "approve" to mean
    something slightly different.

        "APPROVE"                    -> {"decision": "approve", "note": ""}
        "REJECT\nno sources for X"   -> {"decision": "reject", "note": "no ..."}
        "Looks good to me!"          -> {"decision": "", ...} -> reject

    Failing closed here costs a revision round rather than a wrongly published
    answer, which is a different trade from the human gate's - and `max_revisions`
    is what bounds it. Worth being clear-eyed about: a critic whose output
    format drifts turns into a critic that always rejects, and the symptom is a
    run that always exhausts its revisions. The `note` carries the raw text so
    that case is diagnosable from the state dump rather than mysterious.
    """
    text = (raw or "").strip()
    if not text:
        return {"decision": "", "note": "the critic returned nothing"}

    first, _, rest = text.partition("\n")
    # Tolerate the common decorations a model adds to a keyword on its own line -
    # "**APPROVE**", "APPROVE.", "REJECT:" - without tolerating a whole sentence
    # that merely contains the word. A critic that writes "I would not APPROVE
    # this" must not be read as an approval, so the line has to *be* the verdict
    # once punctuation and emphasis are stripped, not contain it.
    verdict = first.strip().strip("*_`#").strip(" .:;-").lower()

    if verdict == "approve":
        return {"decision": "approve", "note": rest.strip()}
    if verdict == "reject":
        return {"decision": "reject", "note": rest.strip()}

    # Unrecognized. Hand the whole reply through as the note so the writer still
    # gets whatever the critic actually said, and let `_parse_verdict` decide
    # (it rejects).
    return {"decision": "", "note": text}


def _parse_plan(raw: str, max_sub_questions: int) -> list[str]:
    """Turn the planner's reply into a list of sub-questions, possibly empty.

    "NONE" means the question is best researched whole, and so does anything
    this cannot make sense of. That default is the opposite of `_parse_critique`'s
    and deliberately so: failing closed means failing towards *doing less*, and
    for a planner "less" is no decomposition. An unreadable plan should cost
    nothing and leave the run behaving exactly as Phase 4 did - not invent
    sub-questions that then drive retrieval and tool calls.
    """
    text = (raw or "").strip()
    if not text or text.strip().strip("*_`.").upper() == "NONE":
        return []

    questions = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Accept the "- " the prompt asks for, and the numbering a model adds
        # anyway. A line with no marker at all is prose - a preamble like "Here
        # are the sub-questions:" - and is dropped rather than searched for.
        stripped = line.lstrip("-*•").strip()
        if stripped == line:
            stripped = line.lstrip("0123456789").lstrip(").: ").strip()
            if stripped == line:
                continue
        if stripped:
            questions.append(stripped)

    return questions[:max_sub_questions]


def turn_input(question: str, mode: Mode = "live-search") -> dict:
    """The state update that starts one turn.

    Phase 3 could inline this; Phase 4 cannot, because with a checkpointer the
    turn does not start from `{}`. It starts from whatever the last turn left in
    the thread, and this dict is merged *onto* that. Which makes the keys that
    are reset here as important as the keys that are set:

      messages    appended (the `add_messages` reducer), so the new question
                  joins the existing transcript rather than replacing it
      question    overwritten - this turn's question, not the thread's first
      mode        overwritten; a thread may switch modes between turns
      iterations  RESET TO 0, and this is the subtle one. It is a per-turn tool
                  budget, but a checkpointed counter never returns to zero on its
                  own: turn 2 would start at 2, turn 4 at 6, and should_continue
                  would start refusing tool calls for work done in earlier turns.
                  Phase 3 got this for free because every run began empty.
      draft       cleared, so a state dump never shows a stale draft as pending
      status      reset to "drafting" - see the ReviewStatus note in state.py
                  for the misrouting this prevents

      revisions       RESET TO 0 (Phase 5), for the same reason as `iterations`
                      and one loop up: it is a per-turn budget for the
                      reflection loop, so a checkpointed thread would otherwise
                      arrive at turn 3 with no revisions left.
      critique,       CLEARED (Phase 5). These are reviewer notes about *this*
      human_feedback  turn's draft, and `call_model` reads them as a revision
                      instruction whenever `revisions > 0`. Left behind, they
                      would be the last turn's complaint attached to a question
                      it was never about. Phase 4 did not clear `human_feedback`
                      because nothing read it back; Phase 5 does, which is the
                      general lesson - a field becomes a lifecycle problem the
                      moment something downstream consumes it.
      sub_questions   CLEARED (Phase 5): a plan is made per question.

    `summary` is deliberately *not* reset: it is the compressed remainder of
    this thread's history and it has to survive the turn boundary, exactly like
    `messages`.
    """
    return {
        "question": question,
        "mode": mode,
        # The transcript starts with the user's turn. In live-search mode the
        # loop appends to it; in knowledge-base mode call_model answers it
        # from the retrieved context.
        "messages": [HumanMessage(content=question)],
        "iterations": 0,
        "draft": "",
        "status": "drafting",
        # --- Phase 5 ---
        "revisions": 0,
        "critique": "",
        "human_feedback": "",
        "sub_questions": [],
    }


def revision_input(state: State) -> dict:
    """The state update that starts one revision round. (Part A)

    `turn_input` one level down, and written next to it on purpose: same job,
    same shape, different scope. A turn resets the per-turn budgets; a revision
    round resets the per-round ones.

      revisions   incremented - this is the counter `should_revise` caps, and
                  `call_model` reads it as "am I revising?"
      iterations  RESET TO 0, which is the whole reason this function exists.
                  The tool loop's budget is per *round*, not per turn: a
                  revision that has to search again should get a full budget to
                  do it in, and a revision penalised for the searches an earlier
                  draft made would make the agent quietly worse the harder it
                  tries. This is the concrete form of the argument in state.py
                  for keeping the two counters apart - two counters are only
                  actually separate if they are reset on separate schedules.
      status      back to "drafting", so a stale "rejected" cannot send the run
                  round again on a draft that no longer exists. Same reasoning
                  as `turn_input` resetting it; see ReviewStatus in state.py.

    Deliberately *not* touched:

      draft            this is the text being revised. `call_model` shows it to
                       the model as "your previous draft" and then overwrites
                       it. `status == "drafting"` is what says it is no longer
                       pending, which is exactly the job that field was added
                       for.
      critique,        the revision instruction. Clearing them here would route
      human_feedback   the run back to the writer having just deleted the
                       reason it was sent back.
      messages         a revision is another attempt at the same turn, not a new
                       turn. Nothing about the rejected draft enters the
                       transcript - the same argument that made `draft` a
                       separate key in Phase 4: `messages` records what was
                       said, and a rejected draft was never said.
    """
    return {
        "revisions": state.get("revisions", 0) + 1,
        "iterations": 0,
        "status": "drafting",
    }


def run_graph(
    question: str,
    *,
    mode: Mode = "live-search",
    graph: Runnable | None = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    thread_id: str | None = None,
    config: dict | None = None,
    **build_kwargs,
) -> State:
    """Invoke the graph on one question and return the final state.

    The input is a partial State. Anything left out (`documents`, `context`)
    simply isn't there yet; nodes fill it in as they run, which is why State is
    declared `total=False`.

    `.invoke()` returns the *final* state, not just the answer - every key every
    node wrote. `.stream()` on the same graph would yield one update per node
    instead, which is what Studio consumes to animate execution.

    PHASE 4: `thread_id` is the second argument to `.invoke()`, never part of
    the first. The state dict says what this turn is about; the config says
    which conversation it belongs to. Passing no thread_id runs the graph the
    Phase 3 way - fine for a graph with no checkpointer, and with one it means
    every run is anonymous and nothing accumulates.

    If the returned state contains `__interrupt__`, the graph did not finish: it
    is parked at `review_draft` waiting for `resume_graph`.
    """
    graph = graph or build_graph(max_iterations=max_iterations, **build_kwargs)
    if config is None and thread_id is not None:
        config = thread_config(thread_id)
    return graph.invoke(turn_input(question, mode), config)


def resume_graph(
    graph: Runnable,
    verdict: Any,
    *,
    thread_id: str | None = None,
    config: dict | None = None,
) -> State:
    """Restart a graph parked at `interrupt()`, carrying the human's verdict.

    CONCEPT: Command
    `Command(resume=v)` goes where a state dict normally goes. That substitution
    is the whole API: the first argument to `.invoke()` is either "here is new
    input, start a turn" or "here is a control instruction, continue the parked
    one". A `Command` is not merged into State and no reducer sees it; LangGraph
    reads it, finds the interrupted task in the checkpoint, and makes `v` the
    return value of the `interrupt()` call that stopped it.

    The config is not optional in practice. Resume means "continue *that*
    conversation", and without a thread_id there is no conversation to name - a
    resume against the wrong thread_id, or a thread with nothing parked, does not
    raise. It reads the snapshot, finds no interrupted task, and quietly runs
    nothing, handing back the state as it already was. Callers therefore check
    `pending_interrupt` first; `cli.py` does exactly that.
    """
    if config is None:
        if thread_id is None:
            raise RuntimeError(
                "resume_graph needs a thread_id: a resume continues one "
                "specific parked conversation, and config is how it is named."
            )
        config = thread_config(thread_id)
    return graph.invoke(Command(resume=verdict), config)


def pending_interrupt(graph: Runnable, *, thread_id: str) -> dict | None:
    """The payload a parked run is waiting on, or None if nothing is parked.

    CONCEPT: reading a thread without running it
    `graph.get_state(config)` returns a `StateSnapshot` - the stored values, the
    `next` nodes, and the tasks with their pending interrupts - without
    executing anything. This is how a fresh process discovers that a thread owes
    somebody a decision, which is the whole premise of resuming from a different
    invocation than the one that paused.

    An unknown thread_id is not an error here either: `get_state` returns an
    empty snapshot with `values == {}` and no tasks, which reads as "nothing
    pending". `cli.py` distinguishes the two cases explicitly, because "this
    thread has no draft waiting" and "this thread does not exist" deserve
    different messages.
    """
    snapshot = graph.get_state(thread_config(thread_id))
    interrupts = getattr(snapshot, "interrupts", ())
    if not interrupts:
        return None
    return interrupts[0].value


def final_answer(state: State) -> str:
    """The text of the last AIMessage, which is the run's answer.

    The graph has no `answer` key on purpose: the answer is already in
    `messages`, and a second copy is a second thing to keep correct. This reads
    it back out.
    """
    for message in reversed(state.get("messages", [])):
        if isinstance(message, AIMessage) and message.text:
            return message.text
    return "(the graph produced no answer)"


def make_graph(config: dict | None = None) -> Runnable:
    """Factory for the LangGraph dev server / Studio (see langgraph.json).

    langgraph.json can point at either a compiled graph object or a function
    that returns one. A function is the better choice here: the server imports
    this module at startup and calls this, so the graph is built once the server
    is up rather than at import time. `config` is accepted (and ignored) because
    the CLI may pass its run config to the factory.
    """
    return build_graph()


# --------------------------------------------------------------------------
# PHASE 6 NOTE: what the Critic becomes when it is an agent
# --------------------------------------------------------------------------
# Written here rather than in a plan document because the seam is in this file,
# and because the thing to notice is how little of it is new.
#
# `critique_draft` is already a Critic. It reads a draft, forms a judgement, and
# the graph routes on that judgement without knowing or caring who produced it.
# Phase 6 changes the *inside* of the node, not its position: the Critic gets
# its own system prompt, its own tools (so it can go and check a citation rather
# than only doubting it), and possibly its own several-step loop. The verdict it
# returns and the edge it returns it on are unchanged, which is the payoff for
# having built the human gate and the machine gate to the same shape.
#
# What genuinely has to change is on the other side of the loop. Right now
# `call_model` both drafts and revises - `should_revise` routes a rejection back
# to the same node, and `_revision_instruction` switches its behaviour by
# reading `revisions`. That works because there is one model doing one job with
# two preambles. Split it into a Researcher, a Writer, and a Critic and three
# things stop being free:
#
#   1. A revision has to be routed to *somebody*. "Back to call_model" becomes a
#      real decision: a critique about a missing source belongs to the
#      Researcher, one about structure belongs to the Writer, and something has
#      to read the critique to tell them apart. That reader is the Supervisor,
#      and it is why Phase 6's routing comes from structured LLM output rather
#      than from a field - `route_by_mode`'s docstring already flags that it is
#      the same function with a different decision maker.
#
#   2. The counters multiply again, for exactly the reason `iterations` and
#      `revisions` had to be split. Each agent has its own internal loop, and a
#      Researcher that burns its tool budget inside revision 2 must not be
#      charged for what it spent in revision 1. The rule generalizes: one
#      counter per loop, reset by whoever begins a pass of that loop. Phase 6
#      will want per-agent iteration budgets nested inside `revisions`, which
#      means a third reset site and probably a small dataclass rather than three
#      more flat int keys.
#
#   3. `draft` stops being one field. With separate agents there is research
#      output, a written draft, and a critique of it, and they are produced by
#      different nodes at different times - so "the current draft" needs an
#      owner, or the Writer and the Researcher will overwrite each other. This
#      is where LangGraph's subgraphs and per-node state schemas start earning
#      their complexity, and where `total=False` plus `state.get(...)` stops
#      being sufficient discipline on its own.
#
# The short version: the Critic ports over almost unchanged, and the drafting
# side is where the work is. A reviewer only needs to produce a verdict; a
# writer that has been told it is wrong needs to know which of several agents
# should act on that, and with whose budget.
