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

The shape, with Phase 4's three new nodes marked (*):

    START ─→ prune_history(*) ──route_by_mode──┬─ "knowledge-base" ─→ retrieve_docs ─┐
                                               └─ "live-search" ───────────────────┐ │
                                                                                   │ │
                                       ┌───────────────────────────────────────────┴─┘
                                       ↓
                                   call_model ──should_continue──┬─ "call_tool" ─→ call_tool ┐
                                       ↑                          │                          │
                                       └──────────────────────────┼──────────────────────────┘
                                                                  ├─ "review_draft" ─→ review_draft(*)
                                                                  │                        ↓  (interrupt)
                                                                  │                   finalize_answer(*)
                                                                  │                        ↓
                                                                  └─ "end" ──────────────→ END
"""

from collections.abc import Sequence
from typing import Any

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
from research_copilot.prompts import AGENT_SYSTEM_PROMPT, RAG_PROMPT, SUMMARY_PROMPT
from research_copilot.retrieval import format_docs, get_retriever
from research_copilot.state import Mode, ReviewStatus, State
from research_copilot.tools import search_arxiv

# The same default as Phase 1's `run_tool_loop`.
DEFAULT_MAX_ITERATIONS = 6

# How many recent messages `prune_history` keeps verbatim before it will consider
# summarizing. Mirrors ConversationMemory.keep_last_messages.
DEFAULT_KEEP_LAST_MESSAGES = 4


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
        documents = active_retriever.invoke(state["question"])
        return {"documents": documents, "context": format_docs(documents)}

    def call_model(state: State) -> dict:
        """Ask the model for the next step: a tool call, or a final answer.

        This is Phase 1's `model_with_tools.invoke(messages)` line, plus the
        choice of which system prompt and which model variant the mode calls for.
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
        if require_approval and not getattr(ai_message, "tool_calls", None):
            update["draft"] = ai_message.text
            update["status"] = "awaiting_approval"
            return update

        # The whole AIMessage is returned, not just its text: it carries the
        # tool-call blocks the next turn needs, and (with Claude) thinking blocks
        # that must be replayed unchanged. `add_messages` appends it.
        update["messages"] = [ai_message]
        if require_approval:
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
        # Phase 5 has the better answer and this is the seam for it: instead of
        # an edge to END, a rejection becomes an edge back to call_model with
        # `human_feedback` as the revision instruction and `revisions` counting
        # the attempts (see the Phase 5 note in state.py).
        note = state.get("human_feedback", "")
        return {
            "messages": [
                AIMessage(content=f"(draft withheld - rejected by reviewer: {note})")
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
        if require_approval and state.get("status") == "awaiting_approval":
            return "review_draft"

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

    # PHASE 4: pruning is the entry point, so the transcript is brought inside
    # its budget before anything reads it - including retrieve_docs, and
    # including the first call_model of the turn. In Phase 3 this edge ran
    # straight from START into the mode branch.
    builder.add_edge(START, "prune_history")

    # The entry branch. Passing a path_map (the dict) rather than letting the
    # function's return value name the node directly is what lets LangGraph know
    # the full set of destinations *without running anything* - which is how
    # Studio can draw both arrows before the first token. Without it, the drawn
    # graph shows a branch into the unknown.
    builder.add_conditional_edges(
        "prune_history",
        route_by_mode,
        {"retrieve_docs": "retrieve_docs", "call_model": "call_model"},
    )

    # Unconditional: retrieval always feeds the model.
    builder.add_edge("retrieve_docs", "call_model")

    # The loop branch, now three-way: keep looping, stop for review, or finish.
    builder.add_conditional_edges(
        "call_model",
        should_continue,
        {"call_tool": "call_tool", "review_draft": "review_draft", "end": END},
    )

    # The edge that closes the cycle. A graph is allowed to contain cycles -
    # that is the main thing a StateGraph gives you that an LCEL chain cannot.
    # The cycle is safe only because should_continue can leave it.
    builder.add_edge("call_tool", "call_model")

    # The review path. Unconditional: once a human has given a verdict, that
    # verdict is always acted on. `review_draft` decided *what* happens
    # (approved text vs. a withheld note); `finalize_answer` only carries it out.
    builder.add_edge("review_draft", "finalize_answer")
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
