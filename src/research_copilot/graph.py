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
`builder.compile()` validates the graph (every node reachable, every edge target
real) and returns a `CompiledStateGraph`, which is a Runnable like everything
else in the project: `.invoke()`, `.stream()`, `.batch()`. Building and
compiling are separate so the structure can be checked - and drawn - before any
model is called. Compile takes a `checkpointer` that makes state persist between
invocations; that is Phase 4, so nothing is passed here.
"""

from collections.abc import Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph

# Reused verbatim from Phase 1 so that the manual loop and the graph execute
# tools through identical code. Every behavioural difference between
# `agent --structured` and `graph-agent` is then a difference in control flow,
# not in tool handling. (LangGraph ships `langgraph.prebuilt.ToolNode`, which is
# this node already written - see prebuilt.py for what leaning on it costs you.)
from research_copilot.agent_loop import _execute_tool_call
from research_copilot.models import get_chat_model
from research_copilot.prompts import AGENT_SYSTEM_PROMPT, RAG_PROMPT
from research_copilot.retrieval import format_docs, get_retriever
from research_copilot.state import Mode, State
from research_copilot.tools import search_arxiv

# The same default as Phase 1's `run_tool_loop`.
DEFAULT_MAX_ITERATIONS = 6


def build_graph(
    *,
    model: BaseChatModel | None = None,
    tools: Sequence[BaseTool] | None = None,
    retriever: BaseRetriever | None = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> Runnable:
    """Wire up and compile the graph.

    Everything injectable is an argument, for the same reason Phase 2's
    `build_rag_chain` took a model and a retriever: the tests swap in fakes and
    run the whole graph offline.

    The nodes are defined inside this function as closures. A node's signature
    is fixed - LangGraph calls it with the state - so anything else it needs
    (the model, the tool table, the retriever) has to be captured, not passed.
    """
    tools = list(tools) if tools is not None else [search_arxiv]
    tools_by_name = {tool.name: tool for tool in tools}

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
            request = [SystemMessage(content=AGENT_SYSTEM_PROMPT), *state["messages"]]
            next_step = responder(with_tools=True)

        ai_message = next_step.invoke(request)

        # The whole AIMessage is returned, not just its text: it carries the
        # tool-call blocks the next turn needs, and (with Claude) thinking blocks
        # that must be replayed unchanged. `add_messages` appends it.
        return {
            "messages": [ai_message],
            "iterations": state.get("iterations", 0) + 1,
        }

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
        last = state["messages"][-1]

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

    builder = StateGraph(State)

    builder.add_node("retrieve_docs", retrieve_docs)
    builder.add_node("call_model", call_model)
    builder.add_node("call_tool", call_tool)

    # The entry branch. Passing a path_map (the dict) rather than letting the
    # function's return value name the node directly is what lets LangGraph know
    # the full set of destinations *without running anything* - which is how
    # Studio can draw both arrows before the first token. Without it, the drawn
    # graph shows a branch into the unknown.
    builder.add_conditional_edges(
        START,
        route_by_mode,
        {"retrieve_docs": "retrieve_docs", "call_model": "call_model"},
    )

    # Unconditional: retrieval always feeds the model.
    builder.add_edge("retrieve_docs", "call_model")

    # The loop branch.
    builder.add_conditional_edges(
        "call_model",
        should_continue,
        {"call_tool": "call_tool", "end": END},
    )

    # The edge that closes the cycle. A graph is allowed to contain cycles -
    # that is the main thing a StateGraph gives you that an LCEL chain cannot.
    # The cycle is safe only because should_continue can leave it.
    builder.add_edge("call_tool", "call_model")

    # No checkpointer: state lives for one .invoke() and is discarded. Phase 4
    # passes `checkpointer=...` here, and the same graph gains memory across
    # calls plus the ability to pause mid-run for human approval.
    return builder.compile(name="research-copilot")


def run_graph(
    question: str,
    *,
    mode: Mode = "live-search",
    graph: Runnable | None = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    **build_kwargs,
) -> State:
    """Invoke the graph on one question and return the final state.

    The input is a partial State. Anything left out (`documents`, `context`,
    `iterations`) simply isn't there yet; nodes fill it in as they run, which is
    why State is declared `total=False`.

    `.invoke()` returns the *final* state, not just the answer - every key every
    node wrote. `.stream()` on the same graph would yield one update per node
    instead, which is what Studio consumes to animate execution.
    """
    graph = graph or build_graph(max_iterations=max_iterations, **build_kwargs)
    return graph.invoke(
        {
            "question": question,
            "mode": mode,
            # The transcript starts with the user's turn. In live-search mode the
            # loop appends to it; in knowledge-base mode call_model answers it
            # from the retrieved context.
            "messages": [HumanMessage(content=question)],
        }
    )


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
