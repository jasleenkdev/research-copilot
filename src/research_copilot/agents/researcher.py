"""The Researcher agent: find the evidence, write it up, hand it on.

This agent takes over the *first half* of Phase 5's `call_model`, the half that
decided what to look up and looked it up. It does not write the answer. Its
output is `research_notes`, and the Writer (agents/writer.py) turns those notes
into prose.

--------------------------------------------------------------------------
CONCEPT: an agent that is a subgraph
--------------------------------------------------------------------------
The Researcher is built here as its own compiled StateGraph, and that compiled
graph is added to the parent graph as a single node:

    parent.add_node("researcher", build_researcher(...))

From the parent's side it is one node. It takes state in, returns an update,
and the next edge fires. Inside, it is a graph of its own, with its own loop:

    START ──route_research──┬─ "knowledge-base" ─→ retrieve ─────────────────────────→ END
                            └─ "live-search" ───→ research_model ──should_search──┬─ research_tools ─┐
                                                        ↑                         │                  │
                                                        └─────────────────────────┼──────────────────┘
                                                                                  └─ compile_notes ─→ END

Why a subgraph and not a node? Because the Researcher's job really is several
steps, and the steps depend on each other. Each tool call is decided after
reading the previous result. That loop is exactly the call_model <-> call_tool
cycle from Phase 3, and it needs what that cycle needed: its own message list
and its own iteration counter.

A single node *could* run that loop internally, with a Python `while` inside
the node function. That is Phase 1's `agent_loop.py`, and it has the cost
Phase 3 was built to remove. The loop becomes invisible. Studio shows one box,
LangSmith shows one step, and nothing between the first search and the last is
inspectable, interruptible, or checkpointed. A subgraph keeps each research
step a real graph step, while the parent graph still sees one agent.

--------------------------------------------------------------------------
CONCEPT: the private message channel
--------------------------------------------------------------------------
The tool loop's messages - AIMessage(tool_calls=[...]), ToolMessage(result),
AIMessage(tool_calls=[...]), ... - go into `research_messages`, a key that
exists only in `ResearcherState`. It is not in the parent's `MultiAgentState`
and not in `ResearcherOutput`, so when the subgraph finishes, LangGraph drops
it. It never reaches the parent's state, so no other agent ever sees it, and
`graph.get_state(config)` on the thread does not show it.

"Private" means private *from the other agents*, not private from the disk.
With a checkpointer attached, the subgraph writes checkpoints of its own,
under a separate namespace (`checkpoint_ns = "researcher:<task-id>"`), and
those do contain `research_messages`: every search and every raw result.
Phase 4's warning applies again, one level down. Keeping a value out of the
state you read is not the same as keeping it off the disk. If a tool result
must never be persisted, the fix is to keep it out of the checkpointer, not to
give it a private key. `test_private_channel_is_hidden_from_state_not_from_disk`
pins down both halves.

This is the thing Phase 5 could not do. There, the tool loop wrote into the
shared `messages`, which was fine with one model: the searches *were* part of
its conversation. With several agents, the shared transcript is read by all of
them. If the Researcher's working-out landed there:

  - the Writer would see raw tool-call blocks and tool results in its history,
    and would be tempted to cite them directly instead of the notes. The notes
    are the Researcher's actual judgement of what mattered.
  - a follow-up turn would carry every search from every earlier turn in its
    transcript, at full token cost. This is the unbounded growth Phase 4's
    pruning was built for, now caused by an agent's scratch work.
  - the transcript would stop being a record of the *conversation*. The user
    asked a question and got an answer. The fourteen searches in between were
    how one agent did its job, not something that was said.

The private channel is also reset for free. The experiment behind this file
(and `test_research_messages_start_empty_every_invocation`) shows that a
subgraph added as a node gets a fresh state each time it is invoked, even
under a parent checkpointer. So there is no "clear the scratchpad" step to
forget. Each invocation gets a new task id, and therefore a new checkpoint
namespace, so the next invocation starts on an empty channel. That holds even
though the old namespace's checkpoints are still on disk (see above).

What *does* cross the boundary is the write-up: `research_notes`, plus
`documents` and `research_iterations`. That is the whole of the Researcher's
contract with the rest of the graph, and `ResearcherOutput` in
multi_agent_state.py is where it is declared.

--------------------------------------------------------------------------
CONCEPT: why the Writer is *not* shaped like this
--------------------------------------------------------------------------
See the top of agents/writer.py. In short: the Writer makes one model call
with no tools, so it has no loop to expose and no working-out to hide. A
subgraph around it would be a box containing one box.
"""

from collections.abc import Sequence
from typing import Annotated, TypedDict

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from research_copilot.agent_loop import _execute_tool_call
from research_copilot.models import get_chat_model
from research_copilot.multi_agent_state import ResearcherInput, ResearcherOutput
from research_copilot.retrieval import format_docs, get_retriever
from research_copilot.tools import search_arxiv

# The Researcher's tool-loop cap. It is the same number as Phase 3-5's
# DEFAULT_MAX_ITERATIONS, because it is the same loop moved into an agent. It
# is named for the agent, though, because in 6.3 every agent gets its own
# budget and "max_iterations" stops meaning one thing.
DEFAULT_MAX_RESEARCH_ITERATIONS = 6


# The Researcher's own prompt. It lives beside the agent rather than in
# prompts.py because in a multi-agent system an agent *is* mostly its prompt,
# its tools, and its loop, and keeping the three in one file makes each agent
# readable on its own.
#
# The instruction that matters most is the one about output. Phase 5's
# AGENT_SYSTEM_PROMPT asked for an answer. This asks for findings: what was
# found, where it came from, and what could not be found. The Researcher is
# writing for another agent, not for the user. The Writer can only cite what the
# notes name, so source titles and URLs have to be in the notes verbatim.
RESEARCHER_SYSTEM_PROMPT = (
    "You are the Researcher on a small research team. You do not write the "
    "final answer - a separate Writer does that, using only the notes you hand "
    "over. Your job is to gather the evidence.\n\n"
    "You can search arXiv. Search for what the question needs; skip searching "
    "for things that are general knowledge. Keep queries short - every keyword "
    "must match.\n\n"
    "When you have enough, stop calling tools and reply with your research "
    "notes, in this form:\n"
    "- Findings: the facts that bear on the question, one per line, each "
    "followed by the source it came from.\n"
    "- Sources: every paper you relied on, with its exact title and arXiv URL, "
    "copied from the search results. Never invent or reconstruct a URL.\n"
    "- Gaps: anything the question asks that you could not find evidence for.\n\n"
    "Write notes, not an answer: no introduction, no conclusion, no advice to "
    "the reader."
)


class ResearcherState(ResearcherInput, ResearcherOutput, total=False):
    """The Researcher's whole internal state: what it reads, what it writes,
    and what it keeps to itself.

    Composed from the two contract schemas plus the private keys. So the three
    categories are visible in the type itself:

        ResearcherInput     read from the parent
        ResearcherOutput    written back to the parent
        (the keys below)    never leave this subgraph
    """

    # The private tool-loop transcript. It has the same reducer as the
    # parent's `messages` because it is the same kind of thing: an accumulating
    # conversation, here between the Researcher and its tools. The *name* is
    # what keeps it private. Had it been called `messages`, it would share a
    # channel with ResearcherInput's `messages`, and the separation would depend
    # entirely on ResearcherOutput filtering it back out. See the try-and-break
    # notes in the 6.1 report.
    research_messages: Annotated[list[BaseMessage], add_messages]


def build_researcher(
    *,
    model: BaseChatModel | None = None,
    tools: Sequence[BaseTool] | None = None,
    retriever: BaseRetriever | None = None,
    max_iterations: int = DEFAULT_MAX_RESEARCH_ITERATIONS,
) -> Runnable:
    """Compile the Researcher subgraph.

    Same injection pattern as `build_graph`: everything that talks to the
    outside world is an argument, so tests can run the agent offline. And the
    same laziness: the model and the retriever are built on first use, so the
    parent graph can be compiled and drawn without an API key or an embedding
    model.
    """
    tools = list(tools) if tools is not None else [search_arxiv]
    tools_by_name = {tool.name: tool for tool in tools}

    _bound: dict[str, Runnable] = {}

    def researcher_model() -> Runnable:
        if "model" not in _bound:
            _bound["model"] = (model or get_chat_model()).bind_tools(tools)
        return _bound["model"]

    # ----------------------------------------------------------------- nodes

    def retrieve(state: ResearcherState) -> dict:
        """Knowledge-base mode: Phase 5's `retrieve_docs`, now inside an agent.

        Retrieval per sub-question, merged and deduplicated, exactly as Phase 5
        did it. What is new is where the result goes. It goes to
        `research_notes`, the Researcher's hand-off field, and not to a
        `context` key that the answering model reads directly.

        Note there is no model call on this path. In knowledge-base mode the
        evidence is the excerpts themselves, and a model summarizing them
        before the Writer sees them would be a lossy copy of the text the
        Writer is supposed to cite. So the numbered excerpts *are* the notes.
        """
        active = retriever if retriever is not None else get_retriever()
        queries = [state.get("question", ""), *state.get("sub_questions", [])]

        documents: list[Document] = []
        seen: set[str] = set()
        for query in queries:
            for document in active.invoke(query):
                if document.page_content in seen:
                    continue
                seen.add(document.page_content)
                documents.append(document)

        return {
            "documents": documents,
            # Empty string, not format_docs' "(no excerpts retrieved)"
            # placeholder. "Produced nothing" should be the same value in both
            # modes, so the Writer - and in 6.2 the Supervisor - can test for
            # it without knowing which mode ran.
            "research_notes": format_docs(documents) if documents else "",
            "research_iterations": 0,
        }

    def research_model(state: ResearcherState) -> dict:
        """One step of the Researcher's tool loop: search, or write up notes.

        The request is built from three layers, in this order:
          1. the Researcher's own instructions (+ summary, + plan), rebuilt
             every call and never stored
          2. the shared conversation so far, *read* from `messages`. This is
             how a follow-up question gets researched in context.
          3. the private tool loop, from `research_messages`

        That order is also what keeps the request valid for the API. The shared
        transcript ends on the user's question, and the private loop continues
        from there: AI(tool call), Tool(result), AI(tool call), ... Roles
        alternate as the API requires, and nothing in layer 3 is ever written
        back into layer 2.
        """
        instructions: list[BaseMessage] = [SystemMessage(content=RESEARCHER_SYSTEM_PROMPT)]
        if state.get("summary"):
            instructions.append(
                SystemMessage(content=f"Summary of earlier conversation:\n{state['summary']}")
            )
        if state.get("sub_questions"):
            listed = "\n".join(f"- {q}" for q in state["sub_questions"])
            instructions.append(
                SystemMessage(
                    content=(
                        "The question was broken down into these sub-questions. "
                        "Gather evidence for each of them:\n" + listed
                    )
                )
            )

        request = [
            *instructions,
            *state.get("messages", []),
            *state.get("research_messages", []),
        ]
        ai_message = researcher_model().invoke(request)
        return {
            "research_messages": [ai_message],
            "research_iterations": state.get("research_iterations", 0) + 1,
        }

    def research_tools(state: ResearcherState) -> dict:
        """Run the tools the last research step asked for. Phase 3's `call_tool`,
        with its output going to the private channel instead of `messages`."""
        last = state["research_messages"][-1]
        if not isinstance(last, AIMessage) or not last.tool_calls:
            return {}
        return {
            "research_messages": [
                _execute_tool_call(call, tools_by_name) for call in last.tool_calls
            ]
        }

    def compile_notes(state: ResearcherState) -> dict:
        """Turn the private loop's end state into the one field that leaves it.

        CONCEPT: an agent must always hand something over.
        The normal case is that the model's last message is its written-up
        notes, with no tool calls, and those are the output. But the loop can
        also end because the iteration cap tripped while the model was *still
        asking for a search*. In that case the last message is a tool-call
        request with no text, and taking its text would hand the Writer an empty
        string. Every search the Researcher did would be discarded because it
        ran out of budget before summarizing.

        So the fallback keeps the evidence. The tool results that did come back
        are passed on raw, under a header that says what happened. The Writer
        gets worse notes rather than none, and the header puts the reason in
        the state dump, so "the Writer answered badly" can be traced back to
        "the Researcher ran out of budget".

        No model call here. A summarizing call on the fallback path would be a
        research step taken *after* the research budget said stop.
        """
        messages = state.get("research_messages", [])
        last = messages[-1] if messages else None

        if isinstance(last, AIMessage) and not last.tool_calls and last.text.strip():
            return {"research_notes": last.text.strip()}

        results = [
            m.text.strip() for m in messages if isinstance(m, ToolMessage) and m.text.strip()
        ]
        if not results:
            return {"research_notes": ""}
        return {
            "research_notes": (
                "(The Researcher's search budget ran out before it wrote up its "
                "findings. Raw search results follow, unfiltered.)\n\n"
                + "\n\n".join(results)
            )
        }

    # --------------------------------------------------------------- routing

    def route_research(state: ResearcherState) -> str:
        """Phase 3's `route_by_mode`, moved inside the agent that owns the choice.

        In Phase 3-5 the mode branch was at the top of the whole graph. Here it
        is inside the Researcher, because *how to gather evidence* is the
        Researcher's decision and nobody else's. The Writer does the same job
        in both modes, and the parent graph does not need to know that two
        research strategies exist.
        """
        mode = state.get("mode", "live-search")
        if mode == "knowledge-base":
            return "retrieve"
        if mode == "live-search":
            return "research_model"
        raise ValueError(
            f"researcher got an unknown mode {mode!r}; "
            "expected 'knowledge-base' or 'live-search'"
        )

    def should_search(state: ResearcherState) -> str:
        """Phase 3's `should_continue`, scoped to the Researcher's loop.

        The cap check comes first, and it routes to `compile_notes`, not to END.
        A research loop that stops must still produce notes. See
        `compile_notes` for why.
        """
        if state.get("research_iterations", 0) >= max_iterations:
            return "compile_notes"
        messages = state.get("research_messages", [])
        last = messages[-1] if messages else None
        if isinstance(last, AIMessage) and last.tool_calls:
            return "research_tools"
        return "compile_notes"

    # ---------------------------------------------------------------- wiring

    # CONCEPT: input_schema / output_schema on a StateGraph.
    # `ResearcherState` is the full internal state. The two schema arguments
    # narrow what crosses the boundary in each direction. Without them the
    # subgraph would accept and return every key in ResearcherState. With them,
    # the compiled graph's signature *is* the agent's contract.
    builder = StateGraph(
        ResearcherState,
        input_schema=ResearcherInput,
        output_schema=ResearcherOutput,
    )
    builder.add_node("retrieve", retrieve)
    builder.add_node("research_model", research_model)
    builder.add_node("research_tools", research_tools)
    builder.add_node("compile_notes", compile_notes)

    builder.add_conditional_edges(
        START,
        route_research,
        {"retrieve": "retrieve", "research_model": "research_model"},
    )
    builder.add_edge("retrieve", END)
    builder.add_conditional_edges(
        "research_model",
        should_search,
        {"research_tools": "research_tools", "compile_notes": "compile_notes"},
    )
    builder.add_edge("research_tools", "research_model")
    builder.add_edge("compile_notes", END)

    # `name` is what LangSmith and Studio label the subgraph with. No
    # checkpointer: a subgraph added as a node inherits the parent's, so
    # passing one here would be at best redundant.
    return builder.compile(name="researcher")
