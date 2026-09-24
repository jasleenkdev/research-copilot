"""Phase 6: the multi-agent graph. Steps 6.1 (agents) and 6.2 (Supervisor).

Read `graph.py` beside this file. That graph is left exactly as Phase 5 built
it, so the two shapes can be compared directly:

    graph.py (Phase 5)                      multi_agent_graph.py (Phase 6.1)
    --------------------------------------  --------------------------------------
    one `call_model` node that searched     a Researcher subgraph that searches and
      *and* wrote the answer                  hands over notes, then a Writer node
                                              that writes from them
    tool calls in the shared `messages`     tool calls in the Researcher's private
                                              `research_messages`, never shared
    route_by_mode at the top of the graph   the same branch, inside the Researcher
    `draft` written by whoever ran last     every key has one owner, checked at
                                              runtime (multi_agent_state.py)

The shape in 6.1 is a straight line. The Researcher is drawn expanded to show
it is a subgraph:

    START ─→ plan_question ─→ ┌ researcher ─────────────────────────────────┐ ─→ writer ─→ finalize_answer ─→ END
                              │ retrieve  |  research_model <-> research_tools │
                              │           |        └─→ compile_notes           │
                              └────────────────────────────────────────────────┘

CONCEPT: a fixed hand-off, and why 6.1 starts with one
Every edge above is unconditional. Research always happens, then writing, then
the answer is committed. Nothing decides who acts next, because only one order
is possible.

That is deliberately too simple for the finished system. 6.2 replaces these
edges with a Supervisor that chooses. But building the fixed line first
separates two questions that would otherwise be tangled:

  1. Do the agents compose? Does the Writer produce a good answer from the
     Researcher's notes alone, with the tool loop hidden from it? That is a
     question about *state ownership*, and it is this step.
  2. Does the Supervisor route well? That is a question about *judgement*, and
     it is 6.2.

If 6.2 misbehaves, a known-good 6.1 means the fault is in the routing, not in
the agents underneath it.

--------------------------------------------------------------------------
6.2: the hub
--------------------------------------------------------------------------
The straight line above is gone from the wiring. Every agent now returns to a
Supervisor node, and a routing function sends the run wherever the Supervisor
decided:

    START ─→ plan_question ─→ supervisor ──route_from_supervisor──┬─ "researcher" ─→ [researcher] ─┐
                                  ↑                               ├─ "writer" ─────→ writer ───────┤
                                  └───────────────────────────────┼────────────────────────────────┘
                                                                  └─ "finish" ─────→ finalize_answer ─→ END

The 6.1 hand-off still exists as a *policy* instead of as edges.
`routing="fixed"` runs this same graph with `fixed_policy` (agents/supervisor.py)
in the Supervisor's seat, and it takes exactly 6.1's path. So the change from
6.1 to 6.2 is "who decides", not "what the graph is", the same move Phase 5
made when it put a model in the reviewer's seat.

CONCEPT: hub-and-spoke vs. agents routing to each other
The alternative to a hub is letting each agent pick its successor: the
Researcher decides "now the Writer", the Writer decides "back to research".
It avoids an extra model call per hop, and it spreads the routing logic across
every agent. Then no single place knows the dispatch counts, the budget, or the
history of decisions, and the question "why did the run go there?" has as many
answers as there are agents. The hub costs one model call per hop and buys one
place where routing is decided, capped, logged, and overridable. For a system
whose routing is the thing under test, that is the right trade.

CONCEPT: what is deliberately missing, compared with Phase 5
Several Phase 4-5 features are not wired in yet. Each is left out for a stated
reason, not forgotten:

  prune_history   Not in this graph. `messages` stays small here by
                  construction: tool traffic never enters it, so each turn adds
                  exactly two messages. The 6.1 CLI command also runs one turn
                  per invocation with no thread. Pruning comes back in 6.4,
                  alongside `--thread` and `--checkpointer`.
  critic / human  6.3. The Critic is an agent, and in 6.3 a rejection routes
  review          through the Supervisor, which does not exist until 6.2.
                  `finalize_answer` therefore always commits.
  revisions       6.3. There is nothing to revise without a reviewer.
"""

from collections.abc import Sequence
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from research_copilot.agents.researcher import (
    DEFAULT_MAX_RESEARCH_ITERATIONS,
    build_researcher,
)
from research_copilot.agents.supervisor import (
    DEFAULT_DISPATCH_CAPS,
    make_supervisor,
)
from research_copilot.agents.writer import make_writer
# Reused verbatim from Phase 5: the planner's prompt and its fail-towards-less
# parser. Planning is not an agent in Phase 6. It is still one cheap structural
# call that runs before any agent does.
from research_copilot.graph import DEFAULT_MAX_SUB_QUESTIONS, _parse_plan
from research_copilot.models import get_chat_model
from research_copilot.multi_agent_state import MultiAgentState, owns
from research_copilot.prompts import PLAN_PROMPT
from research_copilot.state import Mode


def build_multi_agent_graph(
    *,
    model: BaseChatModel | None = None,
    researcher_model: BaseChatModel | None = None,
    writer_model: BaseChatModel | None = None,
    planner_model: BaseChatModel | None = None,
    supervisor_model: BaseChatModel | None = None,
    routing: Literal["supervisor", "fixed"] = "supervisor",
    dispatch_caps: dict[str, int] | None = None,
    tools: Sequence[BaseTool] | None = None,
    retriever: BaseRetriever | None = None,
    max_research_iterations: int = DEFAULT_MAX_RESEARCH_ITERATIONS,
    enable_planning: bool = False,
    max_sub_questions: int = DEFAULT_MAX_SUB_QUESTIONS,
    checkpointer: BaseCheckpointSaver | None = None,
) -> Runnable:
    """Wire up and compile the 6.1 multi-agent graph.

    One model argument per agent, each falling back to `model` and then to the
    factory - the same pattern as Phase 5's `critic_model`/`planner_model`.
    Separate models per agent is what a multi-agent system is *for* in
    production. A cheap fast model can research, a stronger one can write, and
    (6.3) a different one can judge. It also lets a test script each agent
    independently and assert on exactly what each one was shown.
    """

    # ------------------------------------------------------------------ agents

    researcher = build_researcher(
        model=researcher_model or model,
        tools=tools,
        retriever=retriever,
        max_iterations=max_research_iterations,
    )
    write_draft = make_writer(model=writer_model or model)
    supervisor = make_supervisor(
        model=supervisor_model or model,
        caps=dispatch_caps or DEFAULT_DISPATCH_CAPS,
        routing=routing,
    )

    # ------------------------------------------------------------------- nodes

    _planner: dict[str, BaseChatModel] = {}

    def plan_question(state: MultiAgentState) -> dict:
        """Phase 5's planning node, unchanged in behaviour.

        Rewritten rather than imported, because Phase 5's is a closure inside
        `build_graph`. The prompt and parser are imported, so the planning
        *logic* exists in one place. Same no-op-when-off rule as Phase 5: one
        graph shape, whatever the flags say.
        """
        if not enable_planning:
            return {}
        question = state.get("question", "")
        if not question:
            return {"sub_questions": []}
        if "model" not in _planner:
            _planner["model"] = planner_model or model or get_chat_model()
        chain = PLAN_PROMPT | _planner["model"] | StrOutputParser()
        raw = chain.invoke(
            {
                "question": question,
                "mode": state.get("mode", "live-search"),
                "max_sub_questions": max_sub_questions,
            }
        )
        return {"sub_questions": _parse_plan(raw, max_sub_questions)}

    def finalize_answer(state: MultiAgentState) -> dict:
        """Commit the Writer's draft to the shared transcript.

        The only node that writes `messages` during a turn - see OWNERS in
        multi_agent_state.py.

        It does not clear `draft` afterwards, which Phase 4's version did.
        `draft` belongs to the Writer, and the next turn boundary clears it.
        Leaving the committed draft in place for the rest of the turn is also
        simply more informative: the final state shows the Writer's output
        and the committed answer side by side. They are the same text now, and
        in 6.4, when a human can edit the answer, they will not always be.

        An empty draft still commits *something*, for Phase 4's reason. The turn
        opened with a HumanMessage, and leaving it unanswered puts two human
        turns back to back on the next turn, which the Anthropic API rejects.
        """
        draft = state.get("draft", "").strip()
        return {"messages": [AIMessage(content=draft or "(the Writer produced no answer)")]}

    def route_from_supervisor(state: MultiAgentState) -> str:
        """Read the Supervisor's decision and name the next node.

        CONCEPT: why the decision and the edge are two steps
        The Supervisor node makes the decision (a model call, plus guards) and
        writes it to `next_agent`. This function only reads it back. It could
        have been one step: LangGraph lets a node return
        `Command(goto="writer")` and route itself. Two steps are used here for
        three reasons:
          - a routing function must not have side effects (graph.py, Phase 3),
            and a model call is a side effect with a price. The decision has to
            be made in a node.
          - `next_agent` in state means the decision is in every checkpoint and
            in every state dump, next to the log entry that explains it.
          - the `path_map` below declares every possible destination up front,
            so the hub is drawn completely before anything runs. A node that
            returns `Command(goto=...)` needs a separate `destinations=` hint to
            get the same drawing.

        Unknown values raise, the same rule as `route_by_mode`. The Supervisor
        only ever writes a guarded route, so reaching the raise means a bug in
        the Supervisor, not a bad model reply.
        """
        route = state.get("next_agent", "")
        if route in ("researcher", "writer"):
            return route
        if route == "finish":
            return "finalize_answer"
        raise ValueError(f"supervisor left an unknown next_agent {route!r}")

    # ------------------------------------------------------------------ wiring

    builder = StateGraph(MultiAgentState)

    # CONCEPT: `owns(...)` on every plain node, and not on the subgraph.
    # The plain nodes are wrapped so that returning a key they do not own
    # raises OwnershipError. The Researcher is added as the compiled subgraph
    # itself, unwrapped, for two reasons:
    #   - its output_schema already restricts what it can return. That is the
    #     same guarantee, enforced structurally rather than checked at runtime.
    #   - wrapping it in a function would hide its structure from the drawing.
    #     `get_graph(xray=True)`, which is what draws the Researcher's inside,
    #     can only expand a compiled graph that *is* the node. A subgraph called
    #     from inside a node function still streams its steps with
    #     `stream(subgraphs=True)`, but it is drawn as one opaque box.
    builder.add_node("plan_question", owns("plan_question")(plan_question))
    builder.add_node("researcher", researcher)
    builder.add_node("writer", owns("writer")(write_draft))
    builder.add_node("finalize_answer", owns("finalize_answer")(finalize_answer))
    builder.add_node("supervisor", owns("supervisor")(supervisor))

    # 6.2: the hub. 6.1's fixed edges researcher -> writer -> finalize_answer
    # are gone. Every agent returns to the Supervisor, and the Supervisor's
    # decision picks the next edge.
    builder.add_edge(START, "plan_question")
    builder.add_edge("plan_question", "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "researcher": "researcher",
            "writer": "writer",
            "finalize_answer": "finalize_answer",
        },
    )
    # The spokes. These are the cycles, and they are safe for the same reason
    # every earlier cycle was: the routing function that enters them can also
    # leave them, and here the dispatch caps guarantee it eventually must.
    builder.add_edge("researcher", "supervisor")
    builder.add_edge("writer", "supervisor")
    builder.add_edge("finalize_answer", END)

    return builder.compile(checkpointer=checkpointer, name="research-copilot-multi-agent")


def multi_agent_turn_input(question: str, mode: Mode = "live-search") -> dict:
    """The state update that starts one turn - `turn_input`'s multi-agent twin.

    CONCEPT: the turn boundary is the one writer that is not an owner.
    Every agent-owned key is reset here, and that is the ownership rule's
    lifecycle, not an exception to it (see multi_agent_state.py). Within a turn,
    only the Researcher writes `research_notes`. Between turns, last turn's
    notes are stale, and nobody should read them as current. The Writer
    especially must not, because it would cite the previous question's sources.

    Resetting here rather than having each agent clear its own field at the
    start of its run is the same choice Phase 4 made for `iterations`. A reset
    that depends on the owner running can be skipped, because in 6.2 the
    Supervisor may not route to an agent at all on a given turn. A reset at the
    boundary cannot be skipped.
    """
    return {
        "question": question,
        "mode": mode,
        "messages": [HumanMessage(content=question)],
        "sub_questions": [],
        "research_notes": "",
        "documents": [],
        "research_iterations": 0,
        "draft": "",
        # --- 6.2 ---
        # All Supervisor-owned, and all per turn. `dispatches` especially: a
        # per-turn budget that is never reset is a budget that only ever runs
        # out (state.py, Phase 5). `supervisor_log` can be reset like this only
        # because it does not use an accumulating reducer - see its
        # declaration.
        "research_outcome": "",
        "researcher_brief": "",
        "next_agent": "",
        "dispatches": {},
        "supervisor_log": [],
    }


def run_multi_agent(
    question: str,
    *,
    mode: Mode = "live-search",
    graph: Runnable | None = None,
    config: dict | None = None,
    **build_kwargs,
) -> MultiAgentState:
    """Invoke the multi-agent graph on one question and return the final state."""
    graph = graph or build_multi_agent_graph(**build_kwargs)
    return graph.invoke(multi_agent_turn_input(question, mode), config)


def make_graph(config: dict | None = None) -> Runnable:
    """Factory for the LangGraph dev server / Studio (see langgraph.json).

    Compiles without an API key or an embedding model, because every agent
    (the Supervisor included) builds its model and retriever on first use. So Studio can draw the graph,
    Researcher subgraph included, before anything has been configured.
    """
    return build_multi_agent_graph()


# --------------------------------------------------------------------------
# NOTE FOR 6.2 / 6.3: counters
# --------------------------------------------------------------------------
# `research_iterations` is the first per-agent counter. Right now it needs no
# reset site of its own. The subgraph's private state starts fresh every time
# the Researcher is invoked (see agents/researcher.py), and the Researcher
# counts from 0 inside it. The parent-level copy is only a *report* of what the
# last research pass spent.
#
# That stopped being enough in 6.2. Once the Supervisor can send work back to
# the Researcher, the Researcher runs more than once per turn, and each run
# starts counting from zero. So the tool budget is per *invocation*.
#
# 6.2 caps the number of invocations instead: `dispatches` (Supervisor-owned)
# against DEFAULT_DISPATCH_CAPS. The total research spend in a turn is now
# bounded by dispatch cap x research-iteration cap. That is two numbers in two
# places, and neither is visible from the other, which is the smell 6.3's
# per-agent budget structure exists to fix. It also has to decide whether a
# budget resets per revision (Phase 5's `iterations`) or per turn (6.2's
# `dispatches`). That choice, not the container type, is the real decision.
