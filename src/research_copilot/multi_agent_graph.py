"""Phase 6: the multi-agent graph. Steps 6.1 (agents), 6.2 (Supervisor), 6.3 (Critic).

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

--------------------------------------------------------------------------
6.3: the Critic, and the loop back through the Supervisor
--------------------------------------------------------------------------
    supervisor ──"critic"──→ [critic] ──after_critique──┬─ approve ──→ review_draft* or finalize_answer
        ↑                                               ├─ reject, rounds left ──→ start_revision ─┐
        │                                               └─ reject, rounds spent ─→ finalize_answer │
        └──────────────────────────────────────────────────────────────────────────────────────┘
    supervisor ──"finish"──→ review_draft* or finalize_answer
    review_draft* ──after_review──┬─ approve/edit ──→ finalize_answer
                                  ├─ reject, rounds left ──→ start_revision ──→ supervisor
                                  └─ reject, rounds spent ─→ finalize_answer          (* only with require_approval)

Phase 5's `should_revise` sent a rejection "back to call_model". Here a
rejection goes back to the *Supervisor*, through `start_revision`, which counts
the round and resets every agent's round budget. The Supervisor reads the
critique and chooses who fixes it. There is no edge from a reviewer to the
Writer anywhere in this graph, and a test checks that.

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
from langgraph.types import interrupt

from research_copilot.agents.researcher import (
    DEFAULT_MAX_RESEARCH_ITERATIONS,
    build_researcher,
)
from research_copilot.agents.critic import DEFAULT_MAX_CRITIC_ITERATIONS, build_critic
from research_copilot.agents.supervisor import (
    approved_by_current_critique,
    critique_is_current,
    default_dispatch_caps,
    make_supervisor,
)
from research_copilot.agents.writer import DEFAULT_MAX_WRITER_CALLS, make_writer
# Reused verbatim from Phase 5: the planner's prompt and its fail-towards-less
# parser. Planning is not an agent in Phase 6. It is still one cheap structural
# call that runs before any agent does.
from research_copilot.graph import DEFAULT_MAX_SUB_QUESTIONS, _parse_plan, _parse_verdict
from research_copilot.models import get_chat_model
from research_copilot.config import get_settings
from research_copilot.multi_agent_state import (
    AGENTS,
    MultiAgentState,
    _register_turn_boundary,
    owns,
)
from research_copilot.pruning import make_prune_history
from research_copilot.prompts import PLAN_PROMPT
from research_copilot.state import Mode


# The reflection loop's cap for this graph. It is 2 here, not Phase 5's library
# default of 0. Phase 5 defaulted to 0 so that an existing Phase 4 caller's
# behaviour would not change under it. This graph has no earlier callers to
# protect, and a Critic that can reject but never ask for a fix (0) is not a
# useful default.
DEFAULT_MAX_REVISIONS = 2


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
    # --- 6.3 ---
    critic_model: BaseChatModel | None = None,
    critic_tools: Sequence[BaseTool] | None = None,
    enable_critic: bool = False,
    require_approval: bool = False,
    max_revisions: int = DEFAULT_MAX_REVISIONS,
    max_critic_iterations: int = DEFAULT_MAX_CRITIC_ITERATIONS,
    max_writer_calls: int = DEFAULT_MAX_WRITER_CALLS,
    # --- 6.4 ---
    memory_strategy: str | None = None,
    max_history_tokens: int | None = None,
    summary_model: BaseChatModel | None = None,
) -> Runnable:
    """Wire up and compile the multi-agent graph.

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
    write_draft = make_writer(
        model=writer_model or model, max_calls=max_writer_calls, max_revisions=max_revisions
    )
    critic = build_critic(
        model=critic_model or model, tools=critic_tools, max_iterations=max_critic_iterations
    )
    round_caps = {
        "researcher": max_research_iterations,
        "writer": max_writer_calls,
        "critic": max_critic_iterations,
    }
    caps = {
        **default_dispatch_caps(enable_critic=enable_critic, max_revisions=max_revisions),
        **(dispatch_caps or {}),
    }
    if not enable_critic:
        # Whatever the caller passed, the Critic is not dispatchable when off.
        caps["critic"] = 0
    supervisor = make_supervisor(
        model=supervisor_model or model,
        caps=caps,
        round_caps=round_caps,
        enable_critic=enable_critic,
        max_revisions=max_revisions,
        routing=routing,
    )

    if require_approval and checkpointer is None:
        # Phase 4's build-time refusal, carried over: interrupt() has nowhere to
        # park a run without a checkpointer.
        raise RuntimeError(
            "require_approval=True needs a checkpointer: interrupt() parks the "
            "run in one. Pass checkpointer=... (see checkpointing.py)."
        )

    # ------------------------------------------------------------------- nodes

    # 6.4: Phase 4's pruning node (pruning.py), at the entry of the turn.
    #
    # CONCEPT: small by construction is not bounded
    # This graph's shared transcript grows much more slowly than Phase 4's. The
    # Researcher's searches and the Critic's verifications live in private
    # channels, so every turn adds exactly two messages - the question and the
    # answer - however much work happened in between. Phase 4 was pruning a
    # transcript that grew by the tool loop's length every turn.
    #
    # Slower is still unbounded. Turn 200 of a long-lived thread resends turns
    # 1-199 to the Researcher (which reads `messages` for context) and to the
    # Writer, on every model call they make. So pruning is not optional here;
    # the pressure is lower, which only changes *when* it bites.
    #
    # Where it sits matters for the same reason as Phase 4, plus one new one:
    # at the entry, it runs before any subgraph is dispatched, so it can never
    # interact with an in-flight Researcher or Critic. And a resumed run
    # (multi-review) re-enters at the parked node, not at START, so resuming
    # never prunes mid-turn either.
    settings = get_settings()
    prune_history = make_prune_history(
        strategy=memory_strategy or settings.memory_strategy,
        budget=settings.max_history_tokens if max_history_tokens is None else max_history_tokens,
        summarizer=lambda: summary_model or model or get_chat_model(max_tokens=1024),
    )

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

    # --------------------------------------------- 6.3: review and revision

    def review_draft(state: MultiAgentState) -> dict:
        """Phase 4's human gate, in the multi-agent graph.

        The same interrupt, the same payload shape, the same `_parse_verdict`,
        and the same discipline: nothing above `interrupt()` except reading
        state, because a resumed node re-runs from its first line.

        What changed is where the verdict goes. Phase 4 wrote "approved" /
        "rejected" into the shared `status` and an edit into `draft`. Here the
        human owns three fields of their own, and an edit never touches the
        Writer's `draft` (see `human_edit` in multi_agent_state.py).
        """
        verdict = interrupt(
            {
                "question": state.get("question", ""),
                "mode": state.get("mode", "live-search"),
                "draft": state.get("draft", ""),
                "critique": state.get("critique", "") if state.get("verdict") else "",
                "prompt": (
                    "Approve, reject, or edit this draft. Resume with "
                    "Command(resume={'decision': 'approve'|'reject'|'edit', "
                    "'text': '<edited answer, for edit>', 'note': '<why>'})"
                ),
            }
        )
        decision, text, note = _parse_verdict(verdict)
        if decision == "edit":
            return {
                "human_verdict": "edit",
                "human_edit": text,
                "human_feedback": note or "(edited by the reviewer)",
            }
        if decision == "approve":
            return {"human_verdict": "approve", "human_edit": "", "human_feedback": note}
        return {
            "human_verdict": "reject",
            "human_edit": "",
            "human_feedback": note or "(rejected without a reason given)",
        }

    def start_revision(state: MultiAgentState) -> dict:
        """Begin a revision round: count it, and reset every agent's round budget.

        CONCEPT: the third reset site
        Phase 4 reset `iterations` per turn (turn_input). Phase 5 reset it per
        revision round (start_revision). 6.3 resets every agent's
        `budgets[agent]["used"]` per revision round, here and only here, so
        there is one place responsible for "a new round is starting".

            turn boundary    resets everything per-turn: dispatches, revisions,
                             budgets, verdicts, the log
            start_revision   resets per-round budgets and bumps `revisions`. It
                             does NOT touch `dispatches` (per turn, by design)

        That last line is the whole scoping decision in one sentence. See the
        note on `dispatches` in multi_agent_state.py.

        What it deliberately keeps: the critique and the human's feedback,
        because they are the revision instruction, and the draft, because it is
        the text being revised. Clearing them here would send the run on having
        deleted the reason it was sent. The same list as Phase 5's
        `revision_input`.
        """
        return {
            "revisions": state.get("revisions", 0) + 1,
            "budgets": {agent: {"used": 0} for agent in AGENTS},
        }

    def finalize_answer(state: MultiAgentState) -> dict:
        """Commit the answer to the shared transcript - or record that it was withheld.

        The only node that writes `messages` during a turn - see OWNERS in
        multi_agent_state.py.

        It does not clear `draft` afterwards, which Phase 4's version did.
        `draft` belongs to the Writer, and the next turn boundary clears it.
        The final state therefore shows the Writer's draft, any human edit, and
        the committed answer side by side.

        6.3: withheld when a reviewer's rejection of the current draft is still
        standing. The only way to arrive here with one is the revision cap
        running out. That is the cap overruling the verdict, as in Phase 5, and
        the message says so, with both reviewers' notes labelled.

        An empty draft still commits *something*, for Phase 4's reason. The turn
        opened with a HumanMessage, and leaving it unanswered puts two human
        turns back to back on the next turn, which the Anthropic API rejects.
        """
        critic_rejects = critique_is_current(state) and state.get("verdict") == "reject"
        human_rejects = require_approval and state.get("human_verdict") not in ("approve", "edit")
        if critic_rejects or human_rejects:
            notes = []
            if state.get("verdict") == "reject" and state.get("critique"):
                notes.append(f"critic: {state['critique']}")
            if state.get("human_verdict") == "reject" and state.get("human_feedback"):
                notes.append(f"human: {state['human_feedback']}")
            revisions = state.get("revisions", 0)
            spent = f" after {revisions} revision{'s' if revisions != 1 else ''}" if revisions else ""
            reason = "; ".join(notes) or "(no reason recorded)"
            return {"messages": [AIMessage(content=f"(draft withheld{spent} - {reason})")]}

        text = (state.get("human_edit") or "").strip() or (state.get("draft") or "").strip()
        return {"messages": [AIMessage(content=text or "(the Writer produced no answer)")]}

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
        if route in AGENTS:
            return route
        if route == "finish":
            # 6.3: "finish" ends the Supervisor's part, not the turn. With a
            # human gate, the human sees the draft before it is committed.
            return "review_draft" if require_approval else "finalize_answer"
        raise ValueError(f"supervisor left an unknown next_agent {route!r}")

    def after_critique(state: MultiAgentState) -> str:
        """Phase 5's `should_revise`, for the Critic's verdict. (6.3)

          approve                  -> the human, if there is one, else commit.
                                      Critic-first ordering lives here, the same
                                      place Phase 5 put it.
          reject, rounds left      -> start_revision, then the SUPERVISOR. This
                                      is the edge that replaces Phase 5's
                                      "always back to call_model". Whether to
                                      revise is decided here, by the cap. Who
                                      revises is decided there, by reading the
                                      critique.
          reject, rounds spent     -> finalize_answer, which withholds. The cap
                                      overrules the verdict (Phase 5, Part B).
        """
        if state.get("verdict") == "approve":
            return "review_draft" if require_approval else "finalize_answer"
        if state.get("revisions", 0) >= max_revisions:
            return "finalize_answer"
        return "start_revision"

    def after_review(state: MultiAgentState) -> str:
        """The same decision for the human's verdict. A human rejection is
        classified by the Supervisor too: "the numbers look wrong" and "too
        long" are different agents' problems, whoever says them."""
        if state.get("human_verdict") in ("approve", "edit"):
            return "finalize_answer"
        if state.get("revisions", 0) >= max_revisions:
            return "finalize_answer"
        return "start_revision"

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
    def begin_turn(state: MultiAgentState) -> dict:
        """The turn boundary, as a node (6.4). See `multi_agent_turn_input`."""
        return per_turn_reset()

    builder.add_node("begin_turn", owns("begin_turn")(begin_turn))
    builder.add_node("prune_history", owns("prune_history")(prune_history))
    builder.add_node("plan_question", owns("plan_question")(plan_question))
    builder.add_node("researcher", researcher)
    builder.add_node("writer", owns("writer")(write_draft))
    builder.add_node("finalize_answer", owns("finalize_answer")(finalize_answer))
    builder.add_node("supervisor", owns("supervisor")(supervisor))
    # 6.3. The Critic is a subgraph, added unwrapped for the Researcher's
    # reasons: its CriticOutput schema is its write contract, and unwrapped it
    # is drawn and streamed from the inside. `review_draft` and
    # `start_revision` are registered whatever the flags say - Phase 4's "one
    # graph shape" rule. `--critic` and `--approve` change which paths are
    # taken, not which paths exist.
    builder.add_node("critic", critic)
    builder.add_node("review_draft", owns("review_draft")(review_draft))
    builder.add_node("start_revision", owns("start_revision")(start_revision))

    # 6.2: the hub. 6.1's fixed edges researcher -> writer -> finalize_answer
    # are gone. Every agent returns to the Supervisor, and the Supervisor's
    # decision picks the next edge.
    # 6.4: the turn boundary first (every caller gets the reset), then
    # pruning, exactly where Phase 4 put it.
    builder.add_edge(START, "begin_turn")
    builder.add_edge("begin_turn", "prune_history")
    builder.add_edge("prune_history", "plan_question")
    builder.add_edge("plan_question", "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "researcher": "researcher",
            "writer": "writer",
            "critic": "critic",
            "review_draft": "review_draft",
            "finalize_answer": "finalize_answer",
        },
    )
    # The spokes. These are the cycles, and they are safe for the same reason
    # every earlier cycle was: the routing function that enters them can also
    # leave them, and here the dispatch caps guarantee it eventually must.
    builder.add_edge("researcher", "supervisor")
    builder.add_edge("writer", "supervisor")

    # 6.3: the reviewers do NOT return to the Supervisor directly. A verdict
    # goes through a routing function first, because "is there a revision
    # left?" is a cap check (code), not a judgement. Only a rejection that can
    # still be acted on reaches the Supervisor, via start_revision.
    builder.add_conditional_edges(
        "critic",
        after_critique,
        {
            "review_draft": "review_draft",
            "start_revision": "start_revision",
            "finalize_answer": "finalize_answer",
        },
    )
    builder.add_conditional_edges(
        "review_draft",
        after_review,
        {"start_revision": "start_revision", "finalize_answer": "finalize_answer"},
    )
    builder.add_edge("start_revision", "supervisor")
    builder.add_edge("finalize_answer", END)

    return builder.compile(checkpointer=checkpointer, name="research-copilot-multi-agent")


# 6.4: the build arguments that define how a turn runs - the part of
# `build_multi_agent_graph`'s signature that is *policy* rather than
# injection (models, tools, retriever, checkpointer are injection, and are
# rebuilt fresh by whoever resumes). Everything here is plain JSON, so it
# checkpoints like any other state.
POLICY_KEYS = (
    "routing",
    "enable_critic",
    "require_approval",
    "max_revisions",
    "dispatch_caps",
    "max_research_iterations",
    "max_critic_iterations",
    "max_writer_calls",
    "enable_planning",
    "max_sub_questions",
    "memory_strategy",
    "max_history_tokens",
)


def run_policy(**kwargs) -> dict:
    """The policy subset of build kwargs, with unset values dropped, so that a
    rebuild with `build_multi_agent_graph(**policy)` lands on the same defaults."""
    unknown = set(kwargs) - set(POLICY_KEYS)
    if unknown:
        raise TypeError(f"not policy keys: {sorted(unknown)}")
    return {k: v for k, v in kwargs.items() if v is not None}


def per_turn_reset() -> dict:
    """Every per-turn field, at its start-of-turn value. Written by `begin_turn`.

    CONCEPT: the turn boundary is the one writer that is not an owner.
    Every agent-owned key is reset here, and that is the ownership rule's
    lifecycle, not an exception to it (see multi_agent_state.py). Within a turn,
    only the Researcher writes `research_notes`. Between turns, last turn's
    notes are stale, and nobody should read them as current - the Writer
    especially, because it would cite the previous question's sources.

    `budgets` goes through merge_budgets, so a reset has to name each agent: `{}`
    would merge as "change nothing". `supervisor_log` can be reset with `[]`
    only because it does not use an accumulating reducer.
    """
    return {
        "sub_questions": [],
        "research_notes": "",
        "documents": [],
        "research_iterations": 0,
        "research_outcome": "",
        "draft": "",
        "researcher_brief": "",
        "next_agent": "",
        "dispatches": {},
        "supervisor_log": [],
        "critique": "",
        "verdict": "",
        "citation_checks": [],
        "human_verdict": "",
        "human_feedback": "",
        "human_edit": "",
        "revisions": 0,
        "budgets": {agent: {"used": 0} for agent in AGENTS},
    }


_register_turn_boundary(per_turn_reset())


def multi_agent_turn_input(
    question: str, mode: Mode = "live-search", *, policy: dict | None = None
) -> dict:
    """The input that starts one turn: the request, and nothing else.

    CONCEPT (6.4): the reset moved from the input into the graph
    Through 6.3, this function also reset every per-turn field - Phase 4's
    `turn_input` pattern. Studio exposed the flaw: Studio (and Phase 7's API)
    does not call this function. It sends `{"question": ..., "messages": [...]}`
    and nothing more. On a second turn in the same Studio thread, the dispatch
    counts, revision count and decision log carried straight over from the
    first turn. The Critic started turn 2 already at its per-turn cap, and the
    Supervisor's guards were enforcing turn 1's budget against turn 2's work.

    A reset that lives in a helper only works for callers who know to use the
    helper. So the reset is now the graph's first node, `begin_turn`, and it
    runs for every caller. It is safe to run unconditionally, because START is
    only ever entered by a new turn: a resume (`Command(resume=...)`) re-enters
    at the parked node, and `update_state` runs no nodes at all.

    What stays here is what only the caller can supply: the question, the mode,
    the user's message, and the policy the caller built the graph with.
    """
    return {
        "question": question,
        "mode": mode,
        # 6.4. Pass the same dict to build_multi_agent_graph(**policy); see
        # `run_policy` in multi_agent_state.py for why it is stored at all.
        "run_policy": dict(policy or {}),
        "messages": [HumanMessage(content=question)],
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
# COUNTERS, AS OF 6.3
# --------------------------------------------------------------------------
# Four counters, three scopes, and each is reset in exactly one place:
#
#   counter                    scope                reset by           caps it
#   -------------------------  -------------------  -----------------  --------------------
#   research_iterations        one Researcher pass  (private start)    - (a report)
#   budgets[agent]["used"]     one revision round   start_revision     max_*_iterations /
#                                                                      max_writer_calls
#   dispatches[agent]          one turn             turn boundary      dispatch caps
#   revisions                  one turn             turn boundary      max_revisions
#
# The rule from Phase 5 held a third time: one counter per loop, reset by
# whoever begins a pass of that loop. What 6.3 added is the table itself. With
# this many counters, the scope of each has to be written down where a reader
# (or Phase 7's eval) will look for it.
