"""The Critic agent: Phase 5's `critique_draft`, promoted to an agent.

Phase 5 predicted this file in its closing note: "the Critic ports over almost
unchanged ... it gets its own system prompt, its own tools (so it can go and
check a citation rather than only doubting it), and possibly its own
several-step loop. The verdict it returns and the edge it returns it on are
unchanged." All four held. Side by side:

    critique_draft (Phase 5)            Critic agent (6.3)
    ----------------------------------  ----------------------------------------
    reads `draft`                       reads `draft` + `research_notes`
    one model call, no tools            a loop: model <-> verify_citation
    CRITIC_PROMPT                       CRITIC_AGENT_PROMPT (below), same
                                          APPROVE / REJECT first-line format
    _parse_critique -> _parse_verdict   the same two functions, imported
    writes status + critique            writes verdict + critique +
                                          citation_checks
    -> should_revise                    -> after_critique (same decision:
                                          approve finishes, reject revises
                                          unless the cap is spent)

What is *not* the same, and why:

  - the verdict goes in `verdict`, not the shared Phase 4 `status`. `status`
    was a field that three nodes wrote. Under 6.1's rule each reviewer owns its
    own verdict field, and the routing function reads whichever one it is
    routing on.
  - a rejection no longer goes straight back to the writer. It counts against
    `revisions` (in `start_revision`) and then goes to the *Supervisor*, which
    reads the critique and decides who fixes it: a missing source is the
    Researcher's job, a structural problem is the Writer's. That
    classification is the reason 6.2's Supervisor exists.

--------------------------------------------------------------------------
CONCEPT: the Critic as a subgraph with a private channel
--------------------------------------------------------------------------
The Critic is shaped like the Researcher (agents/researcher.py), and for the
same reason. Verifying citations is a loop (verify, read, verify the next one,
decide), and a loop gets graph steps and its own message list. That list is
`critic_messages`, private in exactly the Researcher's sense:

  - it is not in MultiAgentState or in CriticOutput, so it never reaches the
    parent state, the shared transcript, or any other agent. The Writer
    revising a draft sees the *critique*, the Critic's judgement, and not
    "NOT FOUND: ..." tool output it might misread as an instruction.
  - it *is* checkpointed, under the `critic:<task-id>` namespace, like the
    Researcher's. "Private" means private from other agents, not off the disk.
  - it starts empty on every invocation, so a second critique of a revised
    draft does not inherit the first critique's verification calls.

This time the tests came first. `tests/test_critic_agent.py` pins down all three
properties, and was written before this file.

What does cross the boundary, beyond the verdict, is `citation_checks`: the
tool results, lifted out of the private channel as structured data. The raw
tool traffic stays private. The *facts* it established are shared, because the
Supervisor routes on them.

--------------------------------------------------------------------------
CONCEPT: failing closed, now with a budget
--------------------------------------------------------------------------
`_parse_critique` fails closed: anything that is not a clear APPROVE is a
rejection. The Critic adds one more way to fail. It can run out of its
per-round budget while still verifying, with no verdict written. That is also
a rejection, with a critique saying so, for Phase 5's reason: an approval gate
must never read "did not finish" as "approved". The cost is a revision round,
and `max_revisions` bounds it.
"""

from collections.abc import Sequence
from typing import Annotated

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from research_copilot.agent_loop import _execute_tool_call
from research_copilot.resilience import ModelCallFailure, invoke_with_recovery, is_invalid_tool_call
# Phase 5's parsers, imported rather than copied: one definition of what a
# verdict is, shared by the Phase 5 critic, the Phase 4 human gate, and this
# agent.
from research_copilot.graph import _parse_critique, _parse_verdict
from research_copilot.models import get_chat_model
from research_copilot.multi_agent_state import CriticInput, CriticOutput, budget_of
from research_copilot.tools.citations import parse_check, verify_citation

# The Critic's per-round model-call budget. Small: a Critic verifies what the
# draft cites and decides, and it does not explore.
DEFAULT_MAX_CRITIC_ITERATIONS = 4

# Phase 5's CRITIC_PROMPT with three additions: the research notes, the tool,
# and how to read the tool's failures. The first-line format is unchanged,
# because `_parse_critique` still reads it.
CRITIC_AGENT_PROMPT = (
    "You are the Critic on a small research team. A Writer drafted an answer "
    "from notes a Researcher gathered. You are not rewriting it - you are "
    "deciding whether it is good enough to send.\n\n"
    "Judge it on:\n"
    "- Does it actually answer the question that was asked?\n"
    "- Is every claim supported by the research notes? A claim the notes do not "
    "contain is unsupported, however plausible.\n"
    "- Are the sources real? Use verify_citation on each arXiv id or URL the "
    "draft cites. A NOT FOUND or INVALID result means the citation is wrong. An "
    "ERROR result means the lookup failed - it is not evidence either way; do "
    "not reject a draft because of it.\n"
    "- Is speculation labelled as speculation?\n\n"
    "Verify, then reply in exactly this format:\n"
    "First line: APPROVE or REJECT, alone on the line.\n"
    "Then, if you rejected it, say what is wrong in concrete terms someone can "
    "act on. Say whether the problem is *evidence* (a source is missing, fake, or "
    "does not support the claim - more research is needed) or *writing* (the "
    "evidence is there but the draft misuses it, skips part of the question, or "
    "is badly organised). Name the claim, the citation, or the part of the "
    "question. A note nobody can act on wastes a revision.\n\n"
    "Approve a draft that is good enough. Holding out for perfect costs "
    "revisions and gets you nothing.\n\n"
    # Phase 7: the same line the Researcher got, for the same reason.
    "verify_citation is your only tool. You cannot open URLs, files, or web pages."
)

# Phase 7: same hint as the Researcher's, for the Critic's one tool.
INVALID_TOOL_NOTE = (
    "Your previous reply tried to call a tool that does not exist. The only tool "
    "you have is verify_citation. Either call it on an arXiv id, or give your "
    "verdict now."
)

MODEL_ERROR_CRITIQUE = (
    "The Critic's model failed before reaching a verdict ({error}). Treated as a "
    "rejection (fail closed)."
)

OUT_OF_BUDGET_CRITIQUE = (
    "The Critic ran out of its verification budget ({used} of {cap} model calls) "
    "before reaching a verdict. Treated as a rejection (fail closed). Citations "
    "checked so far: {checked}."
)


class CriticState(CriticInput, CriticOutput, total=False):
    """Read contract + write contract + the private keys below."""

    # The private verification loop. Named apart from `messages` for the same
    # reason as `research_messages`: the name is the first protection, and
    # CriticOutput is the second.
    critic_messages: Annotated[list[BaseMessage], add_messages]
    # Model calls in *this* invocation. The round total lives in
    # `budgets["critic"]`. This one is private, because only the loop's own
    # stopping rule needs it.
    critic_iterations: int


def build_critic(
    *,
    model: BaseChatModel | None = None,
    tools: Sequence[BaseTool] | None = None,
    max_iterations: int = DEFAULT_MAX_CRITIC_ITERATIONS,
) -> Runnable:
    """Compile the Critic subgraph. Same laziness and injection as the Researcher."""
    tools = list(tools) if tools is not None else [verify_citation]
    tools_by_name = {t.name: t for t in tools}
    _bound: dict[str, Runnable] = {}

    def critic_model_runnable() -> Runnable:
        if "model" not in _bound:
            _bound["model"] = (model or get_chat_model()).bind_tools(tools)
        return _bound["model"]

    def spent(state: CriticState) -> int:
        """Round spend so far: the recorded round total plus this invocation."""
        return budget_of(state, "critic", max_iterations)["used"] + state.get("critic_iterations", 0)

    # ----------------------------------------------------------------- nodes

    def critic_model(state: CriticState) -> dict:
        """One step: verify a citation, or give the verdict."""
        sub_questions = "\n".join(f"- {q}" for q in (state.get("sub_questions") or [])) or "(none)"
        request = [
            SystemMessage(content=CRITIC_AGENT_PROMPT),
            HumanMessage(
                content=(
                    f"Question: {state.get('question', '')}\n\n"
                    f"Sub-questions the plan called for (may be empty):\n{sub_questions}\n\n"
                    f"Research notes the draft was written from:\n"
                    f"{(state.get('research_notes') or '').strip() or '(none)'}\n\n"
                    f"Draft answer:\n"
                    f"{(state.get('draft') or '').strip() or '(the Writer produced an empty draft)'}"
                )
            ),
            *state.get("critic_messages", []),
        ]
        result, attempts = invoke_with_recovery(
            critic_model_runnable(), request,
            recoverable=is_invalid_tool_call, note=INVALID_TOOL_NOTE, where="critic",
        )
        iterations = state.get("critic_iterations", 0) + attempts
        if isinstance(result, ModelCallFailure):
            marker = AIMessage(content="", additional_kwargs={"model_error": result.error})
            return {"critic_messages": [marker], "critic_iterations": iterations}
        return {"critic_messages": [result], "critic_iterations": iterations}

    def critic_tools(state: CriticState) -> dict:
        last = state["critic_messages"][-1]
        if not isinstance(last, AIMessage) or not last.tool_calls:
            return {}
        return {
            "critic_messages": [_execute_tool_call(call, tools_by_name) for call in last.tool_calls]
        }

    def compile_verdict(state: CriticState) -> dict:
        """Turn the private loop's end into the fields that leave the subgraph."""
        messages = state.get("critic_messages", [])

        # The structured facts: every verification this pass made, in order.
        requested = {
            call["id"]: str(call["args"].get("arxiv_id", ""))
            for m in messages
            if isinstance(m, AIMessage)
            for call in (m.tool_calls or [])
            if call["name"] == "verify_citation"
        }
        checks = [
            {"arxiv_id": requested[m.tool_call_id], "status": parse_check(m.text)}
            for m in messages
            if isinstance(m, ToolMessage) and m.tool_call_id in requested
        ]

        used = spent(state)
        budget = {"critic": {"used": used, "cap": max_iterations}}
        last = messages[-1] if messages else None

        if isinstance(last, AIMessage) and last.additional_kwargs.get("model_error"):
            # Phase 7: the model failed even after a retry. Fail closed, as for
            # an exhausted budget: "did not finish" is never "approved".
            return {
                "verdict": "reject",
                "critique": MODEL_ERROR_CRITIQUE.format(error=last.additional_kwargs["model_error"][:300]),
                "citation_checks": checks,
                "budgets": budget,
            }

        if last is None or (isinstance(last, AIMessage) and last.tool_calls):
            # Out of budget before a verdict - or never started, because the
            # round budget was already spent on entry. Fail closed.
            checked = ", ".join(f"{c['arxiv_id']} {c['status']}" for c in checks) or "none"
            return {
                "verdict": "reject",
                "critique": OUT_OF_BUDGET_CRITIQUE.format(
                    used=used, cap=max_iterations, checked=checked
                ),
                "citation_checks": checks,
                "budgets": budget,
            }

        # Phase 5's pipeline, unchanged: first-line convention -> verdict dict ->
        # the shared fail-closed parser.
        decision, _text, note = _parse_verdict(_parse_critique(last.text))
        if decision == "approve":
            return {
                "verdict": "approve",
                "critique": note or "(approved by the critic)",
                "citation_checks": checks,
                "budgets": budget,
            }
        return {
            "verdict": "reject",
            "critique": note or "(rejected by the critic without a reason given)",
            "citation_checks": checks,
            "budgets": budget,
        }

    # --------------------------------------------------------------- routing

    def start_or_skip(state: CriticState) -> str:
        """Defence in depth. The Supervisor's guards never dispatch a Critic
        whose round budget is spent, but if something does, it must not make a
        model call past its cap. It goes straight to the fail-closed verdict."""
        return "compile_verdict" if spent(state) >= max_iterations else "critic_model"

    def should_verify(state: CriticState) -> str:
        if spent(state) >= max_iterations:
            return "compile_verdict"
        last = state["critic_messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "critic_tools"
        return "compile_verdict"

    builder = StateGraph(CriticState, input_schema=CriticInput, output_schema=CriticOutput)
    builder.add_node("critic_model", critic_model)
    builder.add_node("critic_tools", critic_tools)
    builder.add_node("compile_verdict", compile_verdict)
    builder.add_conditional_edges(
        START, start_or_skip, {"critic_model": "critic_model", "compile_verdict": "compile_verdict"}
    )
    builder.add_conditional_edges(
        "critic_model",
        should_verify,
        {"critic_tools": "critic_tools", "compile_verdict": "compile_verdict"},
    )
    builder.add_edge("critic_tools", "critic_model")
    builder.add_edge("compile_verdict", END)
    return builder.compile(name="critic")
