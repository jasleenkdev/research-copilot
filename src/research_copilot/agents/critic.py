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

--------------------------------------------------------------------------
PHASE 7 A1: citations are checked by code, and "did not finish" is its own verdict
--------------------------------------------------------------------------
The loop above - model decides to call verify_citation, one id per call -
failed its first real end-to-end run (Groq gpt-oss-120b, E2E01). The model
checked one citation per call, a draft with three citations used the whole
4-call budget before any verdict, and the review "failed closed" as a
rejection. Twice. The draft was withheld, and nothing was delivered.

Two corrections, both applied here:

CONCEPT: a mechanical lookup is not the model's to invoke
Whether arXiv has a paper with id 2309.15217 is a lookup, not a judgement.
Letting the model decide *whether* and *when* to make it only adds ways to get
it wrong: it can skip a citation, check one per call and run out of budget,
or call a tool that does not exist. So `verify_citations` - a plain code node
- extracts every arXiv reference from the draft and checks each one before the
model is called. The model receives the results as facts, and makes one
judging call with no tools at all. It is the same principle as the
Researcher's reserved final call: when something must happen, code makes it
happen, and the model is left with only the part that needs judgement.

CONCEPT: "incomplete" is not "rejected"
6.3 made every "did not finish" a rejection, on Phase 5's fail-closed rule.
The rule's premise - never read "did not finish" as "approved" - still holds.
Its implementation was wrong: a rejection spends a revision and, through the
Supervisor's guards, forces a rewrite of a draft nobody faulted. E2E01's
Supervisor saw exactly that ("let the critic finish reviewing"), and the
guard overruled it. So a review that could not finish now returns
`verdict="incomplete"`. It is still not an approval (nothing finishes on it
as reviewed), but it spends no revision and forces no rewrite. See
`after_critique` in multi_agent_graph.py.

A garbled verdict is still a rejection. The model *finished* and said
something unreadable; Phase 5's parser fails closed on that, unchanged.
"""

from collections.abc import Sequence
from typing import Annotated

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from research_copilot.resilience import ModelCallFailure, invoke_with_recovery, is_invalid_tool_call
# Phase 5's parsers, imported rather than copied: one definition of what a
# verdict is, shared by the Phase 5 critic, the Phase 4 human gate, and this
# agent.
from research_copilot.graph import _parse_critique, _parse_verdict
from research_copilot.models import get_chat_model
from research_copilot.multi_agent_state import CriticInput, CriticOutput, budget_of
from research_copilot.tools.citations import extract_citations, parse_check, verify_citation

# The Critic's per-round model-call budget. Since Phase 7 each review costs one
# model call (the lookups are code), so 4 allows a retry after a failed call and
# more than one review in a round.
DEFAULT_MAX_CRITIC_ITERATIONS = 4

# Phase 5's CRITIC_PROMPT with the research notes added. Since Phase 7 the
# citation checks are done by code and handed over as facts, so the prompt says
# how to read them instead of asking the model to run them. The first-line
# format is unchanged, because `_parse_critique` still reads it.
CRITIC_AGENT_PROMPT = (
    "You are the Critic on a small research team. A Writer drafted an answer "
    "from notes a Researcher gathered. You are not rewriting it - you are "
    "deciding whether it is good enough to send.\n\n"
    "Judge it on:\n"
    "- Does it actually answer the question that was asked?\n"
    "- Is every claim supported by the research notes? A claim the notes do not "
    "contain is unsupported, however plausible.\n"
    "- Are the sources real and right? Every arXiv reference in the draft has "
    "already been looked up for you; the results are below and are facts. FOUND "
    "gives the paper's real title - check it matches what the draft says the "
    "paper is. NOT FOUND or INVALID means the citation is wrong. ERROR means the "
    "lookup failed - it is not evidence either way; do not reject a draft "
    "because of it.\n"
    "- Is speculation labelled as speculation?\n\n"
    "Reply in exactly this format:\n"
    "First line: APPROVE or REJECT, alone on the line.\n"
    "Then, if you rejected it, say what is wrong in concrete terms someone can "
    "act on. Say whether the problem is *evidence* (a source is missing, fake, or "
    "does not support the claim - more research is needed) or *writing* (the "
    "evidence is there but the draft misuses it, skips part of the question, or "
    "is badly organised). Name the claim, the citation, or the part of the "
    "question. A note nobody can act on wastes a revision.\n\n"
    "Approve a draft that is good enough. Holding out for perfect costs "
    "revisions and gets you nothing. Judge the draft against the question as "
    "asked - do not require topics the question did not ask about.\n\n"
    "You have no tools. Everything you need is in this message."
)

INVALID_TOOL_NOTE = (
    "Your previous reply tried to call a tool. You have no tools: every citation "
    "has already been checked, and the results are in the message. Give your "
    "verdict now, in the required format."
)

# The two ways a review can fail to finish. Both are `incomplete`, not reject.
MODEL_ERROR_CRITIQUE = (
    "The Critic could not finish its review: its model failed ({error}). This says "
    "nothing about the draft."
)
OUT_OF_BUDGET_CRITIQUE = (
    "The Critic could not finish its review: its round budget ({used} of {cap} "
    "model calls) was already spent. This says nothing about the draft."
)


class CriticState(CriticInput, CriticOutput, total=False):
    """Read contract + write contract + the private keys below."""

    # The Critic's own exchange with its model: the judging request's reply.
    # Named apart from `messages` for the same reason as `research_messages`:
    # the name is the first protection, and CriticOutput is the second.
    critic_messages: Annotated[list[BaseMessage], add_messages]
    # Phase 7: the raw lookup results, as the verifier returned them. Private
    # for the same reason as critic_messages - the Writer gets the Critic's
    # judgement, not "NOT FOUND: ..." strings - and, like it, checkpointed under
    # the critic's namespace.
    lookups: list[str]
    # Model calls in *this* invocation. The round total lives in
    # `budgets["critic"]`.
    critic_iterations: int


def build_critic(
    *,
    model: BaseChatModel | None = None,
    tools: Sequence[BaseTool] | None = None,
    max_iterations: int = DEFAULT_MAX_CRITIC_ITERATIONS,
) -> Runnable:
    """Compile the Critic subgraph.

    `tools` is the citation verifier (default [verify_citation]). Since Phase 7
    it is called by code in `verify_citations`, not offered to the model.
    Tests and the live-check harness inject stubs here, as before.
    """
    verifier = (list(tools) if tools is not None else [verify_citation])[0]
    _cache: dict[str, BaseChatModel] = {}

    def critic_model_runnable():
        if "model" not in _cache:
            # No bind_tools: the Critic has nothing to call (see above).
            _cache["model"] = model or get_chat_model()
        return _cache["model"]

    def spent(state: CriticState) -> int:
        """Round spend so far: the recorded round total plus this invocation."""
        return budget_of(state, "critic", max_iterations)["used"] + state.get("critic_iterations", 0)

    # ----------------------------------------------------------------- nodes

    def verify_citations(state: CriticState) -> dict:
        """Check every arXiv reference in the draft - code, not the model.

        One lookup per distinct paper, in order of appearance, capped (see
        tools/citations.py). No model call, so no model-call budget spent.
        """
        checks, lookups = [], []
        for cited in extract_citations(state.get("draft") or ""):
            try:
                result = str(verifier.invoke({"arxiv_id": cited}))
            except Exception as exc:  # noqa: BLE001 - a failed lookup is ERROR, not a crash
                result = f"ERROR: lookup failed ({type(exc).__name__}: {exc}). This says nothing about the citation."
            checks.append({"arxiv_id": cited, "status": parse_check(result)})
            lookups.append(f"{cited}: {result}")
        return {"citation_checks": checks, "lookups": lookups}

    def critic_model(state: CriticState) -> dict:
        """The one judging call: draft, notes, and lookup results as facts."""
        sub_questions = "\n".join(f"- {q}" for q in (state.get("sub_questions") or [])) or "(none)"
        lookups = "\n".join(state.get("lookups") or []) or "(the draft cites no arXiv papers)"
        request = [
            SystemMessage(content=CRITIC_AGENT_PROMPT),
            HumanMessage(
                content=(
                    f"Question: {state.get('question', '')}\n\n"
                    f"Sub-questions the plan called for (may be empty):\n{sub_questions}\n\n"
                    f"Research notes the draft was written from:\n"
                    f"{(state.get('research_notes') or '').strip() or '(none)'}\n\n"
                    f"Draft answer:\n"
                    f"{(state.get('draft') or '').strip() or '(the Writer produced an empty draft)'}\n\n"
                    f"Citation checks (already run; treat as fact):\n{lookups}"
                )
            ),
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

    def compile_verdict(state: CriticState) -> dict:
        """Turn the model's reply into the fields that leave the subgraph."""
        messages = state.get("critic_messages", [])
        checks = state.get("citation_checks") or []
        used = spent(state)
        budget = {"critic": {"used": used, "cap": max_iterations}}
        last = messages[-1] if messages else None

        if last is None:
            # Never called: the round budget was spent on entry.
            return {
                "verdict": "incomplete",
                "critique": OUT_OF_BUDGET_CRITIQUE.format(used=used, cap=max_iterations),
                "citation_checks": checks,
                "budgets": budget,
            }
        if last.additional_kwargs.get("model_error"):
            return {
                "verdict": "incomplete",
                "critique": MODEL_ERROR_CRITIQUE.format(error=last.additional_kwargs["model_error"][:300]),
                "citation_checks": checks,
                "budgets": budget,
            }

        # Phase 5's pipeline, unchanged: first-line convention -> verdict dict ->
        # the shared fail-closed parser. A garbled reply is a *finished* review
        # that said nothing readable, so it is still a rejection.
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
        model call past its cap. It goes straight to an `incomplete` verdict."""
        return "compile_verdict" if spent(state) >= max_iterations else "verify_citations"

    builder = StateGraph(CriticState, input_schema=CriticInput, output_schema=CriticOutput)
    builder.add_node("verify_citations", verify_citations)
    builder.add_node("critic_model", critic_model)
    builder.add_node("compile_verdict", compile_verdict)
    builder.add_conditional_edges(
        START, start_or_skip, {"verify_citations": "verify_citations", "compile_verdict": "compile_verdict"}
    )
    builder.add_edge("verify_citations", "critic_model")
    builder.add_edge("critic_model", "compile_verdict")
    builder.add_edge("compile_verdict", END)
    return builder.compile(name="critic")
