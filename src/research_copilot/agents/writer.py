"""The Writer agent: turn research notes into the answer.

This agent takes over the *second half* of Phase 5's `call_model`: writing the
answer. In 6.3 it will also absorb the revising half, reading a critique beside
the notes. In 6.1 there is no reviewer yet, so it only drafts.

--------------------------------------------------------------------------
CONCEPT: why the Writer is a node and the Researcher is a subgraph
--------------------------------------------------------------------------
Neither shape is "more agent-like" than the other. The question is only whether
the agent's work has internal steps worth making into graph steps.

  Researcher   Several steps that depend on each other: search, read the
               result, decide the next search, stop. That loop needs its own
               messages and its own counter, and each step is worth seeing in
               Studio and LangSmith. It also produces scratch work (tool calls,
               raw results) that nobody else should read. A subgraph gives the
               loop real steps *and* a boundary to keep the scratch work behind.

  Writer       One model call. Notes in, prose out. No tools, so there is no
               loop. There is no scratch work, because its only output is the
               product. Wrap it in a subgraph and you get a box containing one
               box, a private channel with nothing to keep private, and one more
               level of nesting in every trace.

The general test: **make an agent a subgraph when it has a loop, or state it
must keep to itself. Otherwise make it a node.** An agent is a role in the
system, not a unit of graph structure, and the two do not have to line up.

If the Writer ever gains a loop - say, drafting section by section, or calling
a formatting tool - that is the point to promote it. The change would be local:
the parent graph adds a node called "writer" either way.

--------------------------------------------------------------------------
CONCEPT: what the Writer is allowed to know
--------------------------------------------------------------------------
The Writer reads `research_notes`, the question, the plan, and the shared
conversation. It does *not* see the Researcher's tool calls, and it has no
tools of its own. That split is deliberate. A writer that can search is a
second researcher, and its searches would be judged by nobody: they would not
appear in `research_notes`, so in 6.3 the Critic would be checking a draft
against evidence it was never shown.

So when the notes are thin, the Writer's job is to *say* they are thin. It is
not the Writer's job to fill the gap from memory. In 6.2, the Supervisor is what
notices "thin notes" and sends the work back to the Researcher.

--------------------------------------------------------------------------
6.3: revising
--------------------------------------------------------------------------
This node now does what Phase 5's `call_model` did on a revision. When
`revisions > 0`, the request carries Phase 5's REVISION_INSTRUCTIONS, with the
previous draft and both reviewers' notes, each labelled with its author. What
changed is how the Writer got here. Phase 5 routed every rejection "back to
call_model". Here a rejection goes to the Supervisor, and the Writer is
dispatched only if the Supervisor judged the problem to be a *writing* problem.
An evidence problem goes to the Researcher first, and the Writer then revises
from the merged notes.

The Writer also records its own spend in `budgets["writer"]` (one per draft),
and writes only that entry. See BUDGET_ENTRY_OWNERS.
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

import re

from research_copilot.models import get_chat_model
from research_copilot.multi_agent_state import MultiAgentState, budget_of
from research_copilot.request_budget import RETRY_FACTOR, estimate, fit_text, max_request_tokens, trim_record
from research_copilot.resilience import ModelCallFailure, invoke_with_recovery
from research_copilot.tools.citations import extract_citations
# Phase 5's revision instruction, reused verbatim: the Writer is the node that
# inherits call_model's revising half, so it inherits the prompt too.
from research_copilot.prompts import REVISION_INSTRUCTIONS

# Drafts per revision round. 2 leaves room for one rewrite after a mid-round
# re-research, on top of the first draft.
DEFAULT_MAX_WRITER_CALLS = 2

# The Writer's prompt. It has two variants because the modes make different
# promises about sources. The shared part is the discipline: cite what the notes
# name, and nothing else.
WRITER_SYSTEM_PROMPT = (
    "You are the Writer on a small research team. A Researcher has already "
    "gathered evidence and handed you their notes. You write the answer the "
    "user will read.\n\n"
    "Rules:\n"
    "- Base the answer on the research notes. Cite sources exactly as the notes "
    "give them - same titles, same URLs. Never add a source the notes do not "
    "contain.\n"
    "- Distinguish established findings from speculation, and say plainly when "
    "the evidence is thin.\n"
    "- If the notes list gaps, or do not cover part of the question, say what is "
    "missing instead of filling it in from memory.\n"
    "- Write for the user: a direct, connected answer. Do not mention the "
    "Researcher, the notes, or this process."
)

WRITER_KB_RULES = (
    "\n\nThis is a knowledge-base question. The notes are numbered excerpts from "
    "the user's own documents. Answer only from those excerpts - no prior "
    "knowledge, even if you are confident it is correct - and cite them as [1], "
    "[2], ... matching their numbers."
)

# What the Writer is shown when the Researcher handed over nothing. It is an
# explicit sentence, not an empty section. A model given a blank "Research
# notes:" heading reads it as "no constraints" and answers from memory, which
# is exactly the failure this system exists to avoid.
NO_NOTES = (
    "(The Researcher found nothing usable for this question. Tell the user that "
    "no supporting evidence was found, and do not answer from memory.)"
)


# --------------------------------------------------------------------------
# Phase 7 A1: citations that do not come from the notes
# --------------------------------------------------------------------------
# CONCEPT: a check, not a normaliser
# On Groq, gpt-oss-120b wrote its own citation markup into a user-facing answer
# - `【1†L1-L7】`, a reference into a browsing tool it does not have here. It
# points at nothing the user can check, and the Writer's rule ("cite exactly
# as the notes give them") did not stop it. Nothing noticed.
#
# The fix is deliberately narrow: detect citations that cannot be traced to the
# notes, give the Writer one retry that names them, and record what is left in
# `unsupported_citations`. It does not rewrite citation formats. Full
# normalisation is a separate, later decision. What matters now is that bad
# output is no longer *silent*.
_MARKUP = re.compile(r"【([^】]*)】")
_BRACKET = re.compile(r"(?<!\w)\[(\d+)\]")
_EXCERPT_NUMBER = re.compile(r"^\[(\d+)\]", re.MULTILINE)

UNSUPPORTED_CITATIONS_NOTE = (
    "Your draft cites things that do not come from the research notes: {items}. "
    "Rewrite the draft citing only sources the notes contain, exactly as the "
    "notes give them (the arXiv URL or id, or [n] for numbered excerpts). Remove "
    "any other citation or reference marker."
)


def unsupported_citations(draft: str, notes: str, *, mode: str = "live-search") -> list[str]:
    """Citations in `draft` that cannot be traced to `notes`, as readable strings."""
    notes_ids = set(extract_citations(notes))
    issues: list[str] = []
    for cited in extract_citations(draft):
        if cited not in notes_ids:
            issues.append(f"arXiv {cited}")
    for inner in dict.fromkeys(_MARKUP.findall(draft or "")):
        if not (set(extract_citations(inner)) & notes_ids):
            issues.append(f"【{inner}】")
    if mode == "knowledge-base":
        excerpts = {int(n) for n in _EXCERPT_NUMBER.findall(notes or "")}
        for n in sorted({int(n) for n in _BRACKET.findall(draft or "")} - excerpts):
            issues.append(f"[{n}] (no excerpt [{n}] in the notes)")
    return issues


def revision_instruction(state: MultiAgentState, max_revisions: int) -> list[BaseMessage]:
    """Phase 5's `_revision_instruction`, reading 6.3's per-reviewer fields.

    `revisions > 0` is still the signal, for Phase 5's reason: the counter
    already knows, and a separate flag would be a second thing to keep in sync.
    Both reviewers' notes are included and labelled by source. When they
    disagree, the Writer needs to see who said what.
    """
    if state.get("revisions", 0) <= 0:
        return []
    feedback = []
    if state.get("critique") and state.get("verdict") == "reject":
        feedback.append(f"- Critic: {state['critique']}")
    if state.get("human_feedback") and state.get("human_verdict") == "reject":
        feedback.append(f"- Human reviewer: {state['human_feedback']}")
    if not feedback:
        feedback.append(
            "- The draft was rejected without a reason being recorded. Re-read "
            "the question and the notes and write the strongest answer you can."
        )
    return [
        SystemMessage(
            content=REVISION_INSTRUCTIONS.format(
                attempt=state.get("revisions", 0),
                cap=max_revisions,
                draft=(state.get("draft") or "").strip() or "(the previous draft was empty)",
                feedback="\n".join(feedback),
            )
        )
    ]


def make_writer(
    *,
    model: BaseChatModel | None = None,
    max_calls: int = DEFAULT_MAX_WRITER_CALLS,
    max_revisions: int = 0,
):
    """Build the Writer node.

    It returns a plain function, not a compiled graph - see the top of this
    file. The model is resolved on first call, for the same reason every other
    model in the project is: compiling or drawing the graph must not need a key.
    """
    _cache: dict[str, BaseChatModel] = {}

    def writer_model() -> BaseChatModel:
        if "model" not in _cache:
            # No bind_tools. The Writer has no tools, on purpose - see "what the
            # Writer is allowed to know" above.
            _cache["model"] = model or get_chat_model()
        return _cache["model"]

    def write_draft(state: MultiAgentState) -> dict:
        """Notes in, draft out.

        The request is built as: instructions, then the earlier conversation,
        then one final human turn carrying the notes *and* the question.

        The notes go in that final turn, not in a system message, for the
        reason Phase 2's RAG_PROMPT put retrieved context beside the question.
        The model should read the evidence as belonging to this particular
        question, not as a standing instruction. The question already sits at
        the end of `messages` as a HumanMessage (the turn boundary put it
        there), so it is replaced by the combined turn rather than repeated.
        """
        mode = state.get("mode", "live-search")
        system = WRITER_SYSTEM_PROMPT + (WRITER_KB_RULES if mode == "knowledge-base" else "")

        instructions: list[BaseMessage] = [SystemMessage(content=system)]
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
                        "Make sure the answer covers each one, as a single "
                        "connected answer - not a list of separate replies.\n" + listed
                    )
                )
            )

        # Earlier turns only. Because the Researcher's tool loop is private,
        # this history is clean - human turns and committed answers, nothing
        # else. A Phase 5 transcript would have carried every tool call here.
        # 6.3: the revision instruction goes last among the instructions,
        # closest to the turn it asks to be redone - same placement as Phase 5.
        instructions.extend(revision_instruction(state, max_revisions))

        history = list(state.get("messages", []))
        if history and isinstance(history[-1], HumanMessage):
            history = history[:-1]

        notes = state.get("research_notes", "").strip() or NO_NOTES
        question_part = f"\n\nQuestion: {state.get('question', '')}"
        limit = max_request_tokens()
        trims: list = []

        def build(budget: int, extra: list[BaseMessage] = ()) -> list[BaseMessage]:
            """The request, with the notes (the only material part) cut to fit
            `budget` (Phase 7, request_budget.py). `extra` is the retry turn,
            whose size has to be paid for out of the notes too."""
            base = estimate([*instructions, *history, HumanMessage(content="Research notes:\n" + question_part), *extra])
            fitted, before, after = fit_text(notes, budget - base)
            if after < before:
                trims.append(trim_record("writer", "research notes", before, after, budget))
            final_turn = HumanMessage(content=f"Research notes:\n{fitted}{question_part}")
            return [*instructions, *history, final_turn, *extra]

        def call(extra: list[BaseMessage] = ()) -> tuple[str, int]:
            request = build(limit, extra)
            result, attempts = invoke_with_recovery(
                writer_model(), request,
                recoverable=lambda exc: False, note="", where="writer",
                on_too_large=lambda: build(int(limit * RETRY_FACTOR), extra),
            )
            # A Writer whose request cannot be served produces no draft; the
            # Supervisor sees there is nothing to finish. It does not crash.
            return ("" if isinstance(result, ModelCallFailure) else result.text), attempts

        draft, calls = call()

        # Phase 7: one retry, naming the citations that do not come from the
        # notes. The same shape as resilience.py's recovery - a specific hint,
        # once - applied to bad *output* rather than a failed *call*. Checked
        # against the full notes, not the trimmed copy the model saw.
        issues = unsupported_citations(draft, notes if notes != NO_NOTES else "", mode=mode)
        if issues:
            draft, more = call([AIMessage(content=draft),
                                HumanMessage(content=UNSUPPORTED_CITATIONS_NOTE.format(items="; ".join(issues)))])
            calls += more
            issues = unsupported_citations(draft, notes if notes != NO_NOTES else "", mode=mode)

        used = budget_of(state, "writer", max_calls)["used"] + calls
        # The Writer's fields: its draft, what is still unsupported in it, its
        # request-size trims, and its own budget entry. `owns("writer")` checks
        # all four.
        update = {
            "draft": draft,
            "unsupported_citations": issues,
            "budgets": {"writer": {"used": used, "cap": max_calls}},
        }
        if trims:
            update["writer_trims"] = [*(state.get("writer_trims") or []), *trims]
        return update

    return write_draft
