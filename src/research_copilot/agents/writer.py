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
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from research_copilot.models import get_chat_model
from research_copilot.multi_agent_state import MultiAgentState

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


def make_writer(*, model: BaseChatModel | None = None):
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
        history = list(state.get("messages", []))
        if history and isinstance(history[-1], HumanMessage):
            history = history[:-1]

        notes = state.get("research_notes", "").strip() or NO_NOTES
        final_turn = HumanMessage(
            content=f"Research notes:\n{notes}\n\nQuestion: {state.get('question', '')}"
        )

        ai_message = writer_model().invoke([*instructions, *history, final_turn])
        # The Writer's one field. `owns("writer")` in the graph checks that this
        # stays true.
        return {"draft": ai_message.text}

    return write_draft
