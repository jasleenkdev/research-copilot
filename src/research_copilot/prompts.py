"""Prompt templates.

CONCEPT: Message types
Chat models don't take one big string. They take a list of role-tagged messages:

    SystemMessage   instructions from you, the developer (persona, rules)
    HumanMessage    the user's turn
    AIMessage       the model's turn: text, and possibly `tool_calls`
    ToolMessage     the result of running a tool, linked to the request by
                    `tool_call_id`

LangChain uses these same classes for every provider, and each integration
translates them to its own wire format. For example, ChatAnthropic moves
SystemMessage into the API's top-level `system` field and sends ToolMessage as
a `tool_result` block.

CONCEPT: ChatPromptTemplate
Turns a dict of variables into a list of messages. Why not an f-string?
  1. It's a Runnable, so it composes with `|`.
  2. It checks that every required variable was provided.
  3. LangSmith traces show the template and the variables separately, which
     makes debugging prompts much easier.
  4. It can hold whole message lists (MessagesPlaceholder), which is how
     conversation memory is added in Phase 2.

The ("system", "...") tuple is shorthand for SystemMessagePromptTemplate.
`{name}` marks a variable, so a literal brace must be doubled: `{{` / `}}`.
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# describe the system personality  or instruction 
RESEARCHER_PERSONA = (
    "You are Research Copilot, a careful research assistant. Answer the user's "
    "question accurately and concisely. Distinguish established findings from "
    "speculation, and say plainly when you are unsure."
)

# "Whenever I give you a question, construct these messages."
# 1) Plain answer: {question} -> prose.
ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", RESEARCHER_PERSONA),
        ("human", "{question}"),
    ]
)

# 2) Structured answer. {format_instructions} is filled in chains.py via
# .partial(), using the text PydanticOutputParser generates from the schema.
# The schema itself stays defined in exactly one place: schemas.py.
STRUCTURED_ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", RESEARCHER_PERSONA + "\n\n{format_instructions}"),
        (
            "human",
            "Question: {question}\n\n"
            "Research notes gathered so far (may be empty):\n{notes}",
        ),
    ]
)

# 3) Tool-using agent. The tool's own name, description, and argument schema
# reach the model through bind_tools(). This prompt only covers *when* to use it.
AGENT_SYSTEM_PROMPT = RESEARCHER_PERSONA + (
    "\n\nYou can search arXiv. Use it when the question is about research "
    "literature or recent methods, or when citing specific papers would help. "
    "Skip it for questions you can answer reliably without sources. When you cite "
    "a paper, give its title and arXiv URL. If a search fails or finds nothing "
    "useful, say so rather than inventing references."
)

AGENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", AGENT_SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)


# --- Phase 2: conversation memory ---------------------------------------------

# CONCEPT: MessagesPlaceholder
# The templates above build fixed message lists. A conversation needs a slot
# where a whole *list* of past messages gets spliced in. That's
# MessagesPlaceholder: `history` is filled with the running message list from
# memory.py, so the persona stays pinned at the top while the conversation grows
# underneath it.
#
# `optional=True` means the variable may be omitted entirely. The trimming
# strategy never sets `earlier_summary`; the summarization strategy fills it with
# one SystemMessage covering the turns that were folded away. (ChatAnthropic
# merges several leading system messages into the API's `system` field, so two
# system messages are fine here.)
CHAT_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", RESEARCHER_PERSONA),
        MessagesPlaceholder("earlier_summary", optional=True),
        MessagesPlaceholder("history"),
    ]
)

# Used by the summarization strategy. It folds the previous summary and the
# about-to-be-dropped turns into one new summary, so summarizing repeatedly
# doesn't lose everything that came before.
SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You compress conversation history for an AI assistant. Write a terse "
            "third-person summary that preserves facts, decisions, user "
            "preferences, and open questions. Keep any specific names, numbers, "
            "and sources. Drop pleasantries. Never invent detail that isn't there.",
        ),
        (
            "human",
            "Summary so far (may be empty):\n{previous_summary}\n\n"
            "New turns to fold in:\n{conversation}\n\n"
            "Write the updated summary.",
        ),
    ]
)


# --- Phase 2: retrieval (RAG) -------------------------------------------------

# CONCEPT: grounding the model in retrieved text
# The retrieved chunks go in as ordinary prompt text. Nothing about the model
# changes; retrieval is just prompt construction with a search step in front.
# Two instructions matter most, because the default failure mode of RAG is a
# fluent answer built from the model's own memory rather than your documents:
#   1. answer only from the context
#   2. say so explicitly when the context doesn't cover the question
# The [n] citation rule makes the first one checkable: if an answer cites
# nothing, it probably didn't come from the documents.
RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are Research Copilot answering from a fixed set of retrieved "
            "document excerpts.\n\n"
            "Rules:\n"
            "- Use only the context below. Do not use prior knowledge, even if "
            "you are confident it is correct.\n"
            "- If the context does not contain the answer, say exactly what is "
            "missing and stop. Do not guess or fill gaps.\n"
            "- Cite the excerpts you used as [1], [2], ... matching their numbers.\n"
            "- Quote sparingly; summarize in your own words.",
        ),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)


# --- Phase 5: reflection (critique -> revise) ---------------------------------

# CONCEPT: an LLM in the reviewer's seat
# `review_draft` puts a *question* to a human and waits. `critique_draft` puts
# the same question to a model and doesn't. For the two to be interchangeable,
# the model's reply has to reduce to the same thing a human's verdict reduces
# to: a decision, plus a reason.
#
# Hence the rigid first line. The critic is asked for APPROVE or REJECT on its
# own line, and everything after it is the reason - which is exactly the
# `{"decision": ..., "note": ...}` shape `review_draft` already parses. The
# parsing lives in `_parse_critique` in graph.py, and it hands its result to the
# same `_parse_verdict` the human path uses, so an unreadable critique fails
# closed the same way an unreadable human verdict does.
#
# Why not PydanticOutputParser here, given schemas.py exists? Two reasons worth
# knowing. A verdict is one enum and one string, so a schema buys little beyond
# what a first-line convention already gives. And a parse failure on this path
# is not free: it fails closed to "reject", which spends a revision round. The
# looser format fails less often, and `max_revisions` is what bounds the damage
# when it does. Phase 6's Critic *agent*, whose verdict routes a supervisor
# rather than a single edge, is where structured output starts earning its cost.
#
# The instruction to be specific is load-bearing, not politeness. "Too vague"
# is a critique `call_model` cannot act on, so the revision comes back the same
# and the loop burns its whole budget rediscovering that. Asking for concrete,
# addressable objections is the difference between a reflection loop and an
# expensive no-op.
CRITIC_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a demanding but fair reviewer of research answers. You are "
            "reviewing a draft written by another assistant. You are not "
            "rewriting it - you are deciding whether it is good enough to send.\n\n"
            "Judge it on:\n"
            "- Does it actually answer the question that was asked?\n"
            "- Are claims supported, and are sources real and cited where they "
            "matter?\n"
            "- Is speculation labelled as speculation?\n"
            "- Are there obvious gaps, errors, or unsupported leaps?\n\n"
            "Reply in exactly this format:\n"
            "First line: APPROVE or REJECT, alone on the line.\n"
            "Then, if you rejected it, say what is wrong in concrete terms the "
            "writer can act on: name the claim that needs support, the part of "
            "the question left unanswered, the source that is missing. Do not "
            "write vague notes like 'needs more detail' - a note the writer "
            "cannot act on wastes a revision.\n\n"
            "Approve a draft that is good enough. Holding out for perfect costs "
            "revisions and gets you nothing.",
        ),
        (
            "human",
            "Question: {question}\n\n"
            "Sub-questions the plan called for (may be empty):\n{sub_questions}\n\n"
            "Draft answer:\n{draft}",
        ),
    ]
)

# The instruction `call_model` receives when it is re-drafting rather than
# drafting. Not a ChatPromptTemplate: it is rendered into a SystemMessage that
# `call_model` splices into a request it is already assembling by hand, and both
# modes (knowledge-base and live-search) need the same text even though they
# build completely different requests around it. A template would have to be
# invoked twice from two different places to produce one string.
REVISION_INSTRUCTIONS = (
    "You are revising an answer you already wrote. It was reviewed and sent "
    "back. This is revision attempt {attempt} of at most {cap}.\n\n"
    "Your previous draft:\n{draft}\n\n"
    "What the reviewers said:\n{feedback}\n\n"
    "Write a new, complete answer that addresses the feedback. Do not reply to "
    "the reviewers, do not explain what you changed, and do not apologise - "
    "produce the answer itself, in full, as if writing it for the first time. "
    "If a criticism is too vague to act on, or you believe it is wrong, say so "
    "briefly inside the answer and make the best version you can rather than "
    "returning the same draft unchanged."
)


# --- Phase 5: planning --------------------------------------------------------

# CONCEPT: decomposition before research
# A question like "how do RAG and long-context models compare on cost and
# accuracy?" is really four questions. A single retrieval against the whole
# sentence finds chunks that are vaguely about all of it and precisely about
# none of it; a single tool call searches for one blurry thing. Splitting first
# gives the retriever and the tool loop narrower targets.
#
# The instruction that matters most is the one telling the model *not* to split.
# A model asked to decompose will always decompose - it will turn "who wrote the
# BERT paper?" into four sub-questions and turn one cheap lookup into four. So
# NONE is a first-class answer with its own line in the format, and the prompt
# says plainly that most questions deserve it. A planner that never declines to
# plan is a cost multiplier, not a planner.
PLAN_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You break research questions into sub-questions, but only when "
            "that genuinely helps.\n\n"
            "Most questions do not need it. A question needs decomposing only "
            "when it has several distinct parts that would be researched "
            "separately - a comparison across several dimensions, a question "
            "with a prerequisite that must be established first, or one that "
            "spans clearly different topics.\n\n"
            "Reply in exactly this format:\n"
            "- If the question is best researched as a single question, reply "
            "with the single word NONE and nothing else.\n"
            "- Otherwise reply with one sub-question per line, each starting "
            "with '- ', and nothing else. No preamble, no numbering, no "
            "commentary. Give at most {max_sub_questions}.\n\n"
            "Each sub-question must be self-contained and answerable on its "
            "own: no pronouns pointing back at the original question, no 'the "
            "above'. They will be searched independently.",
        ),
        ("human", "Question: {question}\n\nResearch mode: {mode}"),
    ]
)
