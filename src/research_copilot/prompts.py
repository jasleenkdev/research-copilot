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
