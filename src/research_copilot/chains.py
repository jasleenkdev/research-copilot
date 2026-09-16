"""LCEL chains.

CONCEPT: LCEL (LangChain Expression Language)
`a | b | c` builds a RunnableSequence: the output of `a` becomes the input of
`b`, and so on. Every step is a Runnable, so the whole chain is one too and
gets .invoke / .batch / .stream / .ainvoke for free. LangSmith traces it as a
single parent run with one child run per step.

    ANSWER_PROMPT     dict       -> ChatPromptValue (a list of messages)
    model             messages   -> AIMessage
    StrOutputParser   AIMessage  -> str

Why chain the steps instead of calling them one after another yourself?
Streaming flows through the chain (the parser emits text as the model streams),
.batch() runs the whole pipeline over many inputs concurrently, and each step is
a replaceable part. LangGraph nodes are built from these same parts.

The builders take an optional `model` so tests can pass in a fake one. A chain
only needs each step to be a Runnable with the right input and output types.
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.runnables import Runnable

from research_copilot.models import get_chat_model
from research_copilot.prompts import ANSWER_PROMPT, STRUCTURED_ANSWER_PROMPT
from research_copilot.schemas import ResearchAnswer


def build_answer_chain(model: BaseChatModel | None = None) -> Runnable[dict, str]:
    """{question} -> plain-text answer."""
    model = model or get_chat_model()

    # CONCEPT: StrOutputParser
    # A chat model returns an AIMessage, not a string. AIMessage.content is
    # either a string or a list of content blocks. With Claude it's usually a
    # list: "thinking" blocks, then "text" blocks. StrOutputParser keeps only
    # the text, so the chain produces a plain `str` you can print, store, or pass
    # to the next step.
    return ANSWER_PROMPT | model | StrOutputParser()


def build_structured_chain(
    model: BaseChatModel | None = None,
) -> Runnable[dict, ResearchAnswer]:
    """{question, notes} -> validated ResearchAnswer."""
    model = model or get_chat_model()

    # CONCEPT: PydanticOutputParser
    # It fills the same slot in the chain as StrOutputParser, but it parses the
    # text as JSON (Markdown ```json fences are fine) and validates the result
    # into a ResearchAnswer. It also writes the prompt text that describes the
    # schema: get_format_instructions().
    #
    # This approach asks nicely, then validates. It works with any model, but
    # nothing forces the model to comply, so bad output raises
    # OutputParserException. The alternative, model.with_structured_output(...),
    # has the API itself constrain the output. It's worth comparing once you've
    # seen how this version fails.
    parser = PydanticOutputParser(pydantic_object=ResearchAnswer)

    # CONCEPT: .partial()
    # Fills some template variables when the chain is built, so callers only
    # pass the per-request ones ({question}, {notes}). Values supplied this way
    # are not parsed as template text, so the braces in the JSON schema are safe.
    prompt = STRUCTURED_ANSWER_PROMPT.partial(
        format_instructions=parser.get_format_instructions()
    )
    return prompt | model | parser
