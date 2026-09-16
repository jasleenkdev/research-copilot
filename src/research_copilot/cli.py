"""Command-line entry point: `research-copilot <mode> "question"`.

The modes follow the order Phase 1 introduces the concepts:
  messages    call the model directly with hand-built message objects
  answer      prompt | model | StrOutputParser
  structured  prompt | model | PydanticOutputParser
  agent       model.bind_tools(...) + a hand-written tool-call loop
"""

import argparse
import sys

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from research_copilot.agent_loop import run_tool_loop
from research_copilot.chains import build_answer_chain, build_structured_chain
from research_copilot.config import get_settings
from research_copilot.models import get_chat_model
from research_copilot.prompts import RESEARCHER_PERSONA


def cmd_messages(question: str) -> None:
    # No template and no parser: message objects in, AIMessage out. Everything
    # else in Phase 1 is a layer on top of this one call.
    model = get_chat_model()
    ai_message = model.invoke(
        [SystemMessage(content=RESEARCHER_PERSONA), HumanMessage(content=question)]
    )
    print(ai_message.text)

    print("\n--- AIMessage anatomy ---")
    if isinstance(ai_message.content, list):
        print("content blocks:", [block.get("type") for block in ai_message.content])
    else:
        print("content: plain string")
    print("usage_metadata:", ai_message.usage_metadata)
    print("stop_reason:", ai_message.response_metadata.get("stop_reason"))
    print("model:", ai_message.response_metadata.get("model_name"))


def cmd_answer(question: str) -> None:
    chain = build_answer_chain()
    # .stream() works on the whole chain because each step supports streaming.
    # StrOutputParser turns every AIMessageChunk into a string chunk as it arrives.
    for chunk in chain.stream({"question": question}):
        print(chunk, end="", flush=True)
    print()


def cmd_structured(question: str) -> None:
    answer = build_structured_chain().invoke({"question": question, "notes": "(none)"})
    print(answer.model_dump_json(indent=2))


def cmd_agent(question: str, max_iterations: int, structured: bool) -> None:
    result = run_tool_loop(
        question,
        max_iterations=max_iterations,
        on_event=lambda event: print(event, file=sys.stderr),
    )
    print(
        f"[done] iterations={result.iterations} tool_calls={result.tool_calls_made}"
        + (" (stopped early)" if result.stopped_early else ""),
        file=sys.stderr,
    )

    if not structured:
        print(result.answer)
        return

    # Composition across pieces: the agent's raw tool results plus its draft
    # become the {notes} for the structured chain. This way the sources come from
    # real search results, not the model's memory.
    tool_outputs = [m.text for m in result.messages if isinstance(m, ToolMessage)]
    notes = "\n\n".join([*tool_outputs, f"Draft answer:\n{result.answer}"])
    answer = build_structured_chain().invoke({"question": question, "notes": notes})
    print(answer.model_dump_json(indent=2))


def _report_tracing() -> None:
    settings = get_settings()
    if settings.tracing_enabled and not settings.langsmith_api_key_set:
        print(
            "[tracing] LANGSMITH_TRACING=true but LANGSMITH_API_KEY is empty; "
            "traces will not upload.",
            file=sys.stderr,
        )
    elif settings.tracing_enabled:
        print(
            f"[tracing] LangSmith project: {settings.langsmith_project}",
            file=sys.stderr,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="research-copilot", description="Research Copilot (Phase 1)"
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    modes = {
        "messages": "Raw model call with message objects; prints AIMessage anatomy",
        "answer": "prompt | model | StrOutputParser (streamed)",
        "structured": "prompt | model | PydanticOutputParser",
        "agent": "Model + arXiv tool + hand-written tool-call loop",
    }
    for name, help_text in modes.items():
        sub = subparsers.add_parser(name, help=help_text)
        sub.add_argument("question")
        if name == "agent":
            sub.add_argument("--max-iterations", type=int, default=6)
            sub.add_argument(
                "--structured",
                action="store_true",
                help="Pass the agent's findings through the structured chain",
            )

    args = parser.parse_args(argv)
    _report_tracing()

    try:
        if args.mode == "messages":
            cmd_messages(args.question)
        elif args.mode == "answer":
            cmd_answer(args.question)
        elif args.mode == "structured":
            cmd_structured(args.question)
        elif args.mode == "agent":
            cmd_agent(args.question, args.max_iterations, args.structured)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except OutputParserException as exc:
        print(f"The model's output did not match the ResearchAnswer schema:\n{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
