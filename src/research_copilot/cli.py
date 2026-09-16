"""Command-line entry point: `research-copilot <command> ...`.

Phase 1 commands, in the order the concepts were introduced:
  messages    call the model directly with hand-built message objects
  answer      prompt | model | StrOutputParser
  structured  prompt | model | PydanticOutputParser
  agent       model.bind_tools(...) + a hand-written tool-call loop

Phase 2 commands:
  chat        multi-turn conversation with managed (trimmed or summarized) memory
  ingest      load, chunk, embed, and store documents in Chroma
  ask-docs    RAG: retrieve from the ingested documents, answer from them only
  ask         the same question answered in either mode (knowledge base or live
              search), which is the choice a Supervisor agent will make in Phase 6
"""

import argparse
import sys

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from research_copilot.agent_loop import run_tool_loop
from research_copilot.chains import build_answer_chain, build_structured_chain
from research_copilot.config import get_settings
from research_copilot.ingest import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, ingest_path
from research_copilot.memory import ConversationMemory, memory_from_settings
from research_copilot.models import get_chat_model
from research_copilot.prompts import CHAT_PROMPT, RESEARCHER_PERSONA
from research_copilot.retrieval import (
    build_rag_chain,
    count_chunks,
    describe_source,
    get_retriever,
    get_vector_store,
)


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


# --- Phase 2: conversation memory ---------------------------------------------


def _print_memory(memory: ConversationMemory) -> None:
    print(f"--- memory ({memory.strategy}, {memory.token_count()} tokens) ---")
    if memory.summary:
        print(f"[summary] {memory.summary}")
    for message in memory.messages:
        preview = " ".join(message.text.split())
        if len(preview) > 90:
            preview = preview[:90] + "..."
        print(f"  {message.type:>6}: {preview}")
    print("--- end memory ---")


def cmd_chat(strategy: str | None, max_tokens: int | None) -> None:
    model = get_chat_model()
    memory = memory_from_settings(
        strategy=strategy, max_tokens=max_tokens, summary_model=model
    )

    # The chat chain is `prompt | model` with no output parser, because history
    # stores AIMessage objects, not strings. Parsing to text here would throw
    # away tool calls and thinking blocks that have to be replayed.
    chain = CHAT_PROMPT | model

    print(
        f"Chat mode. Memory strategy: {memory.strategy}, "
        f"budget: {memory.max_tokens} tokens."
    )
    print("Type /memory to inspect the history, /exit to quit.\n")

    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue
        if user_input in {"/exit", "/quit"}:
            break
        if user_input == "/memory":
            _print_memory(memory)
            continue

        # One turn: record the question, resend the whole (pruned) history,
        # record the reply, then prune for next time.
        memory.add_user_turn(user_input)
        ai_message = chain.invoke(memory.prompt_variables())
        memory.add_ai_turn(ai_message)
        print(f"copilot> {ai_message.text}\n")

        report = memory.prune()
        print(f"[memory] {report}", file=sys.stderr)


# --- Phase 2: retrieval -------------------------------------------------------


def cmd_ingest(path: str, chunk_size: int, chunk_overlap: int) -> None:
    print(
        f"[ingest] chunk_size={chunk_size} overlap={chunk_overlap} "
        "(first run downloads the embedding model)",
        file=sys.stderr,
    )
    report = ingest_path(path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    for name in report.files:
        print(f"  loaded {name}")
    for name in report.skipped:
        print(f"  skipped (unsupported type) {name}", file=sys.stderr)
    if not report.files:
        print("No supported files found (.txt, .md, .pdf).", file=sys.stderr)
        return
    print(
        f"{report.documents} documents -> {report.chunks} chunks. "
        f"Collection now holds {report.total_in_store} chunks."
    )


def _warn_if_empty_store() -> bool:
    total = count_chunks(get_vector_store())
    if total == 0:
        print(
            "The vector store is empty. Run `research-copilot ingest <path>` first.",
            file=sys.stderr,
        )
        return True
    return False


def cmd_ask_docs(question: str, k: int | None) -> None:
    if _warn_if_empty_store():
        return
    chain = build_rag_chain(retriever=get_retriever(k=k))
    result = chain.invoke(question)

    print(result["answer"])
    print("\n--- retrieved excerpts ---", file=sys.stderr)
    for i, document in enumerate(result["docs"], start=1):
        print(f"[{i}] {describe_source(document)}", file=sys.stderr)


def cmd_ask(question: str, mode: str, k: int | None, max_iterations: int) -> None:
    """One question, two modes.

    Right now *you* pick the mode with a flag. In Phase 6 a Supervisor agent will
    make this same choice from the question itself, which is why both paths are
    deliberately given the same shape: question in, grounded answer out.
    """
    if mode == "knowledge-base":
        cmd_ask_docs(question, k)
        return

    result = run_tool_loop(
        question,
        max_iterations=max_iterations,
        on_event=lambda event: print(event, file=sys.stderr),
    )
    print(result.answer)


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
        prog="research-copilot", description="Research Copilot (Phases 1-2)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name, help_text in {
        "messages": "Raw model call with message objects; prints AIMessage anatomy",
        "answer": "prompt | model | StrOutputParser (streamed)",
        "structured": "prompt | model | PydanticOutputParser",
    }.items():
        sub = subparsers.add_parser(name, help=help_text)
        sub.add_argument("question")

    agent_parser = subparsers.add_parser(
        "agent", help="Model + arXiv tool + hand-written tool-call loop"
    )
    agent_parser.add_argument("question")
    agent_parser.add_argument("--max-iterations", type=int, default=6)
    agent_parser.add_argument(
        "--structured",
        action="store_true",
        help="Pass the agent's findings through the structured chain",
    )

    chat_parser = subparsers.add_parser(
        "chat", help="Interactive multi-turn chat with managed memory"
    )
    chat_parser.add_argument(
        "--memory",
        choices=["trim", "summarize"],
        default=None,
        help="Override RESEARCH_COPILOT_MEMORY_STRATEGY for this session",
    )
    chat_parser.add_argument(
        "--max-history-tokens",
        type=int,
        default=None,
        help="Token budget for the conversation history",
    )

    ingest_parser = subparsers.add_parser(
        "ingest", help="Load, chunk, embed, and store .txt/.md/.pdf files"
    )
    ingest_parser.add_argument("path", help="A file or a directory to walk")
    ingest_parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    ingest_parser.add_argument(
        "--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP
    )

    ask_docs_parser = subparsers.add_parser(
        "ask-docs", help="RAG over ingested documents (same as `ask --mode knowledge-base`)"
    )
    ask_docs_parser.add_argument("question")
    ask_docs_parser.add_argument(
        "-k", type=int, default=None, help="How many chunks to retrieve"
    )

    ask_parser = subparsers.add_parser(
        "ask", help="Answer from the knowledge base or from live arXiv search"
    )
    ask_parser.add_argument("question")
    ask_parser.add_argument(
        "--mode",
        choices=["knowledge-base", "live-search"],
        default="knowledge-base",
        help="knowledge-base: RAG over ingested docs. live-search: the Phase 1 arXiv agent.",
    )
    ask_parser.add_argument("-k", type=int, default=None)
    ask_parser.add_argument("--max-iterations", type=int, default=6)

    args = parser.parse_args(argv)
    _report_tracing()

    try:
        if args.command == "messages":
            cmd_messages(args.question)
        elif args.command == "answer":
            cmd_answer(args.question)
        elif args.command == "structured":
            cmd_structured(args.question)
        elif args.command == "agent":
            cmd_agent(args.question, args.max_iterations, args.structured)
        elif args.command == "chat":
            cmd_chat(args.memory, args.max_history_tokens)
        elif args.command == "ingest":
            cmd_ingest(args.path, args.chunk_size, args.chunk_overlap)
        elif args.command == "ask-docs":
            cmd_ask_docs(args.question, args.k)
        elif args.command == "ask":
            cmd_ask(args.question, args.mode, args.k, args.max_iterations)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except OutputParserException as exc:
        print(f"The model's output did not match the ResearchAnswer schema:\n{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
