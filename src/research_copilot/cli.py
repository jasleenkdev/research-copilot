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

Phase 3 commands:
  graph-agent     the same work as `ask`, rebuilt as a LangGraph StateGraph
  prebuilt-agent  the live-search path via langgraph.prebuilt.create_react_agent
  draw-graph      print either graph's structure without calling a model

Phase 4 commands:
  graph-chat      multi-turn conversation against one thread_id, persisted
  review          show a paused thread's draft and approve / reject / edit it
  threads         list the conversations in the checkpointer

`graph-agent` also gains --thread, --checkpointer and --approve, which is what
makes two separate CLI invocations continue the same conversation.
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
from research_copilot.checkpointing import (
    KINDS,
    checkpointer_scope,
    count_checkpoints,
    describe_checkpointer,
    list_thread_ids,
    new_thread_id,
    thread_config,
)
from research_copilot.graph import (
    build_graph,
    final_answer,
    pending_interrupt,
    resume_graph,
    run_graph,
)
from research_copilot.prebuilt import build_prebuilt_agent, run_prebuilt_agent
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


# --- Phase 3: the first LangGraph ---------------------------------------------


def _print_final_state(state: dict) -> None:
    """Show the whole final state, not just the answer.

    This is the habit Phase 3 is trying to build. `.invoke()` on a compiled
    graph returns every key every node wrote, and reading that dict is how you
    tell which path a run actually took - the transcript shape alone tells you
    whether the tool loop ran, and `documents` tells you whether retrieval did.
    """
    print("\n--- final state ---", file=sys.stderr)
    print(f"  question:   {state.get('question')}", file=sys.stderr)
    print(f"  mode:       {state.get('mode')}", file=sys.stderr)
    print(f"  iterations: {state.get('iterations')}", file=sys.stderr)

    # Phase 4 keys, printed only when they hold something, so a Phase 3-style
    # run's state dump looks exactly as it did before.
    if state.get("status") and state.get("status") != "drafting":
        print(f"  status:     {state.get('status')}", file=sys.stderr)
    if state.get("draft"):
        print(f"  draft:      {state['draft'][:70]}...", file=sys.stderr)
    if state.get("human_feedback"):
        print(f"  feedback:   {state.get('human_feedback')}", file=sys.stderr)
    if state.get("summary"):
        print(f"  summary:    {state['summary'][:70]}...", file=sys.stderr)

    documents = state.get("documents") or []
    if documents:
        print(f"  documents:  {len(documents)} retrieved", file=sys.stderr)
        for i, document in enumerate(documents, start=1):
            print(f"    [{i}] {describe_source(document)}", file=sys.stderr)

    print(f"  messages:   {len(state.get('messages', []))}", file=sys.stderr)
    for message in state.get("messages", []):
        preview = " ".join(message.text.split())
        if len(preview) > 70:
            preview = preview[:70] + "..."
        # Tool calls carry no text, so name them explicitly - otherwise the
        # AIMessage that drove a search looks like an empty turn.
        calls = getattr(message, "tool_calls", None)
        if calls:
            preview = (preview + " ") if preview else ""
            preview += "-> " + ", ".join(f"{c['name']}({c['args']})" for c in calls)
        print(f"    {message.type:>6}: {preview}", file=sys.stderr)
    print("--- end state ---", file=sys.stderr)


# --- Phase 4: persistence and human-in-the-loop -------------------------------


def _resolve_thread(thread_id: str | None) -> tuple[str, bool]:
    """Pick the thread_id for this run, and say whether it is a new one.

    CONCEPT: thread_id is yours to manage (see checkpointing.py).
    There is no "current thread" - LangGraph will not remember one for you, and
    a missing thread_id is not an error, just a conversation that accumulates
    nothing. So the CLI does the two halves explicitly: mint a UUID when you
    start, and take `--thread <id>` when you continue. The id is printed on
    every run precisely because the *next* invocation has to pass it back in.
    """
    if thread_id:
        return thread_id, False
    return new_thread_id(), True


def _announce_thread(thread_id: str, is_new: bool, saver) -> None:
    print(f"[thread] {thread_id}" + ("  (new)" if is_new else ""), file=sys.stderr)
    print(f"[checkpointer] {describe_checkpointer(saver)}", file=sys.stderr)
    if is_new and saver is not None:
        print(
            f"[thread] continue this conversation with: --thread {thread_id}",
            file=sys.stderr,
        )


def _print_pending(payload: dict) -> None:
    """Show a parked run's interrupt payload - the question put to the human."""
    print("--- awaiting approval ---")
    print(f"question: {payload.get('question', '')}")
    print(f"mode:     {payload.get('mode', '')}")
    print("\ndraft:")
    print(payload.get("draft", "") or "(empty draft)")
    print("--- end draft ---")


def cmd_graph_agent(
    question: str,
    mode: str,
    max_iterations: int,
    *,
    thread_id: str | None = None,
    checkpointer: str | None = None,
    approve: bool = False,
    memory_strategy: str | None = None,
    max_history_tokens: int | None = None,
) -> None:
    """One turn of the graph, optionally against a persistent thread.

    The Phase 4 change that matters is the `config` argument threaded through
    `run_graph`: the question goes into the state dict, the thread_id goes into
    the config, and running this command twice with the same `--thread` makes
    `state.messages` carry over between two *separate processes*.
    """
    if mode == "knowledge-base" and _warn_if_empty_store():
        return

    with checkpointer_scope(checkpointer) as saver:
        thread, is_new = _resolve_thread(thread_id)
        _announce_thread(thread, is_new, saver)

        graph = build_graph(
            max_iterations=max_iterations,
            checkpointer=saver,
            require_approval=approve,
            memory_strategy=memory_strategy,
            max_history_tokens=max_history_tokens,
        )
        state = run_graph(question, mode=mode, graph=graph, thread_id=thread)

        # A parked run returns early with __interrupt__ instead of an answer.
        if "__interrupt__" in state:
            _print_pending(state["__interrupt__"][0].value)
            print(
                f"\n[paused] resume with: research-copilot review --thread {thread}",
                file=sys.stderr,
            )
            _print_final_state(state)
            return

        print(final_answer(state))
        _print_final_state(state)


def cmd_graph_chat(
    mode: str,
    max_iterations: int,
    *,
    thread_id: str | None = None,
    checkpointer: str | None = None,
    approve: bool = False,
    memory_strategy: str | None = None,
    max_history_tokens: int | None = None,
) -> None:
    """Phase 2's `chat`, rebuilt on the checkpointer instead of a local object.

    Worth comparing the two directly. `cmd_chat` holds a `ConversationMemory`
    in a local variable, and the conversation exists because that variable does
    - close the process and it is gone. Here the loop holds nothing: every turn
    is a fresh `.invoke()` and the history comes back from the checkpointer,
    addressed by a thread_id that never changes between turns.

    That is the thread_id vs State distinction made concrete. `thread` is
    computed once, before the loop, and passed as *config* on every turn. The
    question changes each turn and goes in the *state*. If thread_id lived in
    State, each turn's input would overwrite it and there would be no way to say
    "same conversation, new question".
    """
    with checkpointer_scope(checkpointer) as saver:
        if saver is None:
            print(
                "[warning] --checkpointer none: each turn starts from an empty "
                "transcript, which makes this the same as running graph-agent "
                "repeatedly.",
                file=sys.stderr,
            )

        thread, is_new = _resolve_thread(thread_id)
        _announce_thread(thread, is_new, saver)

        graph = build_graph(
            max_iterations=max_iterations,
            checkpointer=saver,
            require_approval=approve,
            memory_strategy=memory_strategy,
            max_history_tokens=max_history_tokens,
        )

        print("\nType /state to dump the persisted state, /exit to quit.\n")

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
            if user_input == "/state":
                # Reading the thread without running it - see pending_interrupt.
                snapshot = graph.get_state(thread_config(thread))
                _print_final_state(snapshot.values)
                print(f"  next node: {snapshot.next or '(idle)'}", file=sys.stderr)
                continue

            state = run_graph(user_input, mode=mode, graph=graph, thread_id=thread)

            if "__interrupt__" in state:
                _print_pending(state["__interrupt__"][0].value)
                verdict = _prompt_for_verdict()
                state = resume_graph(graph, verdict, thread_id=thread)

            print(f"copilot> {final_answer(state)}\n")


def _prompt_for_verdict() -> dict:
    """Ask the operator for a verdict, interactively.

    Anything unrecognized falls through to `_parse_verdict` in graph.py, which
    fails closed and treats it as a rejection.
    """
    try:
        raw = input("approve / reject / edit> ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return {"decision": "reject", "note": "no verdict given"}

    if raw.startswith("a"):
        return {"decision": "approve"}
    if raw.startswith("e"):
        print("Enter the edited answer:")
        return {"decision": "edit", "text": input("edit> ")}
    return {"decision": "reject", "note": input("why? ").strip()}


def cmd_review(
    thread_id: str,
    *,
    checkpointer: str | None = None,
    decision: str | None = None,
    text: str = "",
    note: str = "",
    max_iterations: int = 6,
) -> None:
    """Show a paused thread and resume it with a verdict. (Part C)

    This command is the payoff of persistence: it runs in a *different process*
    from the one that produced the draft, and it finds the parked run purely
    from the thread_id and the checkpointer. Nothing was held in memory between
    the two invocations.

    Note that the graph is rebuilt here from scratch. A compiled graph is
    stateless - the conversation lives in the checkpointer, not in the object -
    so "the same graph" only has to mean "the same shape and the same saver".
    """
    with checkpointer_scope(checkpointer) as saver:
        if saver is None:
            print(
                "error: review needs a checkpointer; a run parked with "
                "--checkpointer none no longer exists.",
                file=sys.stderr,
            )
            return

        graph = build_graph(
            max_iterations=max_iterations,
            checkpointer=saver,
            require_approval=True,
        )

        snapshot = graph.get_state(thread_config(thread_id))
        payload = pending_interrupt(graph, thread_id=thread_id)

        if payload is None:
            # Two different failures, deliberately reported differently: an
            # unknown thread_id and a known thread with nothing parked both
            # return an empty snapshot's worth of "no interrupts", and telling
            # them apart is the difference between a typo and a no-op.
            if not snapshot.values:
                print(
                    f"error: thread {thread_id!r} has no saved state. Either the "
                    "id is wrong, or it was written by a different checkpointer "
                    f"(this one is {describe_checkpointer(saver)}).",
                    file=sys.stderr,
                )
            else:
                print(
                    f"Thread {thread_id} exists but nothing is awaiting approval "
                    f"(next: {snapshot.next or 'idle'}).",
                    file=sys.stderr,
                )
                _print_final_state(snapshot.values)
            return

        _print_pending(payload)

        if decision is None:
            verdict: dict = _prompt_for_verdict()
        elif decision == "edit":
            verdict = {"decision": "edit", "text": text, "note": note}
        elif decision == "reject":
            verdict = {"decision": "reject", "note": note}
        else:
            verdict = {"decision": "approve"}

        # CONCEPT: Command(resume=...) goes where a state dict normally goes.
        state = resume_graph(graph, verdict, thread_id=thread_id)
        print()
        print(final_answer(state))
        _print_final_state(state)


def cmd_threads(checkpointer: str | None = None) -> None:
    """List the conversations the checkpointer holds.

    Useful in its own right, and it makes two Phase 4 facts visible: thread_ids
    are just strings the checkpointer has seen, and the checkpoint *count* grows
    per super-step rather than per question - which is the real answer to "what
    does an unbounded conversation look like on disk".
    """
    with checkpointer_scope(checkpointer) as saver:
        if saver is None:
            print("--checkpointer none stores nothing; there are no threads.")
            return

        print(f"[checkpointer] {describe_checkpointer(saver)}", file=sys.stderr)
        thread_ids = list_thread_ids(saver)
        if not thread_ids:
            print("No threads yet. Run `graph-chat` or `graph-agent --thread <id>`.")
            return

        print(f"{'thread_id':<40} checkpoints")
        for thread_id in thread_ids:
            print(f"{thread_id:<40} {count_checkpoints(saver, thread_id)}")


def cmd_prebuilt_agent(question: str) -> None:
    state = run_prebuilt_agent(question)
    print(final_answer(state))
    _print_final_state(state)


def cmd_draw_graph(which: str) -> None:
    """Print the compiled graph's structure without running it.

    `.get_graph()` returns the structure LangGraph derived from the nodes and
    edges you declared - the same structure Studio renders. That it can be drawn
    before anything runs is the point of compiling separately from invoking.

    No model is called and no tokens are spent, though ANTHROPIC_API_KEY still
    has to be set: building the graph constructs the ChatAnthropic object (to
    bind the tools to it), and the factory checks for a key at that point.
    """
    graph = build_graph() if which == "graph" else build_prebuilt_agent()
    drawn = graph.get_graph()

    print("--- nodes ---")
    for node in drawn.nodes:
        print(f"  {node}")
    print("--- edges ---")
    for edge in drawn.edges:
        label = f"  [{edge.data}]" if edge.data else ""
        dashed = " (conditional)" if edge.conditional else ""
        print(f"  {edge.source} -> {edge.target}{label}{dashed}")

    # Mermaid needs no extra packages; the ASCII renderer needs grandalf, which
    # isn't worth a dependency when Studio and mermaid.live both draw this better.
    print("\n--- mermaid (paste into https://mermaid.live) ---")
    print(drawn.draw_mermaid())
    try:
        print("--- ascii ---")
        print(drawn.draw_ascii())
    except ImportError:
        print("(install grandalf for ASCII art: pip install grandalf)")


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


def _add_phase4_flags(parser: argparse.ArgumentParser) -> None:
    """Flags shared by the commands that can persist state.

    `--checkpointer` is Phase 4's counterpart to Phase 2's `--memory`: the same
    "policy is a flag, not a rewrite" idea, one layer down. `--memory` chose how
    history was *compressed*; this chooses where it is *kept*.
    """
    parser.add_argument(
        "--checkpointer",
        choices=list(KINDS),
        default=None,
        help=(
            "none: Phase 3 behaviour, state dies with the run. memory: persists "
            "for this process. sqlite: persists to data/ across processes. "
            "Defaults to RESEARCH_COPILOT_CHECKPOINTER."
        ),
    )
    parser.add_argument(
        "--approve",
        action="store_true",
        help=(
            "Pause at interrupt() for human approval before the answer is "
            "committed to the transcript. Requires a checkpointer."
        ),
    )
    parser.add_argument(
        "--memory",
        dest="memory_strategy",
        choices=["trim", "summarize", "none"],
        default=None,
        help="How prune_history keeps the persisted transcript in budget",
    )
    parser.add_argument(
        "--max-history-tokens",
        type=int,
        default=None,
        help="Token budget for the persisted transcript (0 disables pruning)",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="research-copilot", description="Research Copilot (Phases 1-4)"
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

    # --- Phase 3 ---
    graph_parser = subparsers.add_parser(
        "graph-agent",
        help="The same work as `ask`, rebuilt as a LangGraph StateGraph",
    )
    graph_parser.add_argument("question")
    graph_parser.add_argument(
        "--mode",
        choices=["knowledge-base", "live-search"],
        default="live-search",
        help=(
            "Which branch route_by_mode takes. knowledge-base: retrieve_docs -> "
            "call_model. live-search: call_model <-> call_tool until done."
        ),
    )
    graph_parser.add_argument("--max-iterations", type=int, default=6)
    _add_phase4_flags(graph_parser)
    graph_parser.add_argument(
        "--thread",
        default=None,
        help=(
            "Continue an existing conversation. Omit to start a new one (the "
            "generated id is printed, so the next run can pass it back)."
        ),
    )

    prebuilt_parser = subparsers.add_parser(
        "prebuilt-agent",
        help="The live-search path via langgraph.prebuilt.create_react_agent",
    )
    prebuilt_parser.add_argument("question")

    # --- Phase 4 ---
    chat_graph_parser = subparsers.add_parser(
        "graph-chat",
        help="Multi-turn conversation against one persisted thread_id",
    )
    chat_graph_parser.add_argument(
        "--thread",
        default=None,
        help="Resume an existing conversation instead of starting a new one",
    )
    chat_graph_parser.add_argument(
        "--mode", choices=["knowledge-base", "live-search"], default="live-search"
    )
    chat_graph_parser.add_argument("--max-iterations", type=int, default=6)
    _add_phase4_flags(chat_graph_parser)

    review_parser = subparsers.add_parser(
        "review",
        help="Show a thread paused at interrupt() and approve/reject/edit it",
    )
    review_parser.add_argument("--thread", required=True, help="The paused thread_id")
    review_parser.add_argument(
        "--checkpointer",
        choices=list(KINDS),
        default=None,
        help="Must match the checkpointer that wrote the thread",
    )
    # Mutually exclusive: a verdict is one decision, and omitting all three
    # drops into the interactive prompt.
    verdict_group = review_parser.add_mutually_exclusive_group()
    verdict_group.add_argument(
        "--approve", action="store_true", help="Accept the draft as written"
    )
    verdict_group.add_argument(
        "--reject", action="store_true", help="Discard the draft (give --note)"
    )
    verdict_group.add_argument(
        "--edit", metavar="TEXT", default=None, help="Accept this text instead"
    )
    review_parser.add_argument(
        "--note", default="", help="Why, recorded in State['human_feedback']"
    )

    threads_parser = subparsers.add_parser(
        "threads", help="List the conversations stored in the checkpointer"
    )
    threads_parser.add_argument(
        "--checkpointer", choices=list(KINDS), default=None
    )

    draw_parser = subparsers.add_parser(
        "draw-graph", help="Print a graph's structure without running it"
    )
    draw_parser.add_argument(
        "which", nargs="?", choices=["graph", "prebuilt"], default="graph"
    )

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
        elif args.command == "graph-agent":
            cmd_graph_agent(
                args.question,
                args.mode,
                args.max_iterations,
                thread_id=args.thread,
                checkpointer=args.checkpointer,
                approve=args.approve,
                memory_strategy=args.memory_strategy,
                max_history_tokens=args.max_history_tokens,
            )
        elif args.command == "graph-chat":
            cmd_graph_chat(
                args.mode,
                args.max_iterations,
                thread_id=args.thread,
                checkpointer=args.checkpointer,
                approve=args.approve,
                memory_strategy=args.memory_strategy,
                max_history_tokens=args.max_history_tokens,
            )
        elif args.command == "review":
            decision = None
            if args.approve:
                decision = "approve"
            elif args.reject:
                decision = "reject"
            elif args.edit is not None:
                decision = "edit"
            cmd_review(
                args.thread,
                checkpointer=args.checkpointer,
                decision=decision,
                text=args.edit or "",
                note=args.note,
            )
        elif args.command == "threads":
            cmd_threads(args.checkpointer)
        elif args.command == "prebuilt-agent":
            cmd_prebuilt_agent(args.question)
        elif args.command == "draw-graph":
            cmd_draw_graph(args.which)
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
