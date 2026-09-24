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

Phase 5 adds no commands, only flags on the graph commands:
  --critic         a model reviews each draft (the --approve gate, automated)
  --max-revisions  how many critique -> revise rounds a turn may spend
  --plan           decompose the question into sub-questions before research

That there is no new command is the point. Reflection is not a different way of
using the system, it is more edges inside the same graph, so it shows up as
flags on the commands that already run it.

Phase 6 (step 6.1) adds one command, and here a new command *is* right:
  multi-agent     the same question answered by a Researcher agent and a Writer
                  agent in a separate graph (multi_agent_graph.py), with the
                  hand-offs printed as they happen

It is a new command, not a flag on `graph-agent`, because it is a different
graph with a different State, not more edges inside the old one. Phase 5's
graph stays reachable, unchanged, through `graph-agent`.

6.2 adds flags to `multi-agent`, not a command. The Supervisor changes who
decides the route, not what the graph is:
  --routing supervisor|fixed          a model decides (default), or 6.1's
                                      hand-off as a fixed policy
  --max-researcher-runs / --max-writer-runs
                                      per-turn dispatch caps

6.3 adds --critic and --max-revisions to `multi-agent`. The flags share their
names with Phase 5's, but not all of their meaning. What each has meant, phase
by phase:

  flag              Phase 4          Phase 5 (graph-agent)      Phase 6.3 (multi-agent)
  ----------------  ---------------  -------------------------  ---------------------------------
  --critic          (did not exist)  a critique_draft node      the Critic *agent* joins the
                                     reviews every draft;       Supervisor's roster; it verifies
                                     rejection -> call_model    arXiv citations; a rejection goes
                                     re-drafts                  to the Supervisor, which picks
                                                                Researcher or Writer to fix it
  --max-revisions   (did not exist)  rejection rounds per turn  unchanged: rejection rounds per
                                                                turn, from either reviewer. Also
                                                                scales the Writer/Critic dispatch
                                                                caps so every round can be used
  --approve         human gate       unchanged; after the       same meaning: after the Supervisor
                    before commit    critic when both on        finishes, after the Critic approves
                                     (critic first)             when both on. Wired in 6.4 with
                                                                --thread/--checkpointer; resumed by
                                                                `multi-review`, which reads the
                                                                policy from the thread instead of
                                                                taking --critic/--max-revisions
                                                                again (Phase 5's `review` did)
  --plan            (did not exist)  plan_question before the   unchanged: plan_question before
                                     mode branch                the Supervisor's first decision

`graph-agent` keeps its Phase 5 meanings for all four. Nothing on that command
changed.

6.4 adds one command and the persistence flags:
  multi-review    Phase 4's `review` for a paused multi-agent thread
  multi-agent --thread / --checkpointer / --approve / --memory / --max-history-tokens
                  Phase 4's flags, same meanings as on graph-agent
The README's "Phase 6: Multi-agent (reference)" section carries the same flag
table; the two are meant to match.
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
from langgraph.types import Command

from research_copilot.multi_agent_graph import (
    build_multi_agent_graph,
    multi_agent_turn_input,
    run_policy,
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
    # Phase 5. `revisions` is printed beside `iterations` on purpose: seeing the
    # two numbers side by side in every dump is the fastest way to internalise
    # that they count different loops. `iterations` is the *current* round's
    # tool budget, reset by start_revision - so "revisions 2, iterations 1"
    # means the second revision has made one model call, not that three calls
    # have happened in total.
    if state.get("revisions"):
        print(f"  revisions:  {state['revisions']}", file=sys.stderr)
    if state.get("critique"):
        print(f"  critique:   {state['critique'][:70]}", file=sys.stderr)
    if state.get("sub_questions"):
        print(f"  plan:       {len(state['sub_questions'])} sub-questions", file=sys.stderr)
        for i, sub_question in enumerate(state["sub_questions"], start=1):
            print(f"    [{i}] {sub_question}", file=sys.stderr)
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


# --- Phase 5: reflection and planning -----------------------------------------


def _phase5_build_kwargs(
    *, critic: bool, max_revisions: int | None, plan: bool
) -> dict:
    """Resolve the Phase 5 flags into build_graph arguments, and warn.

    `--max-revisions` defaults to the setting (2) rather than to
    `build_graph`'s own default (0). The two defaults differ on purpose: a
    library caller who passes nothing should get Phase 4's behaviour unchanged,
    while on the command line `--critic --max-revisions 0` is a reviewer that
    can veto but never ask for a fix, which nobody means to ask for.
    """
    settings = get_settings()
    resolved = (
        settings.max_revisions if max_revisions is None else max_revisions
    )

    if critic and resolved <= 0:
        print(
            "[warning] --critic with --max-revisions 0: the critic can reject "
            "a draft but the run has no budget to revise it, so a rejection "
            "goes straight to a withheld answer.",
            file=sys.stderr,
        )

    return {
        "enable_critic": critic,
        "max_revisions": resolved,
        "enable_planning": plan,
    }


def _announce_review_setup(*, critic: bool, approve: bool, max_revisions: int) -> None:
    """Say out loud which reviewers are on, and in what order.

    Worth printing rather than leaving implicit, because the ordering has a
    consequence that is invisible from the outside: with both reviewers on, the
    human is asked only about drafts the critic passed, so a critic that never
    approves means `--approve` never pauses and the run just ends with a
    withheld draft. That looks like a bug in the approval gate and isn't.
    """
    if critic and approve:
        print(
            f"[review] critic, then human (max {max_revisions} revisions). "
            "A draft the critic keeps rejecting never reaches you - the run "
            "ends withheld once the revision cap is spent.",
            file=sys.stderr,
        )
    elif critic:
        print(f"[review] critic only (max {max_revisions} revisions)", file=sys.stderr)
    elif approve:
        print(f"[review] human only (max {max_revisions} revisions)", file=sys.stderr)


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
    critic: bool = False,
    max_revisions: int | None = None,
    plan: bool = False,
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

        phase5 = _phase5_build_kwargs(
            critic=critic, max_revisions=max_revisions, plan=plan
        )
        _announce_review_setup(
            critic=critic, approve=approve, max_revisions=phase5["max_revisions"]
        )

        graph = build_graph(
            max_iterations=max_iterations,
            checkpointer=saver,
            require_approval=approve,
            memory_strategy=memory_strategy,
            max_history_tokens=max_history_tokens,
            **phase5,
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
    critic: bool = False,
    max_revisions: int | None = None,
    plan: bool = False,
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

        phase5 = _phase5_build_kwargs(
            critic=critic, max_revisions=max_revisions, plan=plan
        )
        _announce_review_setup(
            critic=critic, approve=approve, max_revisions=phase5["max_revisions"]
        )

        graph = build_graph(
            max_iterations=max_iterations,
            checkpointer=saver,
            require_approval=approve,
            memory_strategy=memory_strategy,
            max_history_tokens=max_history_tokens,
            **phase5,
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

            # PHASE 5: `while`, not `if`. A rejection no longer ends the turn -
            # it starts a revision, which produces another draft, which parks at
            # `interrupt()` again. The loop is bounded by `max_revisions` inside
            # the graph, so this cannot spin: once the cap is spent
            # `should_revise` routes to `finalize_answer` and no further
            # interrupt arrives.
            while "__interrupt__" in state:
                _print_pending(state["__interrupt__"][0].value)
                revisions = state.get("revisions", 0)
                if revisions:
                    print(f"(revision {revisions})", file=sys.stderr)
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
    critic: bool = False,
    max_revisions: int | None = None,
) -> None:
    """Show a paused thread and resume it with a verdict. (Part C)

    This command is the payoff of persistence: it runs in a *different process*
    from the one that produced the draft, and it finds the parked run purely
    from the thread_id and the checkpointer. Nothing was held in memory between
    the two invocations.

    Note that the graph is rebuilt here from scratch. A compiled graph is
    stateless - the conversation lives in the checkpointer, not in the object -
    so "the same graph" only has to mean "the same shape and the same saver".

    PHASE 5 sharpens that "only", and it is the sort of thing you find by
    building the loop rather than by reading about it. Shape is not the whole
    story once a routing function closes over a number. `should_revise` compares
    `revisions` against the `max_revisions` captured *in this process*, not the
    one the run was started with - the counter is checkpointed, the cap is not.
    Resume a thread here with the default cap and a rejection that should have
    started a revision quietly becomes a withheld answer instead.

    Hence `--max-revisions` and `--critic` on this command too: a resume has to
    be told the same policy the pause was started under. The general rule, worth
    carrying into Phase 6 - anything a routing function reads from a closure is
    configuration the caller must repeat on every process that touches the
    thread, because only State survives.
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
            **_phase5_build_kwargs(
                critic=critic, max_revisions=max_revisions, plan=False
            ),
        )

        snapshot = graph.get_state(thread_config(thread_id))
        # PHASE 6.4: both graphs share one checkpoint database. Resuming a
        # multi-agent thread with this graph would replay its `review_draft`
        # into a graph with different nodes and a different State - refuse.
        if _is_multi_agent_thread(_raw_channels(saver, thread_id)):
            print(
                f"error: thread {thread_id!r} was written by the multi-agent graph. "
                f"Use: research-copilot multi-review --thread {thread_id}",
                file=sys.stderr,
            )
            return
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

        # PHASE 5: a rejection may now have started a revision rather than ended
        # the run, in which case the graph is parked at the *next* interrupt
        # with a fresh draft. Say so - "here is another draft" and "here is the
        # answer" look identical otherwise.
        if "__interrupt__" in state:
            _print_pending(state["__interrupt__"][0].value)
            print(
                f"\n[revised] revision {state.get('revisions', 0)}; review again "
                f"with: research-copilot review --thread {thread_id}",
                file=sys.stderr,
            )
            _print_final_state(state)
            return

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
    if which == "multi":
        # PHASE 6: xray=True expands subgraphs, so the Researcher's internal
        # loop is drawn inside it. Without it, the Researcher is one box, which
        # is how the parent graph actually sees it. Studio shows the same two
        # views: collapsed by default, expandable on click.
        graph = build_multi_agent_graph()
        drawn = graph.get_graph(xray=True)
    else:
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


# --- Phase 6: multi-agent -----------------------------------------------------


def _describe_update(node: str, update: dict) -> str:
    """One line summarising what a node just wrote. Used for the hand-off trace."""
    if not update:
        return "(no change)"
    parts = []
    for key, value in update.items():
        if isinstance(value, str):
            text = " ".join(value.split())
            parts.append(f"{key}={text[:60]!r}" + ("..." if len(text) > 60 else ""))
        elif isinstance(value, list):
            parts.append(f"{key}=[{len(value)}]")
        else:
            parts.append(f"{key}={value!r}")
    return ", ".join(parts)


def _describe_supervisor(update: dict) -> str:
    """The latest Supervisor decision as one line: route, override, rationale."""
    log = update.get("supervisor_log") or []
    if not log:
        return _describe_update("supervisor", update)
    e = log[-1]
    line = f"-> {e['routed_to']}"
    if e["override"]:
        proposed = e["proposed"] or "nothing usable"
        line += f"  (proposed {proposed}; OVERRIDDEN: {e['override']})"
    if e.get("brief"):
        line += f"  brief={e['brief']!r}"
    rationale = " ".join(e["rationale"].split())
    return line + f"\n      why: {rationale[:160]}" + ("..." if len(rationale) > 160 else "")


def _describe_critic(update: dict) -> str:
    """The Critic's verdict as one line, with the citation checks it ran."""
    verdict = (update.get("verdict") or "?").upper()
    critique = " ".join((update.get("critique") or "").split())
    checks = update.get("citation_checks") or []
    line = f"{verdict}"
    if checks:
        line += "  checks: " + ", ".join(f"{c['arxiv_id']}={c['status']}" for c in checks)
    if critique:
        line += f"\n      note: {critique[:160]}" + ("..." if len(critique) > 160 else "")
    return line


def _stream_multi_agent(graph, graph_input, config: dict | None) -> tuple[dict, dict | None]:
    """Run (or resume) the multi-agent graph, printing every hand-off.

    Returns (final state, pending interrupt payload or None). `graph_input` is
    either a turn's input dict or a `Command(resume=...)` - `.stream()`, like
    `.invoke()`, takes either in the same position.

    CONCEPT: streaming with subgraphs=True
    `.invoke()` returns only the final state, which for this graph hides the
    very thing Phase 6 is about. So this streams instead. With
    `subgraphs=True`, every event carries a *namespace*: `()` for a step of the
    parent graph, and `("researcher:<task-id>",)` for a step *inside* an agent's
    subgraph. The trace indents the inner steps.

    CONCEPT: which stream modes can see inside a subgraph's private state
    For a *nested* subgraph, the "updates" and "values" stream modes are
    narrowed to the subgraph's output schema: a Researcher step that wrote only
    `research_messages` shows up there as an empty update. The "tasks" mode
    reports each task's full write, private keys included. So:

        "updates"   parent-level steps: what each agent handed over
        "tasks"     steps inside a subgraph, private channel included
        "values"    the parent's state, the same dict `.invoke()` returns

    The privacy boundary is therefore a boundary on *state other agents read*,
    not on visibility: the private channels are kept out of the parent's state
    and out of other agents' inputs, but they are streamed here and they are
    checkpointed under their subgraph's namespace (6.1's finding).
    """
    state: dict = {}
    print("--- hand-offs ---", file=sys.stderr)
    for namespace, stream_mode, payload in graph.stream(
        graph_input,
        config,
        stream_mode=["updates", "tasks", "values"],
        subgraphs=True,
    ):
        if stream_mode == "values":
            if not namespace:
                state = payload
        elif stream_mode == "tasks" and namespace and "result" in payload:
            # A finished step inside an agent's subgraph. (A "tasks" event
            # without "result" is the step *starting*.)
            agent = namespace[0].split(":", 1)[0]
            result = payload["result"] if isinstance(payload["result"], dict) else {}
            print(
                f"    [{agent}/{payload['name']}] {_describe_update(payload['name'], result)}",
                file=sys.stderr,
            )
        elif stream_mode == "updates" and not namespace:
            for node, update in payload.items():
                if node == "__interrupt__":
                    print("  [paused] review_draft is waiting for a human verdict", file=sys.stderr)
                elif node == "supervisor":
                    print(f"  [supervisor] {_describe_supervisor(update or {})}", file=sys.stderr)
                elif node == "critic":
                    print(f"  [critic] {_describe_critic(update or {})}", file=sys.stderr)
                else:
                    print(f"  [{node}] {_describe_update(node, update or {})}", file=sys.stderr)
    print("--- end hand-offs ---", file=sys.stderr)

    pending = None
    if config is not None:
        snapshot = graph.get_state(config)
        if snapshot.interrupts:
            pending = snapshot.interrupts[0].value
    return state, pending


def _print_multi_agent_pending(payload: dict, thread_id: str) -> None:
    """A parked multi-agent run: the draft, and the critique it already passed."""
    print("--- awaiting approval ---")
    print(f"question: {payload.get('question', '')}")
    if payload.get("critique"):
        print(f"critic:   {payload['critique']}")
    print("\ndraft:")
    print(payload.get("draft", "") or "(empty draft)")
    print("--- end draft ---")
    print(
        f"\n[paused] resume with: research-copilot multi-review --thread {thread_id}",
        file=sys.stderr,
    )


def cmd_multi_agent(
    question: str,
    mode: str,
    *,
    max_research_iterations: int,
    plan: bool = False,
    routing: str = "supervisor",
    max_researcher_runs: int | None = None,
    max_writer_runs: int | None = None,
    critic: bool = False,
    max_revisions: int | None = None,
    # --- 6.4 ---
    thread_id: str | None = None,
    checkpointer: str | None = None,
    approve: bool = False,
    memory_strategy: str | None = None,
    max_history_tokens: int | None = None,
) -> None:
    """One turn of the multi-agent graph, optionally on a persistent thread.

    6.4 brings Phase 4's persistence to this command, with Phase 4's exact
    pattern: the thread_id goes in `config`, never in state; `--checkpointer`
    picks the saver; `--approve` pauses at `review_draft` and needs a saver to
    park in.

    What is new is that the turn's *policy* goes into state (`run_policy`), so
    that `multi-review` can rebuild the same graph without being told the same
    flags again. See `run_policy` in multi_agent_state.py.
    """
    if mode == "knowledge-base" and _warn_if_empty_store():
        return

    dispatch_caps = {}
    if max_researcher_runs is not None:
        dispatch_caps["researcher"] = max_researcher_runs
    if max_writer_runs is not None:
        dispatch_caps["writer"] = max_writer_runs
    policy = run_policy(
        routing=routing,
        enable_critic=critic,
        require_approval=approve,
        max_revisions=get_settings().max_revisions if max_revisions is None else max_revisions,
        dispatch_caps=dispatch_caps or None,
        max_research_iterations=max_research_iterations,
        enable_planning=plan,
        memory_strategy=memory_strategy,
        max_history_tokens=max_history_tokens,
    )

    with checkpointer_scope(checkpointer) as saver:
        if approve and saver is None:
            print(
                "error: --approve needs a checkpointer: interrupt() parks the run "
                "in one. Use --checkpointer memory or sqlite.",
                file=sys.stderr,
            )
            return
        thread, is_new = _resolve_thread(thread_id)
        _announce_thread(thread, is_new, saver)
        config = thread_config(thread) if saver is not None else None

        graph = build_multi_agent_graph(**policy, checkpointer=saver)

        if config is not None and not is_new:
            existing = graph.get_state(config)
            raw = _raw_channels(saver, thread)
            if raw and not _is_multi_agent_thread(raw):
                print(
                    f"error: thread {thread!r} belongs to the single-agent graph "
                    "(graph-agent / graph-chat). Start a new multi-agent thread.",
                    file=sys.stderr,
                )
                return
            if existing.interrupts:
                # Found while writing the 6.4 tests. A new turn's input on a
                # thread parked at review_draft does not fail: LangGraph starts
                # the new turn and the parked draft is silently abandoned. The
                # transcript then holds two human turns in a row (the
                # unanswered one and the new one), which the Anthropic API
                # rejects on the *next* call. Refuse instead: the pending
                # decision has to be made first.
                print(
                    f"error: thread {thread} has a draft awaiting approval. Resolve "
                    f"it first: research-copilot multi-review --thread {thread}",
                    file=sys.stderr,
                )
                return

        _announce_policy(policy)
        state, pending = _stream_multi_agent(
            graph, multi_agent_turn_input(question, mode, policy=policy), config
        )

        if pending is not None:
            _print_multi_agent_pending(pending, thread)
            _print_multi_agent_state(state)
            return
        print(final_answer(state))
        _print_multi_agent_state(state)


def _announce_policy(policy: dict) -> None:
    roster = "researcher, writer, critic" if policy.get("enable_critic") else "researcher, writer"
    gate = "; human approval on" if policy.get("require_approval") else ""
    print(
        f"[routing] {policy.get('routing', 'supervisor')}  (roster: {roster}; "
        f"max revisions {policy.get('max_revisions')}{gate})",
        file=sys.stderr,
    )


def _raw_channels(saver, thread_id: str) -> dict:
    """The thread's latest parent checkpoint, as stored - not as any graph sees it.

    CONCEPT (6.4): `graph.get_state()` is a *view*, filtered by that graph's schema
    Found by a failing test. `get_state` returns only the channels the calling
    graph declares. Ask the single-agent graph about a multi-agent thread and
    `run_policy`, `supervisor_log`, `budgets` are simply not in the answer - so
    a "which graph wrote this?" check built on `get_state` always concludes
    "mine", and Phase 4's `review` went on to resume a multi-agent thread with
    the wrong graph. The question of *whose* thread it is has to be asked of
    the checkpointer directly.
    """
    found = saver.get_tuple(thread_config(thread_id))
    return dict(found.checkpoint.get("channel_values", {})) if found else {}


def _is_multi_agent_thread(values: dict) -> bool:
    """Which graph wrote this thread? Both graphs share one checkpoint database,
    so a thread_id alone does not say. `run_policy` is written by every
    multi-agent turn since 6.4 and by nothing in the Phase 3-5 graph;
    `supervisor_log` covers multi-agent threads from 6.2-6.3. Pass it RAW
    channel values (`_raw_channels`), never `get_state().values`."""
    return "run_policy" in values or "supervisor_log" in values


def cmd_multi_review(
    thread_id: str,
    *,
    checkpointer: str | None = None,
    decision: str | None = None,
    text: str = "",
    note: str = "",
) -> None:
    """Phase 4's `review`, for a paused multi-agent thread. (6.4)

    The same three steps: find the parked run from the thread_id alone, show
    the question put to the human, resume with `Command(resume=verdict)`.

    The difference is what this command does *not* take. Phase 5's `review`
    needed `--critic` and `--max-revisions` repeated, because those lived in
    closures and only State survives. Here the graph is rebuilt from the
    thread's own `run_policy`, so the resume runs under exactly the policy the
    pause was started under, whatever flags this process was given. A thread
    written before 6.4 has no stored policy and is refused rather than guessed
    at.
    """
    with checkpointer_scope(checkpointer) as saver:
        if saver is None:
            print(
                "error: multi-review needs a checkpointer; a run parked with "
                "--checkpointer none no longer exists.",
                file=sys.stderr,
            )
            return

        # A bare graph is enough to *read* the snapshot: get_state needs the
        # saver and the thread_id, not the policy.
        probe = build_multi_agent_graph(checkpointer=saver)
        config = thread_config(thread_id)
        snapshot = probe.get_state(config)

        if not snapshot.values:
            print(
                f"error: thread {thread_id!r} has no saved state. Either the id is "
                "wrong, or it was written by a different checkpointer "
                f"(this one is {describe_checkpointer(saver)}).",
                file=sys.stderr,
            )
            return
        if not _is_multi_agent_thread(_raw_channels(saver, thread_id)):
            print(
                f"error: thread {thread_id!r} was written by the single-agent graph. "
                f"Use: research-copilot review --thread {thread_id}",
                file=sys.stderr,
            )
            return
        policy = snapshot.values.get("run_policy")
        if not policy:
            print(
                f"error: thread {thread_id!r} predates 6.4 and has no stored run "
                "policy, so it cannot be resumed under a known configuration.",
                file=sys.stderr,
            )
            return
        if not snapshot.interrupts:
            # The case to be careful with: a thread whose subgraphs have
            # checkpoints of their own (the private channels, under
            # researcher:/critic: namespaces) but nothing parked at the parent.
            # Those subgraph checkpoints are history, not a pending decision.
            print(
                f"Thread {thread_id} exists but nothing is awaiting approval "
                f"(next: {snapshot.next or 'idle'}).",
                file=sys.stderr,
            )
            _print_multi_agent_state(snapshot.values)
            return

        payload = snapshot.interrupts[0].value
        _print_multi_agent_pending(payload, thread_id)

        if decision is None:
            verdict: dict = _prompt_for_verdict()
        elif decision == "edit":
            verdict = {"decision": "edit", "text": text, "note": note}
        elif decision == "reject":
            verdict = {"decision": "reject", "note": note}
        else:
            verdict = {"decision": "approve"}

        graph = build_multi_agent_graph(**policy, checkpointer=saver)
        _announce_policy(policy)
        state, pending = _stream_multi_agent(graph, Command(resume=verdict), config)

        if pending is not None:
            # A rejection went round a revision and reached the gate again.
            print(f"\n[revised] revision {state.get('revisions', 0)}", file=sys.stderr)
            _print_multi_agent_pending(pending, thread_id)
            _print_multi_agent_state(state)
            return
        print(final_answer(state))
        _print_multi_agent_state(state)


def _print_multi_agent_state(state: dict) -> None:
    """The final state, grouped by owner - the same grouping as MultiAgentState."""
    print("\n--- final state (by owner) ---", file=sys.stderr)
    print(f"  question:            {state.get('question')}", file=sys.stderr)
    print(f"  mode:                {state.get('mode')}", file=sys.stderr)
    if state.get("sub_questions"):
        print(f"  plan_question  -> sub_questions: {len(state['sub_questions'])}", file=sys.stderr)
        for i, sub_question in enumerate(state["sub_questions"], start=1):
            print(f"    [{i}] {sub_question}", file=sys.stderr)
    notes = " ".join((state.get("research_notes") or "").split())
    print(
        f"  researcher     -> research_notes: {notes[:70]!r}"
        + ("..." if len(notes) > 70 else "")
        + ("  (EMPTY - the Researcher found nothing)" if not notes else ""),
        file=sys.stderr,
    )
    print(f"                    research_iterations: {state.get('research_iterations', 0)}", file=sys.stderr)
    if state.get("research_outcome"):
        print(f"                    research_outcome: {state['research_outcome']}", file=sys.stderr)
    documents = state.get("documents") or []
    if documents:
        print(f"                    documents: {len(documents)} retrieved", file=sys.stderr)
        for i, document in enumerate(documents, start=1):
            print(f"      [{i}] {describe_source(document)}", file=sys.stderr)
    draft = " ".join((state.get("draft") or "").split())
    print(f"  writer         -> draft: {draft[:70]!r}" + ("..." if len(draft) > 70 else ""), file=sys.stderr)
    if state.get("verdict"):
        print(
            f"  critic         -> verdict: {state['verdict']}"
            f"  ({len(state.get('citation_checks') or [])} citations checked)",
            file=sys.stderr,
        )
    if state.get("revisions"):
        print(f"  start_revision -> revisions: {state['revisions']}", file=sys.stderr)
    budgets = state.get("budgets") or {}
    if budgets:
        print(
            "  (per agent)    -> round budgets: "
            + ", ".join(
                f"{agent} {entry.get('used', 0)}/{entry.get('cap', '?')}"
                for agent, entry in budgets.items()
            ),
            file=sys.stderr,
        )
    log = state.get("supervisor_log") or []
    if log:
        dispatches = state.get("dispatches") or {}
        print(
            "  supervisor     -> dispatches: "
            + (", ".join(f"{k} {v}" for k, v in dispatches.items()) or "none"),
            file=sys.stderr,
        )
        overridden = sum(1 for e in log if e["override"])
        print(f"                    decisions: {len(log)} ({overridden} overridden)", file=sys.stderr)
        for e in log:
            flag = f"  [override: {e['override']}]" if e["override"] else ""
            print(f"      {e['step']}. -> {e['routed_to']}{flag}", file=sys.stderr)
    messages = state.get("messages", [])
    print(f"  finalize_answer-> messages: {len(messages)}", file=sys.stderr)
    for message in messages:
        preview = " ".join(message.text.split())
        if len(preview) > 70:
            preview = preview[:70] + "..."
        print(f"    {message.type:>6}: {preview}", file=sys.stderr)
    # The Phase 6 invariant, printed rather than assumed: the Researcher's
    # private tool traffic must not be in the shared transcript.
    leaked = [m for m in messages if m.type == "tool" or getattr(m, "tool_calls", None)]
    if leaked:
        print(
            f"  [warning] {len(leaked)} tool message(s) in the shared transcript - "
            "the Researcher's private channel leaked",
            file=sys.stderr,
        )
    print("--- end state ---", file=sys.stderr)


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


def _add_phase5_flags(parser: argparse.ArgumentParser) -> None:
    """Phase 5's reviewers and planner.

    `--critic` is the deliberate counterpart to Phase 4's `--approve`: the same
    approve/reject gate with a model in the reviewer's seat. Having them as two
    independent switches is what makes the four combinations runnable, which is
    the point of the phase -

        (neither)            Phase 4 with approval off: draft, commit, done.
        --approve            a human gate. Pauses at interrupt(); needs a
                             checkpointer.
        --critic             a machine gate. No pause, no checkpointer needed.
        --critic --approve   both, critic first. The human is asked only about
                             drafts the critic passed.

    - and the four combinations run over *one graph shape*, because the review
    nodes are registered whatever the flags say. The flags choose paths, not
    structures.
    """
    parser.add_argument(
        "--critic",
        action="store_true",
        help=(
            "Have a model review each draft before it is finalized. Combine "
            "with --approve to put a human after the critic."
        ),
    )
    parser.add_argument(
        "--max-revisions",
        type=int,
        default=None,
        help=(
            "How many critique -> revise rounds one turn may spend. Separate "
            "from --max-iterations, which caps the tool loop *inside* each "
            "round. 0 means a rejection ends the run (Phase 4 behaviour). "
            "Defaults to RESEARCH_COPILOT_MAX_REVISIONS."
        ),
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help=(
            "Decompose the question into sub-questions before research. Most "
            "questions come back undecomposed; that is the planner working."
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="research-copilot", description="Research Copilot (Phases 1-6)"
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
    _add_phase5_flags(graph_parser)
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
    _add_phase5_flags(chat_graph_parser)

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
    # Phase 5: the resume has to be told the same policy the pause ran under.
    # `revisions` is checkpointed; `max_revisions` is a closure in this process.
    # See the note in cmd_review.
    review_parser.add_argument(
        "--max-revisions",
        type=int,
        default=None,
        help=(
            "Must match the run that produced the draft: a rejection starts a "
            "revision only if this process allows one."
        ),
    )
    review_parser.add_argument(
        "--critic",
        action="store_true",
        help="Match a run started with --critic, so revised drafts are critiqued too",
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
        "which", nargs="?", choices=["graph", "prebuilt", "multi"], default="graph"
    )

    # --- Phase 6 ---
    multi_review_parser = subparsers.add_parser(
        "multi-review",
        help="Show a paused multi-agent thread and approve/reject/edit it (6.4)",
    )
    multi_review_parser.add_argument("--thread", required=True, help="The paused thread_id")
    multi_review_parser.add_argument(
        "--checkpointer", choices=list(KINDS), default=None,
        help="Must match the checkpointer that wrote the thread",
    )
    multi_verdict = multi_review_parser.add_mutually_exclusive_group()
    multi_verdict.add_argument("--approve", action="store_true", help="Accept the draft as written")
    multi_verdict.add_argument("--reject", action="store_true", help="Send it back (give --note)")
    multi_verdict.add_argument("--edit", metavar="TEXT", default=None, help="Accept this text instead")
    multi_review_parser.add_argument("--note", default="", help="Why - recorded as human_feedback")
    # Deliberately no --critic / --max-revisions here, unlike Phase 5's review:
    # the policy is read from the thread (see cmd_multi_review).

    multi_parser = subparsers.add_parser(
        "multi-agent",
        help="Researcher agent -> Writer agent, with hand-offs printed (Phase 6.1)",
    )
    multi_parser.add_argument("question")
    multi_parser.add_argument(
        "--mode",
        choices=["knowledge-base", "live-search"],
        default="live-search",
        help=(
            "How the Researcher gathers evidence. knowledge-base: retrieval "
            "from ingested docs. live-search: its own arXiv tool loop."
        ),
    )
    multi_parser.add_argument(
        "--max-research-iterations",
        type=int,
        default=6,
        help="Cap on the Researcher's private tool loop (its model calls)",
    )
    multi_parser.add_argument(
        "--plan",
        action="store_true",
        help="Decompose the question first (Phase 5's planner, unchanged)",
    )
    multi_parser.add_argument(
        "--routing",
        choices=["supervisor", "fixed"],
        default="supervisor",
        help=(
            "supervisor: a model decides who acts next (6.2). fixed: 6.1's "
            "research -> write -> finish, as a policy on the same graph."
        ),
    )
    multi_parser.add_argument(
        "--max-researcher-runs",
        type=int,
        default=None,
        help="How many times the Supervisor may dispatch the Researcher per turn (default 2)",
    )
    multi_parser.add_argument(
        "--max-writer-runs",
        type=int,
        default=None,
        help=(
            "How many times the Supervisor may dispatch the Writer per turn "
            "(default 2 + max revisions)"
        ),
    )
    # 6.4: Phase 4's persistence flags (--checkpointer, --approve, --memory,
    # --max-history-tokens), shared with graph-agent, plus --thread.
    _add_phase4_flags(multi_parser)
    multi_parser.add_argument(
        "--thread",
        default=None,
        help="Continue an existing multi-agent conversation (the id is printed on each run)",
    )
    multi_parser.add_argument(
        "--critic",
        action="store_true",
        help=(
            "Put the Critic agent on the Supervisor's roster: it reviews drafts "
            "and verifies arXiv citations; rejections go back to the Supervisor"
        ),
    )
    multi_parser.add_argument(
        "--max-revisions",
        type=int,
        default=None,
        help="Rejection rounds per turn. Defaults to RESEARCH_COPILOT_MAX_REVISIONS (2).",
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
                critic=args.critic,
                max_revisions=args.max_revisions,
                plan=args.plan,
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
                critic=args.critic,
                max_revisions=args.max_revisions,
                plan=args.plan,
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
                critic=args.critic,
                max_revisions=args.max_revisions,
            )
        elif args.command == "threads":
            cmd_threads(args.checkpointer)
        elif args.command == "prebuilt-agent":
            cmd_prebuilt_agent(args.question)
        elif args.command == "draw-graph":
            cmd_draw_graph(args.which)
        elif args.command == "multi-agent":
            cmd_multi_agent(
                args.question,
                args.mode,
                max_research_iterations=args.max_research_iterations,
                plan=args.plan,
                routing=args.routing,
                max_researcher_runs=args.max_researcher_runs,
                max_writer_runs=args.max_writer_runs,
                critic=args.critic,
                max_revisions=args.max_revisions,
                thread_id=args.thread,
                checkpointer=args.checkpointer,
                approve=args.approve,
                memory_strategy=args.memory_strategy,
                max_history_tokens=args.max_history_tokens,
            )
        elif args.command == "multi-review":
            decision = None
            if args.approve:
                decision = "approve"
            elif args.reject:
                decision = "reject"
            elif args.edit is not None:
                decision = "edit"
            cmd_multi_review(
                args.thread,
                checkpointer=args.checkpointer,
                decision=decision,
                text=args.edit or "",
                note=args.note,
            )
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
