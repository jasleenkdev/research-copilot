# Research Copilot

An assistant that researches a question and produces a structured answer. It
starts as a single LangChain chain and grows, one phase at a time, into a
multi-agent, stateful LangGraph system with a human in the loop.

This is a **learning project**. Each phase introduces a small set of LangChain /
LangGraph concepts, and the code explains the *why* in comments where each
concept first appears. The goal is to understand each phase before starting the
next, not to ship the final system as fast as possible.

## Phase plan

- [x] **Phase 1: LangChain basics.** `ChatPromptTemplate`, message types, output
  parsers (`StrOutputParser`, then `PydanticOutputParser`), LCEL chaining with
  `|`, and one `@tool` bound to the model with a hand-written tool-call loop.
- [x] **Phase 2: Memory & retrieval.** Conversation memory with
  trimming/summarization. Basic RAG (load → chunk → embed → Chroma/FAISS →
  retriever chain) as a "knowledge base" mode alongside live search.
- [x] **Phase 3: First LangGraph.** Rebuild Phase 1's tool loop as a `StateGraph`
  (TypedDict state, `call_model` / `call_tool` nodes, conditional edges).
  Compare with `langgraph.prebuilt.create_react_agent`, then return to the
  hand-rolled version.
- [x] **Phase 4: State design & persistence.** Richer state (`draft`, `status`,
  `human_feedback`, `summary`), checkpointers (`MemorySaver`, then
  `SqliteSaver`), a `prune_history` node that keeps the *persisted* transcript
  in budget with `RemoveMessage`, and `interrupt()` for human approval before an
  answer is finalized.
- [x] **Phase 5: Multi-step reasoning.** A reflection loop (draft → critique →
  revise, looping back until a quality threshold or max iterations) and a
  planning node that splits the question into sub-questions before research.
- [x] **Phase 6: Multi-agent.** Researcher, Writer, and Critic nodes coordinated
  by a Supervisor that routes on structured LLM output. Each agent has its own
  tools and system prompt. Built in four reviewed steps:
  - [x] 6.1 Researcher (subgraph, private tool loop) → Writer (node), fixed
    hand-off, one owner per state field
  - [x] 6.2 Supervisor routing via structured output, with code guards,
    a decision log, and the Researcher's brief/outcome/merge-on-rerun contract
  - [x] 6.3 Critic agent (subgraph, `verify_citation`), rejections classified by
    the Supervisor, per-agent round budgets, graph-level human gate
  - [x] 6.4 threads, `multi-review`, stored run policy, pruning, `begin_turn`,
    Studio demo graph, this reference (see "Phase 6: Multi-agent (reference)")
- [ ] **Phase 7: Production.** Streaming via `astream_events`, per-node error
  handling, retries and fallback models, LangSmith dataset evaluation, and a
  FastAPI wrapper around the compiled graph (or LangGraph Studio).

Each phase lives on its own branch (`phase-1`, `phase-2`, …) and is merged into
`main` once it's understood.

## Setup

Requires Python 3.11+ (the venv here uses 3.13).

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env   # then fill in ANTHROPIC_API_KEY and LANGSMITH_API_KEY
```

LangSmith tracing is on by default (`LANGSMITH_TRACING=true`). Every run shows
up as a nested trace in the `research-copilot` project at
https://smith.langchain.com. Open the traces while you work: they show the
rendered prompt, the raw model output, each tool call, and token usage.

## Usage (Phase 1)

The modes follow the order the concepts are introduced:

```bash
# 1. Raw model call with hand-built SystemMessage/HumanMessage, plus AIMessage anatomy
research-copilot messages "What is retrieval-augmented generation?"

# 2. prompt | model | StrOutputParser, streamed token by token
research-copilot answer "What is retrieval-augmented generation?"

# 3. prompt | model | PydanticOutputParser -> validated ResearchAnswer JSON
research-copilot structured "What is retrieval-augmented generation?"

# 4. Model + arXiv tool + hand-written tool-call loop
research-copilot agent "What are recent approaches to evaluating RAG systems?"

# 5. Same, then pipe the findings through the structured chain
research-copilot agent --structured "What are recent approaches to evaluating RAG systems?"
```

`python -m research_copilot ...` works too.

## Usage (Phase 2)

### Conversation memory

```bash
# Interactive chat. /memory prints the current history, /exit quits.
research-copilot chat
research-copilot chat --memory summarize
research-copilot chat --memory trim --max-history-tokens 300
```

After every turn the chat prints a `[memory]` line to stderr showing what
pruning did. Set a small `--max-history-tokens` to watch it work within a few
turns.

Two strategies, swappable with `--memory` or `RESEARCH_COPILOT_MEMORY_STRATEGY`:

| | keeps | costs | loses |
| --- | --- | --- | --- |
| `trim` | the newest turns that fit the token budget | nothing | dropped turns, completely |
| `summarize` | recent turns verbatim + one summary message | one model call per compression | detail, and whatever the summary gets wrong |

The budget is in **tokens**, not messages, because tokens are what the API bills
and limits. One message can be 5 tokens or 5,000, so a message-count cap tells
you nothing about cost or context overflow. See the notes in `memory.py`.

### RAG over your own documents

```bash
# Load, chunk, embed, and store .txt/.md/.pdf (a file or a whole directory)
research-copilot ingest ./docs
research-copilot ingest ./docs --chunk-size 400 --chunk-overlap 80

# Answer from the ingested documents only
research-copilot ask-docs "How is chunk size chosen?"
research-copilot ask-docs "How is chunk size chosen?" -k 8

# Same question, either mode
research-copilot ask "How is chunk size chosen?" --mode knowledge-base
research-copilot ask "recent methods for evaluating RAG" --mode live-search
```

`--mode` is the seam for Phase 6: right now you choose knowledge base vs. live
arXiv search by hand, and later a Supervisor agent will make that choice itself.

**Embedding model:** `sentence-transformers/all-MiniLM-L6-v2`, running locally.
Anthropic serves no embeddings API, so the embedder is never the model that
writes the answer. This one is free, needs no key, and works offline after a
~90 MB first download. It is small and weaker than paid embeddings on technical
text, and it truncates input at 256 word pieces (~1,000 characters) — which is
why ingestion chunks at 800 characters. A chunk larger than the embedding window
is only partly searchable. Swap it with `RESEARCH_COPILOT_EMBEDDING_MODEL`, and
re-ingest afterwards: vectors from different models aren't comparable.

Ingesting the same file twice replaces its chunks instead of duplicating them.

## Usage (Phase 3)

Phase 1's manual loop and Phase 2's RAG chain, rebuilt as one `StateGraph`.
`agent_loop.py` and `retrieval.py` are untouched — the graph is additive, so the
two implementations can be run side by side on the same question.

```bash
# The hand-rolled graph. --mode picks which branch route_by_mode takes.
research-copilot graph-agent "recent methods for evaluating RAG" --mode live-search
research-copilot graph-agent "How is chunk size chosen?" --mode knowledge-base

# The same question through Phase 1's manual loop, for comparison
research-copilot agent "recent methods for evaluating RAG"

# The prebuilt ReAct agent: same tool, ~4 lines instead of ~100
research-copilot prebuilt-agent "recent methods for evaluating RAG"

# Print either graph's structure without running it (no tokens spent)
research-copilot draw-graph
research-copilot draw-graph prebuilt
```

`graph-agent` prints the answer on stdout and the **whole final state** on
stderr — question, mode, iteration count, retrieved documents, and every message
in the transcript. Reading that dict is the habit Phase 3 is trying to build:
the shape of `messages` tells you which path the run actually took.

### The graph

```
START ──route_by_mode──┬─ "knowledge-base" ─→ retrieve_docs ─→ call_model
                       └─ "live-search" ────────────────────→ call_model
                                                                  │
                                   ┌──────────────────────────────┘
                                   │
                        should_continue
                             ├─ "call_tool" ─→ call_tool ─→ call_model  (loop)
                             └─ END
```

Two conditional edges, and they are the first real branching in the project:

| | decides | from | generalizes into |
| --- | --- | --- | --- |
| `route_by_mode` | which research strategy | the `mode` field | Phase 6's Supervisor |
| `should_continue` | whether the tool loop continues | the last message's `tool_calls` | Phase 5's reflection loop |
| `should_revise` (Phase 5) | whether a rejected draft is revised | `status` + the `revisions` cap | Phase 6's Supervisor routing a critique to an agent |

`route_by_mode` sits at the **entry**, not after `call_model`, because retrieval
has to happen before the model speaks — the knowledge-base path's whole promise
is that the model never answers ungrounded. `call_tool` is the mirror image: the
model asks for a search, so that node necessarily runs after it. Same fork,
opposite sides of the model call.

`should_continue` checks `mode` too, even though knowledge-base mode binds no
tools and so "can't" produce tool calls. The edge from `call_model` to
`call_tool` exists for every run that reaches `call_model`, whichever branch got
it there; the routing function is the only thing keeping the two paths apart.

### What moved where

| Phase 1 / 2 | Phase 3 |
| --- | --- |
| a local `messages` list in `agent_loop.py` | `State["messages"]` + the `add_messages` reducer |
| `ConversationMemory.messages` in `memory.py` | the same `State["messages"]` — one home, not three |
| `for iteration in range(max_iterations)` | the `call_tool` → `call_model` edge, plus `State["iterations"]` |
| `if not ai_message.tool_calls: return` | the `should_continue` conditional edge |
| `if mode == ...` in `cli.py` | the `route_by_mode` conditional edge |
| `RunnableParallel(question=…, docs=retriever)` | the `retrieve_docs` node |

One behavioural difference worth knowing: at the iteration cap, `run_tool_loop`
executes tools and *then* notices it is out of budget (N iterations, N tool
calls, transcript ending on a `ToolMessage`). The graph checks the budget in
`should_continue`, which sits between `call_model` and `call_tool`, so the last
tool request is never run (N model calls, N−1 tool calls, transcript ending on
an `AIMessage`). Ending on an `AIMessage` is the better place to stop: there is
something to show the user.

### Hand-rolled vs. `create_react_agent`

`prebuilt.py` builds the live-search path in one call. What it does for you that
`graph.py` spells out:

| | `create_react_agent` | `graph.py` |
| --- | --- | --- |
| State schema + reducer | `AgentState` (`messages`, `remaining_steps`) | `state.py`, written by hand |
| `bind_tools` | automatic | explicit in `build_graph` |
| Running tools → `ToolMessage` | `ToolNode` | the `call_tool` node |
| Tool-call routing | `tools_condition` | `should_continue` |
| Message accumulation | falls out of its state schema | the `add_messages` annotation |
| Loop guard | `remaining_steps` | `State["iterations"]` + `max_iterations` |
| Nodes | 2 (`agent`, `tools`) | 3 + an entry branch |
| Extra state (`mode`, `documents`, …) | needs a custom `state_schema` | already there |
| A node that is neither model nor tool | no | `retrieve_docs` |

Run `draw-graph` and `draw-graph prebuilt` to see the two structures next to
each other.

**The trade-off.** Use `create_react_agent` when the shape genuinely is "model
plus tools, loop until done" — less code to write, read, and get wrong. Write
the graph out when you need state beyond a transcript, a node that isn't a model
or a tool, or a branch it doesn't have (all three arrive in Phases 4–6). The
reason to hand-roll it once, here, is that the prebuilt stays opaque until you
have built the thing it hides. Both return the same `CompiledStateGraph`, so
nothing is lost by switching later.

> `langgraph.prebuilt.create_react_agent` is **deprecated** in LangGraph 1.0 and
> prints a warning; it has moved to `langchain.agents.create_agent` (`prompt` is
> renamed `system_prompt`, and a `middleware` hook is added). `prebuilt.py` keeps
> the old name because the phase plan names it — the lesson is the same either
> way.

### LangGraph Studio

Studio is a visual debugger for a compiled graph: it draws the nodes and edges,
runs a question through them, and shows the state after every node. It runs
against a local dev server, which reads `langgraph.json` at the repo root.

```bash
# One-time: the dev server (a separate package from langgraph itself)
pip install -e ".[studio]"

# Launch. Opens Studio in the browser, pointed at http://127.0.0.1:2024
langgraph dev

# Useful flags
langgraph dev --no-browser       # just the server; connect to it yourself
langgraph dev --port 2025        # if 2024 is taken
langgraph dev --allow-blocking   # see the note below
```

If the browser doesn't open, go to:
<https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024>
(the UI is hosted, but your graph and state never leave your machine — the
server is local).

`langgraph.json` registers both graphs, so Studio's assistant dropdown lists
them side by side:

```json
{
  "dependencies": ["."],
  "graphs": {
    "research_copilot": "./src/research_copilot/graph.py:make_graph",
    "prebuilt": "./src/research_copilot/prebuilt.py:make_graph"
  },
  "env": ".env"
}
```

Each path points at a **factory function** rather than a module-level graph, so
nothing is built until the server starts.

To run one, pick a graph and submit an input matching `State`:

```json
{
  "question": "recent methods for evaluating RAG",
  "mode": "live-search",
  "messages": [{"role": "user", "content": "recent methods for evaluating RAG"}]
}
```

What to look at once it runs:

- the **entry branch**: only one of the two arrows out of `START` lights up, and
  which one is decided before any model call
- the **state panel after each node** — `documents` appears only after
  `retrieve_docs`, `messages` grows by one after `call_model`
- the **cycle**: in live-search mode `call_model` and `call_tool` are visited
  repeatedly, and the same node appears several times in the run timeline
- **editing state and re-running from a node**, which is the feature Phase 4's
  checkpointer turns into human-in-the-loop

Notes:

- `--allow-blocking`: the arXiv tool uses blocking `urllib`, and the embedding
  model loads synchronously. The dev server may refuse synchronous I/O inside a
  node; this flag permits it. Fine for local development, and a real signal that
  a production deployment would want async tools.
- Studio needs `ANTHROPIC_API_KEY` in `.env` to actually run a graph. Drawing
  the structure with `research-copilot draw-graph` needs no key for the
  hand-rolled graph (the model is built lazily), but does for
  `draw-graph prebuilt`, since `create_react_agent` binds tools to a real model
  at build time.
- The first `--mode knowledge-base` run downloads the ~90 MB embedding model, so
  give it a minute.
- Dev-server state is in memory and disappears when you stop the server. Phase 4
  adds a checkpointer, and Studio's thread list becomes genuinely useful.

## Usage (Phase 4)

Phase 3's graph, now with a checkpointer under it. `graph.py` gains three nodes
and three arguments; every Phase 1-3 command behaves exactly as before.

```bash
# One turn, persisted. The generated thread_id is printed on stderr.
research-copilot graph-agent "recent methods for evaluating RAG" --checkpointer sqlite

# Continue that conversation from a completely separate CLI invocation
research-copilot graph-agent "which of those needs human labels?" --thread <id>

# Interactive multi-turn against one thread. /state dumps the persisted state.
research-copilot graph-chat --thread my-thread

# Pause for approval before the answer is committed to the transcript
research-copilot graph-agent "..." --thread t1 --approve

# ...then, from another shell, review the parked draft
research-copilot review --thread t1 --approve
research-copilot review --thread t1 --edit "a better answer" --note "tightened"
research-copilot review --thread t1 --reject --note "no sources"
research-copilot review --thread t1            # interactive prompt

# What is on disk
research-copilot threads
```

### Part A: the checkpointer, and thread_id vs State

`compile(checkpointer=saver)` is the whole change. After every super-step,
LangGraph writes a snapshot of State keyed by the **thread_id in `config`**:

```python
graph.invoke(state_input, {"configurable": {"thread_id": "abc"}})
#            ^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#            what this     which conversation it belongs to
#            turn is about
```

They are separate on purpose. State is *content*; config is an *address*. A
node returns partial State, so a node has no way to change which thread it is
writing to — a run cannot wander into another conversation halfway through.
Nodes never see the thread_id unless they ask for `config: RunnableConfig`.

The practical half is in `cli.py`: `_resolve_thread` mints a UUID when you start
and takes `--thread` when you continue, and the id is printed on every run
because the *next* invocation has to pass it back. There is no "current thread".

| | `MemorySaver` | `SqliteSaver` |
| --- | --- | --- |
| lives in | a dict in the process | a file under `data/` |
| survives | across `.invoke()` calls | across processes |
| right for | tests, one interactive session | local dev, seeing what a checkpoint *is* |

Swap in `PostgresSaver` and nothing else changes. Persistence is a compile-time
argument, not a rewrite.

### What persistence quietly breaks

Two Phase 3 assumptions stop holding the moment state survives a turn:

1. **`messages` grows without bound.** Phase 3 started every run empty, so
   Phase 2's token budget stopped mattering. Turn 40 now resends turns 1–39.
   Part B is the fix.
2. **`iterations` never resets.** It is a per-turn tool budget living in
   per-thread storage, so turn 2 would start at 2 and turn 6 at 10 — and
   `should_continue` would refuse tool calls for work earlier turns did. Fixed
   by seeding `iterations: 0` in `turn_input`, and asserted in
   `test_iterations_resets_each_turn_despite_being_checkpointed`.

### Part B: pruning node vs `RemoveMessage`

These look like two options. They are really two questions, and a node answers
both:

| | what the model sees this turn | what stays in the state |
| --- | --- | --- |
| **filtering node** (trim, send, `return {}`) | short | unchanged — grows forever |
| **`RemoveMessage` node** | short | short — the next snapshot really lacks them |

`prune_history` does the second. Two reasons: filtering fixes the token bill and
leaves the durability problem, which is the one Phase 4 is actually about; and
filtering has to be repeated identically by every future reader of that state
(Studio, a Phase 7 API handler, Phase 6's other agents), whereas pruning into
the state makes the shorter history the *actual* history.

The honest cost, and the thing to keep straight: **deleted from state is not
deleted from disk.** A checkpointer writes a new row per super-step and never
rewrites old ones, so a removed message is gone from the *latest* snapshot and
still sits in every earlier row of that thread. That is what makes time travel
work. Pruning is a context-window and token-cost mechanism; if you need a
message gone for real, `checkpointer.delete_thread(thread_id)` is the only
operation that touches history. Asserted both ways in
`test_pruning_shortens_the_current_state_but_not_the_checkpoint_history`.

The node runs **once per turn at the entry**, not inside the tool loop:
pruning between `call_model` and `call_tool` risks orphaning a `ToolMessage`
from the `AIMessage` that requested it, which the Anthropic API rejects
outright. Within a turn growth is bounded by `max_iterations`; across turns it
is unbounded. Prune where the growth is unbounded.

`--memory summarize` reuses Phase 2's `SUMMARY_PROMPT` and writes the gist to
`State["summary"]` — a separate key rather than a `SystemMessage` in `messages`,
because `add_messages` appends and a summary would land *after* the turns it
summarizes.

### Part C: `interrupt()` and `Command`

```
call_model  drafts into State["draft"], status "awaiting_approval"
              (NOT into messages — nothing has been said to the user yet)
review_draft  interrupt(payload) -> the run parks; .invoke() returns __interrupt__
              ... a human decides, possibly days later, in another process ...
              Command(resume=verdict) -> interrupt() returns the verdict
finalize_answer  commits the approved (or edited) text to messages
```

`interrupt()` does not suspend a Python frame. **The node re-runs from its first
line on resume**, and LangGraph feeds the stored resume value to `interrupt()`
when execution reaches it again. So everything above that call happens twice —
which is exactly why `call_model` produces the draft and `review_draft` only
reviews it. A model call before the `interrupt()` would be paid for twice and,
being non-deterministic, would hand the reviewer a verdict on text that no
longer matches what they approved. Keep pre-interrupt work cheap and idempotent.

A draft needs its own state key because `messages` can represent a turn that
*happened*, not a turn that is *proposed*. Had the answer been appended first,
rejecting it would mean editing history instead of declining to write it.

Unrecognized verdicts **fail closed** (`_parse_verdict` treats them as
rejections): an approval gate must never read confusion as consent.

### Part D: the second iteration counter (a note, not code)

`iterations` counts `call_model` calls *within one turn* — the tool loop's
budget. Phase 5 adds a nested cycle (draft → critique → revise), and sharing one
field breaks three things at once: the tool cap trips during revision 2 for work
revision 1 did, so the agent gets worse the harder it tries; the revise cap
trips on tool calls; and neither number means anything when you read the final
state. Phase 5 adds `revisions: int` alongside, with its own cap and its own
routing function. The full reasoning is at the bottom of `state.py`.

### The graph now

```
START ─→ prune_history ──route_by_mode──┬─ "knowledge-base" ─→ retrieve_docs ─┐
                                        └─ "live-search" ───────────────────┐ │
                                    ┌───────────────────────────────────────┴─┘
                                    ↓
                                call_model ──should_continue──┬─ "call_tool" ─→ call_tool ┐
                                    ↑                          │                          │
                                    └──────────────────────────┼──────────────────────────┘
                                                               ├─ "review_draft" ─→ review_draft
                                                               │                       ↓ (interrupt)
                                                               │                  finalize_answer
                                                               │                       ↓
                                                               └─ "end" ─────────────→ END
```

The review nodes are registered **even when `--approve` is off**, in which case
`should_continue` never routes to them and they never run (`compile()` checks
edge targets, not reachability). One graph shape whatever the flags say, so
toggling approval does not mean a checkpoint written by one shape is resumed by
another.

## Usage (Phase 5)

Phase 5 adds no commands, only flags on the graph commands. That is the point:
reflection is not a different way of using the system, it is more edges inside
the same graph.

```bash
# A model reviews the draft before it is finalized. No checkpointer needed.
research-copilot graph-agent "Is RAG obsolete with long-context models?" \
  --critic --max-revisions 2

# Human review only - Phase 4, unchanged.
research-copilot graph-agent "..." --thread t1 --approve

# Both. The critic reviews first; the human is asked only about drafts it passed.
research-copilot graph-agent "..." --thread t1 --critic --approve

# Decompose the question before researching it.
research-copilot graph-agent "Compare RAG and long context on cost and accuracy" \
  --plan --mode knowledge-base
```

### Part A: two loops, two counters

The graph now has two cycles, nested:

| loop | cycle | counter | cap | routing function |
| --- | --- | --- | --- | --- |
| tool | `call_model` ↔ `call_tool` | `iterations` | `max_iterations` | `should_continue` |
| reflection | `call_model` → review → `start_revision` | `revisions` | `max_revisions` | `should_revise` |

They share nothing but the turn they live in. One revision attempt can itself
run the tool loop several times, so a shared counter would make the tool cap
trip during revision 2 for work revision 1 did — the agent would get quietly
*worse* the harder it tried — and neither number would mean anything in the
final state, because you could not tell which cycle spent it.

The rule the split forces: whoever begins a revision resets `iterations` to 0.
`start_revision` does that via `revision_input`, which sits beside `turn_input`
because it is the same pattern one level down. A per-round budget that is never
reset is a budget that only ever runs out.

In a state dump the two numbers read together: `revisions: 2, iterations: 1`
means the second revision has made one model call — not that three have
happened.

### Part B: `should_revise`, and a cap that overrules the verdict

Phase 4's rejection was a dead end. Now it is an edge: `should_revise` sends a
rejected draft back through `start_revision` to `call_model`, with the critique
and the human's note carried in state as the revision instruction.

**Unless the cap is spent, in which case the run ends although the verdict still
says reject.** That override is the substance of the function, not a corner of
it. A reflection loop's exit condition is supplied by the thing the loop is
meant to be checking, and a critic prompted to find fault will find some —
there is always another caveat to want. No reviewer can emit a verdict meaning
"and I promise to stop asking", because that promise is not the kind of thing a
judgement contains. So the loop counts its own attempts and stops on its own
authority: **the verdict decides whether the answer is good, the cap decides
when we are done spending.** Anything that loops on a judgement it did not make
needs a bound it controls itself.

The exhausted branch routes to `finalize_answer`, not to `END` directly:
"stop looping" and "leave the transcript valid" are two requirements, and
ending without `finalize_answer` would leave the opening `HumanMessage`
unanswered — two adjacent human turns, which the Anthropic API rejects.

### Part C: the critic is the human, structurally

| | `review_draft` | `critique_draft` |
| --- | --- | --- |
| reads | `draft` | `draft` |
| asks | a human | a model |
| mechanism | `interrupt()` | `chain.invoke()` |
| parses with | `_parse_verdict` | `_parse_critique` → `_parse_verdict` |
| writes | `status` + `human_feedback` | `status` + `critique` |
| routes into | `should_revise` | `should_revise` |

Everything load-bearing is in the identical rows. A reviewer is structurally
just something that turns a draft into an approve/reject verdict; the graph
routes on the verdict and is indifferent to where it came from. That is why
human-in-the-loop was worth building first even though it is the less automatic
feature — the pattern covers a person at a terminal, a model, a test suite, a
schema validator, or (Phase 6) an agent with its own tools.

**The one real asymmetry:** only the human pauses. `interrupt()` needs a
checkpointer to park a run in, so `--approve` without one is refused at build
time. `--critic` needs nothing; a model call blocks and returns.

Both reviewers share `should_revise` but declare different `path_map`s — each
source lists the destinations reachable *from it*, so the drawn graph shows no
paths a run cannot take.

### Ordering, when both reviewers are on

Worth deciding explicitly rather than falling into. **The critic goes first.**

- **For:** the critic is cheap, automatic and available at 3am; the human is
  none of those, so filtering with the expendable reviewer before spending the
  scarce one is the whole reason to have two. It also keeps the human's word
  final — human-first would mean a model overturning a person's approval.
- **Against:** the human never sees the drafts the critic rejected, so a critic
  with bad taste silently narrows what reaches a person. Worse, with a critic
  that never approves the revision cap is spent *before* the interrupt is ever
  hit: the run ends withheld and `--approve` looks like it did nothing. The CLI
  prints a warning for exactly that, and `test_a_critic_that_never_approves_means_the_human_is_never_asked`
  pins it as behaviour rather than leaving it a surprise.

Human-first is a one-line change (swap the branch in `should_continue` and the
`"awaiting_approval"` case in `should_revise`).

### Failing closed, in two directions

`_parse_critique` hands its result to the same `_parse_verdict` the human path
uses, so an unreadable critique fails closed exactly as an unreadable human
verdict does — but the *cost* differs, and so does the right default elsewhere:

| parser | unreadable input means | costs |
| --- | --- | --- |
| `_parse_verdict` | reject | a retry, vs. publishing something nobody approved |
| `_parse_critique` | reject | one revision round, bounded by `max_revisions` |
| `_parse_plan` | **no plan** | nothing — the run behaves exactly as Phase 4 |

The planner is the odd one out on purpose. Failing closed means failing towards
*doing less*, and for a planner "less" is no decomposition — inventing
sub-questions that then drive retrieval and tool calls would be failing open.

A consequence worth knowing: a critic whose output format drifts becomes a
critic that always rejects, and the symptom is a run that always exhausts its
revisions. The raw reply is kept as the note so that is diagnosable from a state
dump.

### Part D: planning

`plan_question` runs between pruning and the mode branch and writes
`sub_questions` — very often an empty list, which is a *result*, not a failure.
Most of the prompt is about when **not** to decompose, because a model asked to
split a question will split it: "who wrote the BERT paper?" comes back as four
sub-questions and one cheap lookup becomes four.

The plan is advisory. `retrieve_docs` retrieves once per sub-question as well as
for the whole question (question first, duplicates dropped, so a bad plan can
only ever *add* chunks), and `call_model` gets the list as a checklist. There is
no fan-out and no per-sub-question orchestration — one model call still writes
one answer. That is Phase 6's Researcher.

Two interactions are **noted but not solved**, because neither is load-bearing
for Parts A–C:

- **knowledge-base mode.** `n` sub-questions means up to `(1 + n) * k` chunks.
  `prune_history`'s budget does not cover `context` (it is built per call and
  never enters `messages`), so nothing currently stops a wide plan from
  producing a very large prompt.
- **live-search mode.** Sub-questions are paid for out of `max_iterations`. A
  plan with 4 sub-questions against a cap of 6 leaves little room to iterate on
  any of them, and nothing couples the two numbers. `--max-sub-questions`' cap
  is the blunt instrument in the meantime.

### The graph now

```
START ─→ prune_history ─→ plan_question ──route_by_mode──┬─ "knowledge-base" ─→ retrieve_docs ─┐
                                                         └─ "live-search" ───────────────────┐ │
                                    ┌────────────────────────────────────────────────────────┴─┘
                                    ↓
 ┌──────────────────────────→  call_model ──should_continue──┬─ "call_tool" ─→ call_tool ┐
 │                                  ↑                         │                          │
 │                                  └─────────────────────────┼──────────────────────────┘
 │                                                            ├─ "critique_draft" ─→ critique_draft
 │                                                            ├─ "review_draft" ──→ review_draft
 │                                                            └─ "end" ─────────────────→ END
 │                                                                     │       │
 │                                  both route through should_revise ──┴───────┘
 │                                              │
 │  start_revision ←── "start_revision" ────────┤
 └────────┘                                     ├── "review_draft" ─→ review_draft   (critic passed;
                                                │                                     human still owes
                                                │                                     a verdict)
                                                └── "finalize_answer" ─→ finalize_answer ─→ END
```

All five Phase 4/5 optional nodes are registered **whatever the flags say**.
`plan_question` is on the unconditional path and returns `{}` when planning is
off; the review nodes are simply unreachable. Flags choose paths, not
structures — so a checkpoint written under one set of flags is not resumed by a
different graph.

### What needs a real `ANTHROPIC_API_KEY`

The fakes pin down mechanics — where the run goes after a verdict, which
counter moves, whether the cap saves you. They can say nothing about output
*quality*, which is the whole question Phase 5 raises:

1. **Do critiques come back specific enough to act on?** A critique like "needs
   more detail" is one `call_model` cannot use, so the revision returns
   unchanged and the loop burns its budget rediscovering that. This is the
   single most important thing to check, because it is the difference between a
   reflection loop and an expensive no-op.
2. **Do revised drafts actually improve?** Run the same question with
   `--max-revisions 0` and `--max-revisions 2` and read both. Reflection is
   widely assumed to help and does not always.
3. **Does the critic ever approve?** A real model on a real prompt may reject
   everything, in which case every run exhausts its cap and `--critic` is pure
   cost. Check the approve rate before trusting the feature.
4. **Does the first-line format hold?** `_parse_critique` fails closed to
   "reject", so format drift is indistinguishable from a harsh critic.
5. **Does the planner decline to plan?** Ask something simple and confirm you
   get `NONE`. A planner that always plans is a cost multiplier.
6. **`--critic --approve` on a real thread**, to see the critic actually filter
   before a human is asked — and to confirm the withheld-after-N-revisions path
   reads clearly when it happens for real.
7. **Token cost.** `--critic --plan` on a live-search question is up to
   `1 + n` planner/retrieval calls plus a critic call per round plus the tool
   loop per round. Watch a LangSmith trace once before leaving it on.

## Tests

```bash
pytest
```

Tests use LangChain's fake chat models, so they need no API key or network
access and cost nothing. Being able to swap a fake model into a chain is part
of what Phase 1 teaches.

## Layout

```
src/research_copilot/
  config.py        loads .env once; typed settings
  models.py        chat model factory (ChatAnthropic)
  prompts.py       ChatPromptTemplates + message-type notes
  schemas.py       Pydantic output schemas (ResearchAnswer)
  chains.py        LCEL chains: answer (str) and structured (Pydantic)
  tools/arxiv.py   @tool search_arxiv (free arXiv API, no key)
  agent_loop.py    hand-written tool-call loop
  memory.py        conversation history + trim/summarize strategies (Phase 2)
  ingest.py        load -> chunk -> embed -> store (Phase 2)
  retrieval.py     embeddings, Chroma store, retriever, RAG chain (Phase 2)
  state.py         the graph's State TypedDict + reducer notes (Phase 3, 4)
  graph.py         the hand-rolled StateGraph: nodes, edges, routing (Phase 3, 4)
  prebuilt.py      the same agent via create_react_agent, for comparison (Phase 3)
  checkpointing.py checkpointer factory + the thread_id vs State notes (Phase 4)
  multi_agent_state.py  Phase 6 State: one owner per field, budgets, contracts
  multi_agent_graph.py  Phase 6 graph: the Supervisor hub, reviews, revisions
  agents/          researcher.py (subgraph), writer.py (node), critic.py (subgraph),
                   supervisor.py (routing + guards) - Phase 6
  tools/citations.py    @tool verify_citation, the Critic's tool (Phase 6)
  pruning.py       Phase 4's prune_history as a factory, for the Phase 6 graph
  studio_demo.py   the Phase 6 graph with scripted models, for offline Studio runs
  cli.py           command-line entry point
langgraph.json     tells LangGraph Studio where the graphs are (Phase 3; Phase 6 adds two)
data/chroma/       the local vector store (gitignored, created by `ingest`)
data/checkpoints.sqlite3  the checkpoint database (gitignored, Phase 4)
tests/             offline tests using fake models
                   test_reflection.py / test_planning.py are Phase 5
                   test_multi_agent*.py, test_supervisor.py, test_critic_agent.py,
                   test_citations.py, test_revisions_and_budgets.py are Phase 6
```

`agent_loop.py` and `retrieval.py` stay in place on purpose. `graph.py` is
additive, so the manual loop and the graph can be run against the same question
to see exactly what changed.

## Where each Phase 1 concept lives

| Concept | File |
| --- | --- |
| Message types (System/Human/AI/Tool) | `prompts.py`, `cli.py` (`messages` mode) |
| `ChatPromptTemplate`, `.partial()` | `prompts.py`, `chains.py` |
| Chat models as Runnables | `models.py` |
| LCEL `\|` composition | `chains.py` |
| `StrOutputParser` | `chains.py` → `build_answer_chain` |
| `PydanticOutputParser` | `schemas.py`, `chains.py` → `build_structured_chain` |
| `@tool` and tool schemas | `tools/arxiv.py` |
| `bind_tools` + manual tool-call loop | `agent_loop.py` |

## Where each Phase 2 concept lives

| Concept | File |
| --- | --- |
| `MessagesPlaceholder` (+ optional placeholders) | `prompts.py` → `CHAT_PROMPT` |
| Stateless models / history as memory | `memory.py` |
| Token budget vs. message count | `memory.py` → `token_counter` |
| `trim_messages` | `memory.py` → `_trim` |
| Summarization as compression | `memory.py` → `_summarize`, `prompts.py` → `SUMMARY_PROMPT` |
| Document loaders | `ingest.py` → `_load_file` |
| `RecursiveCharacterTextSplitter`, chunk overlap | `ingest.py` → `chunk_documents` |
| Embeddings and their input limits | `retrieval.py` (module docstring) |
| Vector store, retriever as a Runnable | `retrieval.py` → `get_vector_store`, `get_retriever` |
| `RunnableParallel` / `RunnablePassthrough.assign` | `retrieval.py` → `build_rag_chain` |
| Grounding and refusal-to-guess prompting | `prompts.py` → `RAG_PROMPT` |

## Where each Phase 3 concept lives

| Concept | File |
| --- | --- |
| `StateGraph`, nodes, edges, `compile()` | `graph.py` (module docstring) |
| State as a `TypedDict` | `state.py` → `State` |
| Reducers, and why `messages` needs one | `state.py` (module docstring) |
| `add_messages` (append, id-replace, coercion) | `state.py` → `State.messages` |
| Nodes as `State -> partial update` | `graph.py` → `retrieve_docs`, `call_model`, `call_tool` |
| Conditional edges / routing functions | `graph.py` → `route_by_mode`, `should_continue` |
| `path_map`, and why it matters for drawing | `graph.py` → `add_conditional_edges` calls |
| Cycles, and the guard that escapes them | `graph.py` → `should_continue`, `State["iterations"]` |
| `START` / `END` sentinels | `graph.py` → wiring section |
| Inspecting structure before running | `cli.py` → `cmd_draw_graph` |
| `create_react_agent` and what it hides | `prebuilt.py` (module docstring) |

## Where each Phase 4 concept lives

| Concept | File |
| --- | --- |
| Checkpointers, and what a checkpoint contains | `checkpointing.py` (module docstring) |
| `thread_id` in `config` vs. content in `State` | `checkpointing.py`, `cli.py` → `_resolve_thread`, `cmd_graph_chat` |
| `MemorySaver` vs `SqliteSaver` (one interface) | `checkpointing.py` → `checkpointer_scope` |
| `compile(checkpointer=...)` | `graph.py` → end of `build_graph` |
| A per-turn budget in per-thread storage | `graph.py` → `turn_input` (`iterations: 0`) |
| `RemoveMessage` + `add_messages` as deletion | `graph.py` → `prune_history` |
| Pruning node vs. filtering, and disk vs. state | `graph.py` (Part B docstring), `tests/test_pruning.py` |
| `interrupt()` and re-running a node on resume | `graph.py` → `review_draft` |
| `Command(resume=...)` | `graph.py` → `resume_graph` |
| Reading a thread without running it | `graph.py` → `pending_interrupt` |
| A draft as state the transcript cannot hold | `state.py` → `draft`, `status`, `ReviewStatus` |
| Failing closed on an unparseable verdict | `graph.py` → `_parse_verdict` |
| Splitting iteration counters for Phase 5 | `state.py` → `revisions` |

## Where each Phase 5 concept lives

| Concept | File |
| --- | --- |
| Two nested loops, two counters, two caps | `state.py` → `revisions`; `graph.py` → `revision_input` |
| Resetting a nested budget per round | `graph.py` → `start_revision`, `revision_input` |
| A cap that must overrule a verdict | `graph.py` → `should_revise` |
| Rejection as an edge rather than an ending | `graph.py` → wiring, `critique_draft`/`review_draft` edges |
| Critic-as-reviewer symmetry | `graph.py` → `critique_draft` (docstring) |
| Why the reviewers' `path_map`s differ | `graph.py` → wiring section |
| One verdict shape, two reviewers | `graph.py` → `_parse_critique` → `_parse_verdict` |
| Failing closed in two directions | `graph.py` → `_parse_critique` vs `_parse_plan` |
| Ordering when both reviewers are on | `graph.py` → `critique_draft`; `cli.py` → `_announce_review_setup` |
| How a critique reaches the writer | `graph.py` → `_revision_instruction` |
| Why a verdict is not a routing decision | `state.py` → `ReviewStatus` |
| Decomposition, and declining to decompose | `graph.py` → `plan_question`; `prompts.py` → `PLAN_PROMPT` |
| Config a resume must repeat (closures vs State) | `cli.py` → `cmd_review` |
| What Phase 6 needs on top of this | `graph.py` (bottom, PHASE 6 NOTE) |

## Phase 6: Multi-agent (reference)

Phase 5 had one `call_model` node that searched, wrote, and revised: one model
wearing several prompts. Phase 6 splits that work across three agents
coordinated by a Supervisor, in a **new graph** (`multi_agent_graph.py`).
Phase 5's `graph.py` and `state.py` are untouched, and `graph-agent` still runs
them.

This section is the reference to read before Phase 7. It describes the system
as it stands after 6.4, not the order it was built in (that is in the
`phase-6` commits: 6.1 agents, 6.2 Supervisor, 6.3 Critic and budgets,
6.4 persistence, Studio, and this document).

### The shape

```
START → begin_turn → prune_history → plan_question → supervisor ─┬─ researcher ─┐   (Researcher: subgraph)
                                                         ↑        ├─ writer ─────┤   (Writer: node)
                                                         │←───────┴──────────────┘
                                                         │
                                                         │        ├─ critic ──after_critique──┬─ approve ─→ review_draft* / finalize_answer
                                                         │        │  (subgraph)               ├─ reject, rounds left ─→ start_revision ─┐
                                                         │        │                           └─ reject, rounds spent → finalize_answer │
                                                         │        └─ finish ─→ review_draft* / finalize_answer                          │
                                                         └───────────────────────────────────────────────────────────────────────────────┘
review_draft* ──after_review──┬─ approve / edit → finalize_answer
                              ├─ reject, rounds left → start_revision → supervisor
                              └─ reject, rounds spent → finalize_answer (withheld)        * only with --approve
```

Every agent reports back to the Supervisor. A rejection from either
reviewer counts a revision in `start_revision` and then goes to the
Supervisor, which decides who fixes it. There is no edge from any reviewer to
the Writer.

### Running it

```bash
research-copilot multi-agent "How are RAG pipelines evaluated?"                 # Supervisor routes (needs a key)
research-copilot multi-agent "..." --routing fixed                              # research -> write -> finish, no Supervisor model
research-copilot multi-agent "..." --critic                                     # Critic on the roster
research-copilot multi-agent "..." --critic --approve                           # ...then a human, after the Critic
research-copilot multi-review --thread <id> [--approve | --reject --note ".." | --edit ".."]
research-copilot multi-agent "follow-up" --thread <id>                          # next turn, same conversation
research-copilot multi-agent "..." --mode knowledge-base --plan
research-copilot draw-graph multi                                               # both subgraphs expanded (xray)
```

Every run prints its hand-offs as they happen. Supervisor decisions show their
rationale, and any override appears beside what the model proposed. Steps
inside a subgraph are indented:

```
  [supervisor] -> critic  (proposed finish; OVERRIDDEN: finish proposed before the critic approved this draft)
      why: Draft ready.
    [critic/critic_model] critic_messages=[1], critic_iterations=1
    [critic/critic_tools] critic_messages=[1]
    [critic/compile_verdict] verdict='reject', critique='Writing: cites ARES ...', citation_checks=[2], ...
  [critic] REJECT  checks: 2309.15217=found, 2311.09476=found
      note: Writing: cites ARES (2311.09476), which exists but is not in the research notes - the Writer added it.
  [start_revision] revisions=1, budgets={... used: 0 ...}
  [supervisor] -> writer
      why: Critic says the Writer added a source not in the notes: a writing problem.
```

### The agents

| Agent | Shape | Tools | Reads | Writes (owns) |
| --- | --- | --- | --- | --- |
| Researcher | subgraph (tool loop) | `search_arxiv` / retriever | question, mode, plan, transcript, summary, `researcher_brief`, its own previous notes | `research_notes`, `documents`, `research_iterations`, `research_outcome`, `budgets["researcher"]` |
| Writer | node (one call) | none | question, plan, transcript, summary, notes, critique + human feedback on a revision | `draft`, `budgets["writer"]` |
| Critic | subgraph (verification loop) | `verify_citation` | question, plan, draft, research notes | `critique`, `verdict`, `citation_checks`, `budgets["critic"]` |
| Supervisor | node (structured output) | none | everything above, excerpted, plus its own log | `next_agent`, `researcher_brief`, `dispatches`, `supervisor_log` |

**Subgraph vs. node.** Make an agent a subgraph when it has a loop, or state
it must keep to itself. Otherwise make it a node. The Researcher and Critic
loop over tools and produce scratch work nobody else should read; the Writer
makes one call. An agent is a role, not a unit of graph structure.

**Private channels.** The Researcher's tool loop runs in `research_messages`,
and the Critic's verifications in `critic_messages`. Neither is in
`MultiAgentState` or in the subgraph's output schema, so neither reaches the
shared transcript, another agent's input, or `get_state()`. Two things
"private" does **not** mean:

- **Not off disk.** With a checkpointer, each subgraph checkpoints under its
  own namespace (`researcher:<task-id>`, `critic:<task-id>`), and those
  checkpoints hold every search result and verification. If a tool result
  must never be persisted, keep it out of the checkpointer. A private key does
  not do that.
- **Not invisible while running.** In the `updates` and `values` stream modes,
  a nested subgraph's steps are narrowed to its output schema, so a step that
  wrote only its private channel looks like it did nothing. The `tasks` mode
  shows the full write. The CLI trace uses `tasks` for inner steps for this
  reason.

Each channel has two protections: a name the parent does not have, and an
output schema that filters it. Removing either one alone leaks nothing;
removing both leaks (the tests check each case).

**The Researcher's contract across passes.** The Supervisor can send the
Researcher back with a `researcher_brief`. A follow-up pass **merges**: code
appends the new findings under a header naming the brief, and never lets the
model rewrite earlier notes. Knowledge-base excerpts keep their `[n]` numbers.
`research_outcome` (`findings` / `nothing_found` / `budget_exhausted`)
describes the latest pass as a fact the guards can branch on.

**The Critic** is Phase 5's `critique_draft` promoted to an agent. It keeps
the same APPROVE/REJECT first line and the same fail-closed parsers, imported
from `graph.py`, not copied. What it adds: it reads the research notes, and it
checks each arXiv citation with `verify_citation`, which returns FOUND,
NOT FOUND, INVALID or ERROR. **ERROR is not evidence against a citation.** A
timed-out lookup must not reject a draft, and the prompt says so. Running out
of budget mid-verification counts as a rejection (fail closed).

### One field, one owner

During a turn, every state key has exactly one writer. Plain nodes are
wrapped in `owns(...)`, which raises `OwnershipError` if a node returns a key
(or a `budgets` entry) it does not own. Subgraphs are held to the rule by
their `output_schema`, and a test keeps each schema equal to its `OWNERS`
entry.

The deliberate exceptions, each written down:

| Writer | Writes | Why it is allowed |
| --- | --- | --- |
| `begin_turn` | every per-turn field | it is the turn boundary. It runs at START, before any agent, for every caller |
| `prune_history` | `messages` (removals), `summary` | runs once, at the start of the turn, never while an agent works. `finalize_answer` only appends, at the end |
| `start_revision` | every agent's `budgets` entry, `revisions` | the per-round reset site |

A human edit goes into `human_edit`, never into the Writer's `draft`. The
final state shows both.

**The turn reset is a node, not an input helper.** Through 6.3 the per-turn
reset lived in `multi_agent_turn_input`. Studio does not call that helper, and
neither will Phase 7's API, so a second turn in Studio inherited the first
turn's dispatch counts, revisions and log. `begin_turn` fixes that for every
caller. It is safe to run unconditionally because START is only entered by a
new turn: a resume re-enters at the parked node.

### The Supervisor

- **Why a model:** "which agent should fix this critique?" means reading the
  critique against the notes. "Cites a paper that is not in the notes" is the
  Writer's fault even though it is about a citation. That is a judgement, not
  a lookup.
- **Structured output:** `with_structured_output(SupervisorDecision,
  method="json_schema")`. This uses the API's native structured output, which
  constrains generation to the schema. The default `function_calling` method
  forces a tool call, which the API rejects when thinking is on (and newer
  models reject forced tool calls outright).
- **Rationale first:** `rationale` precedes `next` in the schema, so the route
  is generated after the reasoning, not justified afterwards.
- **What it sees** (`render_supervisor_view`): the question and plan; notes
  (excerpted, with truncation marked) and `research_outcome`; the draft,
  marked STALE or REJECTED when it is; the critique, marked current or earlier,
  plus the citation checks; human feedback; revisions used; per-agent
  dispatches and round budgets; its own earlier decisions. It never sees a
  private channel.
- **Code guards have the last word** (`apply_guards`):

| Guard | Proposal | Routed to |
| --- | --- | --- |
| agent not on the roster (e.g. Critic with `--critic` off) | that agent | `fixed_policy` |
| dispatch cap reached, or round budget spent | that agent | `fixed_policy` |
| no research yet | `writer` | `researcher` |
| draft missing, stale, or rejected | `critic` / `finish` | `writer` |
| Critic on, and it has not approved *this* draft | `finish` | `critic` |
| output raises or fails validation | — | `fixed_policy` (the attempted route is still logged when recoverable) |

  A guard never routes to an agent that is unavailable. Every route except
  `finish` spends a capped dispatch, so a turn makes at most
  `sum(dispatch caps) + max_revisions + 1` decisions, **whatever the model
  says**. Tests check this against adversarial scripts.
- **The log** (`supervisor_log`) records, per decision: what the model
  proposed, where the run went, why they differ, the brief, and the revision
  number. This is the debugging surface, and what Phase 7's evaluation should
  read.

### Staleness, from dispatch order alone

No `critiqued_draft` field and no hashes. The log's order already holds the
answer.

| Question | Derived from `supervisor_log` |
| --- | --- |
| Is the draft stale? | the last dispatch was the Researcher, and that pass changed the notes |
| Was the draft rejected and not rewritten? | `revisions > 0`, and no Writer dispatch is stamped with the current revision |
| Is the critique current? | the Critic was dispatched after the last Writer dispatch (one rewrite since or three, it is stale either way) |

A stale *rejection* is normal: it is the reason the Writer is revising. A
stale *approval* approves nothing.

### Counters: four of them, three scopes

| Counter | Scope | Reset by | What it bounds |
| --- | --- | --- | --- |
| `research_iterations` | one Researcher pass | (private start) | nothing: a report of that pass |
| `budgets[agent]["used"]` | one **revision round** | `start_revision` | work inside a round, e.g. a Researcher sent back twice in a round shares one tool budget |
| `dispatches[agent]` | one **turn** | `begin_turn` only, **never** `start_revision` | the hub: termination |
| `revisions` | one **turn** | `begin_turn` only | rejection rounds (`max_revisions`) |

`dispatches` is the number to read as "how many times this turn sent work to
that agent". It means that and nothing else, because it is reset on exactly
one schedule. A mutation that reset it per revision broke the termination
test. Dispatch caps default to room for every allowed revision: Writer
`2 + max_revisions`, Critic `1 + max_revisions` (0 when off), Researcher 2.

`budgets` is `dict[str, AgentBudget]`, where `AgentBudget` is a TypedDict
`{used, cap}`. A TypedDict rather than a dataclass because it is a plain dict:
it serializes into checkpoints and shows in Studio like everything else. It is
reduced by `merge_budgets`, which merges per agent *and* per field, so the
Writer's update cannot erase the Researcher's entry, and a reset can write
`{"used": 0}` without knowing the cap. `cap` in state is a record for readers;
enforcement reads the build config.

### Schema migration: reducer-backed keys vs. plain keys

**Read this before adding or renaming a key.** An old thread (checkpointed
before a key existed) reads back *differently* depending on how the key is
declared, and the difference is easy to get backwards by analogy:

| Declared as | On a thread from before the key existed | Example |
| --- | --- | --- |
| plain key (`verdict: str`) | **missing**: `"verdict" not in state` | `state.get("verdict", "")` is correct |
| reducer-backed (`budgets: Annotated[dict, merge_budgets]`) | **present and empty**: `state["budgets"] == {}` | `"budgets" in state` is `True`, which says nothing |

LangGraph gives every reducer channel an empty default, so a presence check
on a reducer-backed key is always true, and "is this an old thread?" gets the
wrong answer. The same applies to `messages` and any future `Annotated` key.

**`budget_of(state, agent, default_cap)` is the only sanctioned way to read a
budget.** It treats absent, empty, and partially-filled entries the same:
"nothing used, configured cap", which is exactly what an agent that has not
run this round has spent. Never index `state["budgets"][agent]` directly.
`tests/test_revisions_and_budgets.py::test_a_pre_6_3_thread_with_no_budgets_resumes_cleanly`
pins both behaviours.

A related trap, found in 6.4: **`graph.get_state()` is filtered by the reading
graph's schema.** Ask the single-agent graph about a multi-agent thread and
the multi-agent keys are simply absent. So "which graph wrote this thread?"
must be asked of the checkpointer's raw channel values (`_raw_channels` in
`cli.py`), never of `get_state()`. The first version of the cross-graph guard
used `get_state()`, and Phase 4's `review` resumed a multi-agent thread with
the wrong graph.

### Persistence and review

- `--thread` / `--checkpointer` / `--approve` / `--memory` /
  `--max-history-tokens` work exactly as on `graph-agent` (Phase 4). The
  thread_id goes in `config`, never in state.
- **`run_policy` is stored in state.** Phase 5's `review` had to be given
  `--critic` and `--max-revisions` again, because closures do not survive a
  process. Phase 6 has far more such configuration, so each turn writes its
  build policy into state, and `multi-review` rebuilds the graph from it. A
  resume cannot run under a different policy from the pause. A *new turn* may
  change it (`--critic` on turn 2 only is fine).
- **Refusals, each for a reason:**
  - `multi-agent --thread X` while X is paused at review → refused. LangGraph
    would start the new turn, abandon the draft, and leave two human turns
    adjacent, which the Anthropic API rejects on the next call.
  - `review` on a multi-agent thread, `multi-review` on a single-agent thread
    → refused. Both graphs share one checkpoint DB.
  - `multi-review` on a multi-agent thread from before 6.4 (no `run_policy`)
    → refused rather than resumed under a guessed policy.
  - `multi-review` where only subgraph checkpoints exist and nothing is parked
    → "nothing is awaiting approval". Subgraph checkpoints are history, not a
    pending decision.
- **Pruning** (`prune_history`, Phase 4's node via `pruning.py`) runs at the
  entry. This transcript is small by construction: two messages per turn,
  because tool and verification traffic lives in private channels. But a
  long-lived thread still grows without bound, so pruning is on by default. It
  never runs mid-turn: a resume re-enters at the parked node.

### Flag meanings across phases

Flags keep their names across phases, not always their meaning. `graph-agent`
keeps its Phase 5 meanings; `multi-agent` has these:

| Flag | Phase 4 | Phase 5 (`graph-agent`) | Phase 6 (`multi-agent`) |
| --- | --- | --- | --- |
| `--critic` | — | a `critique_draft` node reviews every draft; a rejection goes back to `call_model` | puts the Critic *agent* on the Supervisor's roster; it verifies citations against arXiv and the draft against the notes; a rejection goes to the Supervisor, which picks the Researcher or Writer |
| `--max-revisions` | — | rejection rounds per turn | same, from either reviewer; also sizes the Writer/Critic dispatch caps |
| `--approve` | human gate before commit | same; after the critic when both are on | same: after the Supervisor finishes, after the Critic approves when both are on; resumed with `multi-review`; a human rejection is classified by the Supervisor too |
| `--plan` | — | `plan_question` before research | unchanged |
| `--routing` | — | — | `supervisor` (a model decides) or `fixed` (research → write → [critic] → finish, no model) |

### Studio

```bash
pip install -e ".[studio]"
langgraph dev --allow-blocking
# then open https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
```

`langgraph.json` registers four graphs. Two are Phase 6:

- **`multi_agent`**: the real graph. Studio can *draw* it without a key.
  Running it needs `ANTHROPIC_API_KEY`.
- **`multi_agent_demo`** (`studio_demo.py`): the same graph with scripted
  models and stub tools. It runs offline and takes the same instructive path
  every time: research, draft, finish overridden to the Critic, rejection
  classified as a writing problem, rewrite, approval.

What was verified against the dev server's API (the data Studio renders):
the collapsed graph shows `researcher` and `critic` as single nodes; xray
expands both into their loops, and `writer` has nothing inside. The subgraph
state schemas include the private channels. The parent thread state never
contains them. The `tasks` stream carries the private writes, while the
`updates` stream shows those inner steps as empty. After the `begin_turn`
fix, turn 2 on a thread starts from fresh counters.

Things that look odd in Studio and are deliberate:

- `review_draft` and the Critic are drawn even when a run cannot reach them
  (one graph shape, whatever the flags).
- Conditional edges have no labels. The drawing shows what is *possible*; the
  run timeline and `supervisor_log` show what *happened*.
- An inner Critic or Researcher step can show an empty update in Studio's
  updates view (see "private channels" above).
- A thread started in Studio has no `run_policy`, so `multi-review` refuses
  to resume it. Studio threads are for looking at, not for the CLI.

### What needs a real `ANTHROPIC_API_KEY` (updated by Phase 7, Part A)

This list was untested when Phase 6 closed: the fakes fix every model reply in
advance. Phase 7 Part A ran it live on **Groq, `openai/gpt-oss-120b`** (free tier),
not on Anthropic. So each item now has one of three statuses:

- **verified on Groq**: evidence about prompts and logic, on one open model
- **pending Anthropic**: the question is about Anthropic's API itself; Groq cannot answer it
- **open**: tried, and the answer was "not yet"

Details, per-run reports and the hand reviews are in `docs/live_check/`; start at
`SUMMARY-groq-gpt-oss-120b.md`.

1. **Structured output in this configuration** (`json_schema` + the
   server-side-fallback beta). **Pending Anthropic.** Groq's own path works
   (strict `json_schema` on gpt-oss), but that is a different method on a
   different provider. Found on Groq and still to check on Anthropic: **field
   order is not generation order**. gpt-oss writes `next` before `rationale`
   even under strict `json_schema`, so rationale-first is a hint, not a
   guarantee (docstring corrected in `agents/supervisor.py`).
2. **Supervisor routing quality.** **Verified on Groq, with caveats:** 12/17
   strict, 16/17 lenient. Single samples; 4 routes flipped between runs with
   no targeted change, so run-to-run variance can't be separated from prompt
   effects until Part E scores rates over repeated runs. Open design
   question: an open gap in the notes can outrank the critique (SUP06, SUP08).
3. **Rationale vs. route.** **Verified on Groq, and sharper than expected.** No
   rationale argued against its own route. The real failure was a consistent
   rationale built on a **false premise** (SUP04, SUP08): the Supervisor
   repeated a critique's claim without checking the notes it was holding.
   Fixed with a "check the critique against the state" instruction: 2 → 0.
   New after the fix: **scope creep**, invented "be comprehensive"
   requirements, 1 → 2. Scenarios now carry `premise_facts`, so a judge can
   check a rationale against the state, not just against its route.
4. **Researcher.** **Open.** It never ended its own search loop (5 searches,
   then the cap). Fixed by code: the last call is made with **no tools** and
   the results as text, so every pass ends in written notes (RES01 passes).
   Still open: it restates facts its notes already hold instead of answering
   `NOTHING NEW` (RES02, every run), and showing it the remaining budget did
   not reduce its searching.
5. **Writer.** **Verified on Groq for isolated scenarios, open end to end.**
   4/4 alone: it cites only the notes, admits empty notes, and drops flagged
   citations. End to end it did worse:
   - gpt-oss wrote its own citation markup (`【1†L1-L7】`), which points at
     nothing. Now caught by `unsupported_citations` (one retry, then
     recorded), not normalised.
   - E2E01's confirming run delivered a real paper, arXiv 2408.12398, that
     the notes did not contain. The check fired and the retry did not remove
     it. Detection now works; *acting* on the detection does not yet
     (see item 6).
6. **Critic.** **Verified on Groq**: 6/6, twice, the second time after the
   redesign below. It reads lookup titles (spots "Attention Is All You Need"
   cited as RAGAS), ignores ERROR, and catches fake, malformed and
   misattributed ids. Redesigned after E2E01: citation lookups are now code,
   not model-invoked tool calls, so a review costs one model call. **Open:**
   it approved E2E01's final draft despite the unsupported citation in item 5.
   It checks that each cited paper *exists*, but it is not shown the Writer's
   "not in the notes" flag, and it did not notice on its own.
7. **Cost and latency, and whether a full run works at all.** **Verified on
   Groq.** E2E01 with the Critic delivered nothing before the fixes (70k
   tokens, 7.5 min, withheld). After them: an approved answer in one pass (0
   revisions, no fallbacks), 48k tokens, 4.8 min, 12 model calls, with the
   Critic taking 1 of them. Without the Critic: 14k tokens, 57 s. The largest
   single request was the Researcher's 5th call, at ~6,000 tokens (the
   per-request limit is below).
8. **Real arXiv.** **Verified.** A well-formed id that does not exist returns an
   empty feed, which reads as NOT FOUND, as assumed. The shared 3-second
   throttle held with both tools active.
9. **Studio, visually.** **Still open.** API-level checks only.

### Open going into Phase 7

**Blocking for any parallel fan-out design:**

- **Budget passthrough from subgraphs is a lost-update bug under concurrency.**
  The Researcher and Critic subgraphs read `budgets` and return the *whole*
  dict: their own entry, plus every other agent's entry passed through
  unchanged. Under `merge_budgets` that passthrough is a no-op **only because
  nodes run one at a time**. The moment two branches run in parallel (two
  Researchers on different sub-questions, or Researcher and Critic together),
  each merges a full-dict snapshot taken before the other's update, and the
  later merge silently restores the earlier value. `owns()` cannot catch this,
  because subgraphs are not wrapped and output schemas work per key, not per
  entry. **Resolve before fan-out:** per-agent keys (`researcher_budget`, ...)
  or a delta reducer, not a cleverer merge.
  `test_subgraph_budget_passthrough_leaves_other_entries_alone` pins the
  sequential behaviour only.

Also open:

- **Private channels are persisted.** Every search result and verification is in
  the checkpoint DB under subgraph namespaces. Deployment needs a retention or
  deletion story (`delete_thread` is the only thing that removes them).
- **Cost:** a Supervisor model call per hop, including hops the guards would
  have forced anyway. Skipping the call when only one route is legal would
  change what the log records, so it is left undecided.
- **No quality measurement.** Routing, classification, citation checking, and
  answer quality have no evaluation. `supervisor_log` (proposed vs. routed,
  overrides, rationale, revision) and the counter scope table above are the
  inputs a LangSmith dataset evaluation should read.
- **Streaming UX:** the CLI streams hand-offs, but not tokens (Phase 7's
  `astream_events`), and the right stream mode differs for parent vs. inner
  steps.
- **Retries and fallbacks:** a failed model call inside an agent currently fails
  the run. Only the Supervisor has a fallback (the fixed policy).
- **Blocking I/O:** arXiv lookups and embeddings are synchronous. `langgraph dev`
  needs `--allow-blocking`, and a deployment would want async tools.
- **`run_policy` validation:** a resumed thread trusts whatever policy it stored.
  A policy written by an older build with different keys is passed straight to
  `build_multi_agent_graph`.

### Where each Phase 6 concept lives

| Concept | File |
| --- | --- |
| Why one `draft` field breaks with several agents | `multi_agent_state.py` (module docstring) |
| One field, one owner; `owns`; budget-entry ownership | `multi_agent_state.py` → `OWNERS`, `BUDGET_ENTRY_OWNERS`, `owns` |
| The turn boundary as a node | `multi_agent_graph.py` → `begin_turn`, `per_turn_reset`, `multi_agent_turn_input` |
| `input_schema` / `output_schema` as an agent's contract | `multi_agent_state.py` → `ResearcherInput/Output`, `CriticInput/Output` |
| An agent as a subgraph; private channels and their limits | `agents/researcher.py`, `agents/critic.py` (module docstrings) |
| Why the Writer is a node | `agents/writer.py` (module docstring) |
| An agent must always hand something over | `agents/researcher.py` → `compile_notes`; `agents/critic.py` → `compile_verdict` |
| Briefs and merge-on-rerun | `agents/researcher.py` (6.2 docstring), `_merge`, `retrieve` |
| Checking a citation; ERROR is not NOT FOUND | `tools/citations.py`; `tests/test_citations.py` |
| Why routing is a model's judgement; structured output | `agents/supervisor.py` (module docstring), `SupervisorDecision` |
| What the Supervisor sees | `agents/supervisor.py` → `render_supervisor_view` |
| Guards, roster, fallback, termination bound | `agents/supervisor.py` → `apply_guards`, `fixed_policy`; `tests/test_supervisor.py` |
| Staleness (draft, rejection, critique) | `agents/supervisor.py` → `draft_is_current`, `draft_was_rejected`, `critique_is_current` |
| Decision in a node, edge in a routing function | `multi_agent_graph.py` → `route_from_supervisor` |
| Rejections through the Supervisor; the cap overrules the verdict | `multi_agent_graph.py` → `after_critique`, `after_review`, `start_revision` |
| Budget TypedDict, reducer, safe access | `multi_agent_state.py` → `AgentBudget`, `merge_budgets`, `budget_of` |
| Counter scopes | `multi_agent_state.py` → `dispatches`; `multi_agent_graph.py` (bottom) |
| Why the decision log does not use `operator.add` | `multi_agent_state.py` → `supervisor_log` |
| A human edit as its own field | `multi_agent_state.py` → `human_edit` |
| Stored run policy vs. repeated flags | `multi_agent_state.py` → `run_policy`; `cli.py` → `cmd_multi_review` |
| `get_state()` is schema-filtered; raw channels | `cli.py` → `_raw_channels` |
| Stream modes and subgraph visibility | `cli.py` → `_stream_multi_agent` |
| Pruning a small-but-unbounded transcript | `pruning.py`; `multi_agent_graph.py` (prune note) |
| The offline Studio demo | `studio_demo.py` |

## Lesson: code guards encode assumptions

This project's recurring rule, since Phase 5's `should_revise`, has been
**a code guard has the last word over the model.** The model decides quality;
code decides when to stop, what is allowed, and what must happen. Every such
guard was built to catch a *model* being wrong.

Phase 7's first real end-to-end run (E2E01, Groq) produced the first case of a
guard overruling a **correct** model decision:

1. The Critic ran out of its call budget mid-review. It "failed closed" as a
   rejection.
2. The Supervisor read the situation right: *"the critic could not complete
   verification due to budget limits … let the critic finish reviewing."*
3. A 6.3 guard - "a rejected draft must be rewritten before it is reviewed
   again" - overruled it and forced a rewrite of a draft nobody had faulted.
   The Critic ran out again, the revision cap was spent, and the run delivered
   nothing.

The guard was not buggy. It did exactly what it was written to do. It
**encoded an assumption** - *every rejection is about the content* - and that
assumption was false the first time a rejection came from somewhere else.
Offline tests could never have caught it, because the fakes that exercised
the guard were written from the same assumption.

What follows from it:

- **A guard is a claim about the world, not a fact about the code.** "Rejected
  means the content is bad", "the model will stop searching when it has
  enough", "tool_choice='none' means no tool call": each is a claim a real
  model can falsify. All three were falsified in Part A.
- **Test guards against real model behaviour, not only against fakes.** A fake
  built from the guard's assumption can only confirm it.
- **When code and model disagree, look at which one was right before trusting
  either.** The log (`supervisor_log`: proposed vs. routed, override reason,
  rationale) is what made E2E01 diagnosable in minutes.
- **Separate facts from judgements, and give each to the side that is good at
  it.** The fixes that held all move a *mechanical* step into code
  (citation lookups, the Researcher's final call) and leave the model only the
  judgement. The fix that did not hold (`tool_choice="none"`) asked the model
  to obey a constraint.
- **Distinguish "could not finish" from "decided no".** The Critic now returns
  `incomplete` for the first. It is not an approval, and it no longer spends a
  revision or forces a rewrite.

The same pattern turned up four more times in Part A, in code as well as in
prompts:

| Assumption | Where | Falsified by |
| --- | --- | --- |
| every rejection is about the content | the 6.3 rewrite guard | E2E01: an unfinished review |
| the model stops searching when it has enough | the Researcher's loop | gpt-oss: 5 searches for one paper, every time |
| `tool_choice="none"` means no tool call | the reserved final call, first version | gpt-oss called the tool anyway |
| a provider counts `max_tokens` against its limits | my headroom probe | Groq: the probe passed with ~1k free |
| a request cannot outgrow the provider's ceiling | every agent, since Phase 6 | a 413 at 8,849 tokens |

And one where the fix was a *detector* but nothing acted on what it found:
E2E01's confirming run delivered a citation the notes did not contain,
after the Writer's check had flagged it and the Supervisor had read the flag.
A check that nobody downstream acts on turns silent bad output into logged
bad output. That is better, but it is not a fix.

This applies directly ahead: **Part D** adds retry and fallback rules, each of
which assumes something about why a call failed. **Part E**'s evaluators will
encode what "correct" means. A wrong evaluator does not fail loudly: it scores
confidently. Both need checking against real behaviour, the way this guard
finally was.

## Phase 7: Production (in progress)

Phase 7 is building "production" in a specific sense: *we now know whether
this works*, not just "it's wrapped in an API". Part A comes first because
every later part (streaming, retries, evaluation, deployment) would otherwise
be built on unverified claims.

### Part A: live verification

- **Provider switch.** `get_chat_model()` builds Anthropic (default) or Groq
  (`--provider groq`, or `RESEARCH_COPILOT_PROVIDER`). Structured output is
  provider-aware (`models.structured_output_kwargs`): `json_schema` on
  Anthropic and on Groq's gpt-oss/Qwen models (strict), tool calling on Groq's
  Llama models. Llama 3.3 70B turned out not to be on Groq's free tier; the
  pass ran on `openai/gpt-oss-120b`.
- **The harness** (`research-copilot live-check list | run | report`):
  - 35 labelled scenarios, with each expectation written before the model ran
  - pacing under the provider's per-minute limit, resumable runs, re-running
    of errored scenarios
  - Anthropic-only items recorded as PENDING, never inferred
  - a two-axis report: route, scored by code, beside rationale, judged against
    the state by a person
  - `to_dataset_example()` already emits the shape Part E will upload
- **Results:** item by item in the updated list above, and in
  `docs/live_check/SUMMARY-groq-gpt-oss-120b.md`.
- **Fixes that came out of it:**
  - `resilience.py`: one retry with a hint, then a degraded result
  - the Researcher's reserved final call, with no tools
  - the Supervisor's premise check
  - the Critic's code-side lookups and `incomplete` verdict
  - the Writer's unsupported-citation check
- **Request size is capped, per provider** (`request_budget.py`). Found by a
  413 on Groq, but the gap exists on every provider. Each agent sizes its
  request before sending it and cuts only its variable content (search
  results, notes - never the draft being judged). Every cut is logged in state
  (`researcher_trims`, `writer_trims`, `critic_trims`) and shown to the
  Supervisor. A provider refusal gets one retry at 70%.
- **Every model call is recorded in live runs** (`CallRecorder`): agent, node,
  estimated size, outcome, kept even when the run dies.
- **Provider limits are the provider's to report.** Groq's per-day limit
  refills continuously, is shared across everything using the key, and counted
  fewer tokens than our records (cached input likely excluded). The harness's
  budget caps a run; Groq's 429 is the only authority on what is left, and the
  runner now parses it into one line.

### Part A: closed

Every fix, and the live run that confirmed it (Groq `openai/gpt-oss-120b`):

| Finding | Fix | Confirmed by |
| --- | --- | --- |
| an invented tool call (`open_file`) killed the run | `resilience.py`: one hinted retry, then degrade | 10 Researcher calls with no crash (v2) |
| wrong-premise Supervisor rationales | "check the critique against the state" | SUP04/SUP08: wrong premise 2 → 0 (v2) |
| the Researcher never ended its loop | a reserved final call with **no tools**, results as text | RES01 passes (v4) |
| the Critic spent its budget on lookups; "did not finish" became a rejection | code-side lookups; an `incomplete` verdict | CRT 6/6 at ~1k tokens (v5); E2E01 approved in one pass (v6) |
| gpt-oss citation markup in answers | `unsupported_citations` check with one retry | unit-tested; fired live in E2E01 (v6) |
| one request outgrew the provider's ceiling (413) | per-request size limit, trims logged | E2E01: largest request ~6,000 under a 6,500 limit (v6) |

Still open, carried forward:

- **An unsupported citation reached the user** (E2E01 v6). Detected, not acted
  on: the Critic never sees the flag.
- **The Researcher restates what its notes already hold** instead of NOTHING
  NEW (RES02).
- **Scope creep, and gap-vs-critique priority**, in the Supervisor.
- **Single-sample variance** in item 2, for Part E's repeated-run scoring.
- **Item 1**, pending an Anthropic key.
- **The 413's exact node was never directly recorded.** That run predates the
  call recorder. The evidence points to the Researcher's tool loop: its
  requests grow with every search (404 → 2,539 → 3,636 → 5,741 → 6,010
  tokens in v6), they are the largest in the run, and the failed run's
  22.5k tokens spent match its 5th-6th call. Any future oversize request
  will be named directly.
