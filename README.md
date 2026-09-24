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
- [ ] **Phase 6: Multi-agent.** Researcher, Writer, and Critic nodes coordinated
  by a Supervisor that routes on structured LLM output. Each agent has its own
  tools and system prompt. Built in four reviewed steps:
  - [x] 6.1 Researcher (subgraph, private tool loop) → Writer (node), fixed
    hand-off, one owner per state field
  - [x] 6.2 Supervisor routing via structured output, with code guards,
    a decision log, and the Researcher's brief/outcome/merge-on-rerun contract
  - [ ] 6.3 Critic agent with its own tools; per-agent budgets
  - [ ] 6.4 CLI flags (`--critic`/`--approve`/`--plan`), threads, Studio
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
  cli.py           command-line entry point
langgraph.json     tells LangGraph Studio where the graphs are (Phase 3)
data/chroma/       the local vector store (gitignored, created by `ingest`)
data/checkpoints.sqlite3  the checkpoint database (gitignored, Phase 4)
tests/             offline tests using fake models
                   test_reflection.py / test_planning.py are Phase 5
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

## Usage (Phase 6.1)

Phase 5's `call_model` did two jobs: it looked things up, and it wrote the
answer. 6.1 gives each job to its own agent, in a new graph
(`multi_agent_graph.py`) beside the old one. `graph-agent` still runs Phase 5's
graph, unchanged.

```bash
research-copilot multi-agent "How are RAG pipelines evaluated?"
research-copilot multi-agent "What do my notes say about chunking?" --mode knowledge-base
research-copilot multi-agent "Compare RAG and long-context models on cost" --plan
research-copilot multi-agent "..." --max-research-iterations 2   # watch the budget fallback
research-copilot draw-graph multi     # xray view: the Researcher's inside is drawn
```

The command prints every hand-off as it happens. Steps inside the Researcher's
subgraph are indented:

```
--- hand-offs ---
  [plan_question] (no change)
    [researcher/research_model] research_messages=[1], research_iterations=1
    [researcher/research_tools] research_messages=[1]
    [researcher/research_model] research_messages=[1], research_iterations=2
    [researcher/compile_notes] research_notes='Findings: ...'
  [researcher] research_notes='Findings: ...', research_iterations=2
  [writer] draft='...'
  [finalize_answer] messages=[1]
```

Notice that `research_messages` appears only on the indented lines. The final
state's `messages` holds exactly one question and one answer.

### The shape

```
START → plan_question → [researcher subgraph] → writer → finalize_answer → END
                          ├ knowledge-base: retrieve
                          └ live-search:    research_model ⇄ research_tools → compile_notes
```

Every edge is fixed. 6.2 replaces the middle ones with a Supervisor.

### One field, one owner

| Field | Owner | Enforced by |
| --- | --- | --- |
| `sub_questions` | `plan_question` | `owns()` wrapper |
| `research_notes`, `documents`, `research_iterations` | `researcher` | the subgraph's `output_schema` |
| `draft` | `writer` | `owns()` wrapper |
| `messages` | `finalize_answer` | `owns()` wrapper |
| everything above, between turns | the turn boundary (`multi_agent_turn_input`) | — |

A node that returns a key it does not own raises `OwnershipError`.

### Subgraph vs. node

The Researcher is a subgraph because it has a loop (search → read → search) and
scratch work nobody else should read. The Writer is a plain node because it
makes one model call with no tools. The rule: **promote an agent to a subgraph
when it has a loop or state of its own, not because it is "an agent".**

The Researcher's tool loop goes to a private `research_messages` channel. It is
kept out of the shared transcript, out of the Writer's input, and out of
`get_state()`. It is **not** kept off the disk: the subgraph checkpoints under
its own namespace (`researcher:<task-id>`), and those checkpoints hold every
search result.

## Usage (Phase 6.2)

The fixed line is now a hub. Every agent reports back to a Supervisor, which
decides who acts next from a structured model reply:

```
START → plan_question → supervisor ⇄ researcher
                            ⇅
                          writer
                            ↓ finish
                     finalize_answer → END
```

```bash
research-copilot multi-agent "How are RAG pipelines evaluated?"            # a model routes
research-copilot multi-agent "..." --routing fixed                         # 6.1's path, no model
research-copilot multi-agent "..." --max-researcher-runs 3 --max-writer-runs 1
```

Each decision is printed with its rationale. Any override is printed beside
what the model proposed:

```
  [supervisor] -> researcher  brief='evidence RAGAS agrees with human judgement'
      why: Draft makes no claim about validation; notes list that as a gap.
  ...
  [supervisor] -> writer  (proposed finish; OVERRIDDEN: finish proposed with a draft that predates the latest research)
      why: Looks complete.
```

### What the Supervisor reads, and what it returns

| Reads (`render_supervisor_view`) | Returns (`SupervisorDecision`) |
| --- | --- |
| question, sub-questions | `rationale`: written *first*, so the route is conditioned on it |
| research notes (excerpted) and `research_outcome` | `next`: `researcher` / `writer` / `finish` |
| draft, marked STALE if it predates the latest notes | `researcher_brief`: what a follow-up pass should find |
| dispatches used / cap per agent | |
| its own earlier decisions this turn | |

The reply uses `with_structured_output(..., method="json_schema")`, the
Anthropic API's native structured output. The default `function_calling`
method forces a tool call, which the API rejects when thinking is on.

### Code guards have the last word

| Guard | Proposal | Routed to |
| --- | --- | --- |
| agent's dispatch cap reached | that agent | `fixed_policy`'s choice |
| no research yet | `writer` | `researcher` |
| no draft, or a stale one | `finish` | `writer` (if budget remains) |
| model output raises or fails validation | — | `fixed_policy` (6.1's hand-off) |

Every route except `finish` spends a capped dispatch, so a turn makes at most
`sum(caps) + 1` decisions whatever the model says.
`tests/test_supervisor.py::test_no_supervisor_can_loop_forever` checks this
against Supervisors that always research, always write, always finish, or
ping-pong.

### The Researcher's 6.2 contract

- **`researcher_brief`** (the Supervisor writes it, the Researcher reads it):
  what this pass should look for. It is reset on every research dispatch, so an
  old brief never steers a new pass.
- **`research_outcome`** (the Researcher writes it): `findings` /
  `nothing_found` / `budget_exhausted` for the *latest* pass. The guards branch
  on this, not on text inside the notes.
- **Merge-on-rerun**: a follow-up pass appends under a code-written header, and
  never rewrites. Knowledge-base excerpts keep their `[n]` numbers. A pass that
  finds nothing leaves the notes unchanged.

## Where each Phase 6 concept lives

| Concept | File |
| --- | --- |
| Why one `draft` field breaks with several agents | `multi_agent_state.py` (module docstring) |
| One field, one owner, and how it is enforced | `multi_agent_state.py` → `OWNERS`, `owns` |
| The turn boundary as the one non-owner writer | `multi_agent_state.py`; `multi_agent_graph.py` → `multi_agent_turn_input` |
| `input_schema` / `output_schema` as an agent's contract | `multi_agent_state.py` → `ResearcherInput`, `ResearcherOutput` |
| An agent as a subgraph | `agents/researcher.py` (module docstring) |
| The private message channel, and its limit (disk) | `agents/researcher.py`; `tests/test_multi_agent.py` |
| Why the Writer is a node, not a subgraph | `agents/writer.py` (module docstring) |
| An agent must always hand something over | `agents/researcher.py` → `compile_notes` |
| Telling the Writer "nothing was found" explicitly | `agents/writer.py` → `NO_NOTES` |
| Which stream modes see inside a subgraph | `cli.py` → `cmd_multi_agent` |
| Per-invocation vs. per-revision agent budgets (for 6.3) | `multi_agent_graph.py` (bottom note) |
| Why routing is a model's judgement now | `agents/supervisor.py` (module docstring) |
| Structured output, `json_schema` vs `function_calling` | `agents/supervisor.py` (module docstring), `make_supervisor` |
| Rationale-before-route field order | `agents/supervisor.py` → `SupervisorDecision` |
| What the Supervisor sees | `agents/supervisor.py` → `render_supervisor_view` |
| Code guards overruling the model; the termination bound | `agents/supervisor.py` → `apply_guards`; `tests/test_supervisor.py` |
| Falling back to the known-good hand-off | `agents/supervisor.py` → `fixed_policy` |
| Stale draft, answered from the decision log | `agents/supervisor.py` → `draft_is_current` |
| Decision in a node, edge in a routing function (vs `Command`) | `multi_agent_graph.py` → `route_from_supervisor` |
| Hub-and-spoke vs agents routing each other | `multi_agent_graph.py` (6.2 docstring) |
| Why the decision log does not use `operator.add` | `multi_agent_state.py` → `supervisor_log` |
| Briefs and merge-on-rerun | `agents/researcher.py` (6.2 docstring), `compile_notes`, `retrieve` |
