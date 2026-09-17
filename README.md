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
- [ ] **Phase 4: State design & persistence.** Richer state (`question`,
  `research_notes`, `draft`, `critique`, `iteration_count`), checkpointers
  (`MemorySaver`, then `SqliteSaver`), and `interrupt()` for human approval before
  an answer is finalized.
- [ ] **Phase 5: Multi-step reasoning.** A reflection loop (draft → critique →
  revise, looping back until a quality threshold or max iterations) and a
  planning node that splits the question into sub-questions before research.
- [ ] **Phase 6: Multi-agent.** Researcher, Writer, and Critic nodes coordinated
  by a Supervisor that routes on structured LLM output. Each agent has its own
  tools and system prompt.
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
  state.py         the graph's State TypedDict + reducer notes (Phase 3)
  graph.py         the hand-rolled StateGraph: nodes, edges, routing (Phase 3)
  prebuilt.py      the same agent via create_react_agent, for comparison (Phase 3)
  cli.py           command-line entry point
langgraph.json     tells LangGraph Studio where the graphs are (Phase 3)
data/chroma/       the local vector store (gitignored, created by `ingest`)
tests/             offline tests using fake models
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
