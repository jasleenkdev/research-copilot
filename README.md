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
- [ ] **Phase 3: First LangGraph.** Rebuild Phase 1's tool loop as a `StateGraph`
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
  cli.py           command-line entry point
data/chroma/       the local vector store (gitignored, created by `ingest`)
tests/             offline tests using fake models
```

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
