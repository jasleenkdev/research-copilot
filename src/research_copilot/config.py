"""Configuration: load .env once and expose typed settings.

Why a config module at all? LangChain, LangSmith, and the Anthropic SDK read
their settings straight from environment variables (ANTHROPIC_API_KEY,
LANGSMITH_TRACING, LANGSMITH_API_KEY, ...). If .env is loaded in one place before
any model is built, auth and tracing work everywhere else with no keys passed
around. Enabling LangSmith tracing needs no code at all; the env vars are enough.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# override=False: a variable already exported in your shell wins over .env.
# (The tests rely on this to force tracing off.)
load_dotenv(override=False)

DEFAULT_MODEL = "claude-opus-5"

# Anchor file paths to the repo, not the shell's current directory, so the
# vector store is the same one wherever you run the CLI from.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    model: str
    tracing_enabled: bool
    langsmith_api_key_set: bool
    langsmith_project: str | None
    # Phase 2: conversation memory
    memory_strategy: str
    max_history_tokens: int
    # Phase 2: retrieval
    embedding_model: str
    chroma_dir: Path
    collection_name: str
    retrieval_k: int
    # Phase 4: persistence and human-in-the-loop
    checkpointer: str
    checkpoint_db: Path
    require_approval: bool


def get_settings() -> Settings:
    return Settings(
        model=os.getenv("RESEARCH_COPILOT_MODEL") or DEFAULT_MODEL,
        tracing_enabled=os.getenv("LANGSMITH_TRACING", "").lower() == "true",
        langsmith_api_key_set=bool(os.getenv("LANGSMITH_API_KEY")),
        langsmith_project=os.getenv("LANGSMITH_PROJECT"),
        # "trim" drops old turns; "summarize" folds them into a summary message.
        memory_strategy=(
            os.getenv("RESEARCH_COPILOT_MEMORY_STRATEGY") or "trim"
        ).lower(),
        # Deliberately small so pruning is easy to trigger and watch.
        max_history_tokens=int(os.getenv("RESEARCH_COPILOT_MAX_HISTORY_TOKENS") or 1200),
        embedding_model=(
            os.getenv("RESEARCH_COPILOT_EMBEDDING_MODEL")
            or "sentence-transformers/all-MiniLM-L6-v2"
        ),
        chroma_dir=Path(
            os.getenv("RESEARCH_COPILOT_CHROMA_DIR") or PROJECT_ROOT / "data" / "chroma"
        ),
        collection_name=os.getenv("RESEARCH_COPILOT_COLLECTION") or "research_copilot",
        retrieval_k=int(os.getenv("RESEARCH_COPILOT_RETRIEVAL_K") or 4),
        # "none" reproduces Phase 3 (state dies with the run), "memory" persists
        # for the life of the process, "sqlite" persists to disk. The default is
        # sqlite because the interesting Phase 4 behaviour - a conversation that
        # survives a separate CLI invocation - is only visible on disk.
        checkpointer=(os.getenv("RESEARCH_COPILOT_CHECKPOINTER") or "sqlite").lower(),
        checkpoint_db=Path(
            os.getenv("RESEARCH_COPILOT_CHECKPOINT_DB")
            or PROJECT_ROOT / "data" / "checkpoints.sqlite3"
        ),
        # Whether the graph pauses for human approval before an answer is
        # committed to the transcript. Off by default so Phases 1-3's commands
        # behave as they always did; `--approve` and `graph-chat` turn it on.
        require_approval=(
            os.getenv("RESEARCH_COPILOT_REQUIRE_APPROVAL", "").lower() == "true"
        ),
    )


def require_anthropic_key() -> None:
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Add it to .env (see .env.example)."
        )
