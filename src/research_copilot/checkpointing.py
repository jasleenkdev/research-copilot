"""Phase 4: making graph state outlive a single `.invoke()`.

CONCEPT: the checkpointer
Phases 1-3 lost everything the moment a run finished. `run_graph` built a state
dict, the nodes filled it in, `.invoke()` returned it, and Python freed it. That
is why Phase 2's `chat` command had to hold `ConversationMemory` in a `while`
loop: the only thing keeping the conversation alive was the process.

A checkpointer changes that. Pass one to `compile()`:

    graph = builder.compile(checkpointer=saver)

and LangGraph now writes a *checkpoint* after every super-step - a snapshot of
the whole State, plus which node runs next, plus any pending writes. On the next
`.invoke()` for the same thread, LangGraph loads the latest snapshot first and
merges your new input into it, instead of starting from `{}`.

Three capabilities fall out of that one argument, and all three are Phase 4:
  1. memory across calls  - `messages` is already there when the next turn starts
  2. pause and resume     - `interrupt()` can stop mid-run because the snapshot
                            says exactly where to pick up (see graph.py)
  3. time travel          - every past snapshot is still readable, so you can
                            inspect or re-run from any point in the thread

CONCEPT: a super-step, and why the checkpoint count is larger than you expect
LangGraph writes a checkpoint per *super-step*, not per turn. One turn of the
live-search loop (call_model -> call_tool -> call_model) is several super-steps,
and each writes a row. Four short turns in this project produce ~16 rows. That
matters twice: it is why the SQLite file grows faster than "one row per
question" suggests, and it is why deleting a message from the current state does
not delete it from the rows already written (see prune_history in graph.py).

CONCEPT: thread_id lives in `config`, not in `State`
This is the distinction worth internalizing, because getting it wrong is the
single most common Phase 4 mistake.

    graph.invoke(state_input, {"configurable": {"thread_id": "abc"}})
                 ^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                 what the run   which conversation this run belongs to
                 is about

`State` is the *content* of one conversation: the messages, the question, the
draft. `config` is *addressing metadata*: it tells the checkpointer which row to
load and which row to write. They are separate because they answer different
questions, and because a node must not be able to change which thread it is
writing to. A node returns partial State; it has no way to return a new
thread_id, and that is deliberate - a run cannot wander into another user's
conversation halfway through.

Practical consequences:
  - Nodes never see the thread_id in `state`. (A node that genuinely needs it
    declares a second parameter, `config: RunnableConfig`, and reads
    `config["configurable"]["thread_id"]` - the runtime passes it in.)
  - Two `.invoke()` calls with the same thread_id continue one conversation.
    Two with different thread_ids are two independent conversations sharing one
    graph and one checkpointer.
  - Forgetting the config entirely is not an error with `MemorySaver` in some
    versions and is an error in others; either way you get no persistence.
    `cli.py` therefore always resolves a thread_id explicitly and prints it.
  - The thread_id is *yours* to generate and reuse. LangGraph will not invent
    one for you, and there is no "current thread". `cli.py` shows both halves of
    this: `new_thread_id()` when you start, `--thread <id>` when you continue.

CONCEPT: MemorySaver vs SqliteSaver
Both implement the same `BaseCheckpointSaver` interface, so the graph code does
not change when you swap them - exactly like Phase 2's `--memory trim` vs
`--memory summarize` switch, where the policy changed and the chat loop did not.

    MemorySaver   a dict in the process. Survives across `.invoke()` calls,
                  dies with the process. Right for tests (no files to clean up,
                  no schema, instant) and for a single interactive session.
    SqliteSaver   rows in a SQLite file under data/. Survives across processes,
                  so two separate `research-copilot` invocations can continue
                  one conversation. Right for local development and for
                  actually *seeing* what a checkpoint is (`sqlite3 data/
                  checkpoints.sqlite3 ".tables"`).

In production the same slot takes `PostgresSaver`. The lesson is that
persistence is a compile-time argument, not a rewrite.

CONCEPT: why this module hands back a context manager
`SqliteSaver` wraps a live `sqlite3.Connection`. The connection has to stay open
for as long as the graph might be invoked, and be closed afterwards - which is
awkward to express as a plain factory returning a saver. So the public entry
point is a context manager:

    with checkpointer_scope("sqlite") as saver:
        graph = build_graph(checkpointer=saver)
        ...                      # every invoke happens inside the `with`

LangGraph's own `SqliteSaver.from_conn_string` is also a context manager, for
the same reason. This wrapper adds the "none" and "memory" cases so callers have
one shape to code against, and points the file at `data/` via config.py.
"""

import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from research_copilot.config import get_settings

# "none" is kept as a real option, not an oversight: it is Phase 3's behaviour,
# and being able to turn persistence off is how you see what it was doing.
CheckpointerKind = Literal["none", "memory", "sqlite"]

KINDS: tuple[str, ...] = ("none", "memory", "sqlite")


def new_thread_id() -> str:
    """Mint a thread_id for a new conversation.

    A thread_id is an opaque string - LangGraph never parses it. A UUID is the
    safe default because nothing else in the process has to coordinate to avoid
    collisions. In a real application this would usually be a value you already
    have: a chat session row id, a ticket number, `f"user:{user_id}"`.

    Note what this function is *not*: it is not "get the current thread". There
    is no such thing. Reusing a thread means passing its id back in, which is
    why `cli.py` prints the id on every run.
    """
    return str(uuid.uuid4())


def thread_config(thread_id: str) -> dict:
    """Build the `config` argument that addresses one conversation.

    The nesting under "configurable" is not decoration: `config` also carries
    runtime knobs LangGraph owns (`recursion_limit`, `callbacks`, `tags`), and
    "configurable" is the sub-dict reserved for values that get threaded through
    to the checkpointer and to nodes. `thread_id` is the only key required for
    persistence; `checkpoint_id` can be added to address one *specific* past
    snapshot instead of the latest, which is how time travel is expressed.
    """
    return {"configurable": {"thread_id": thread_id}}


def resolve_checkpoint_db(db_path: str | Path | None = None) -> Path:
    """Where the SQLite checkpoint file lives.

    Under `data/` beside Phase 2's Chroma store, because `data/` is already
    gitignored - a checkpoint database contains whole conversations, which is
    exactly the kind of thing that should not reach a commit by accident.
    """
    settings = get_settings()
    path = Path(db_path) if db_path is not None else settings.checkpoint_db
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def checkpointer_scope(
    kind: CheckpointerKind | None = None,
    *,
    db_path: str | Path | None = None,
) -> Iterator[BaseCheckpointSaver | None]:
    """Yield the configured checkpointer, closing it afterwards.

    `kind=None` reads RESEARCH_COPILOT_CHECKPOINTER from .env, the same
    settings-with-CLI-override pattern as `memory_from_settings`.

    Yields `None` for "none", and `build_graph(checkpointer=None)` is Phase 3's
    graph exactly - so the flag genuinely switches persistence on and off rather
    than swapping in a no-op saver that still writes.
    """
    resolved = (kind or get_settings().checkpointer).lower()
    if resolved not in KINDS:
        raise RuntimeError(
            f"Unknown checkpointer {resolved!r}; expected one of {', '.join(KINDS)}."
        )

    if resolved == "none":
        yield None
        return

    if resolved == "memory":
        # Nothing to close: the store is a dict that goes away with the process.
        yield MemorySaver()
        return

    path = resolve_checkpoint_db(db_path)
    # check_same_thread=False because LangGraph may touch the connection from a
    # worker thread; SqliteSaver serializes access behind its own lock, which is
    # what makes that safe. (It is also why SqliteSaver is documented as being
    # for small local projects - one lock, one writer.)
    connection = sqlite3.connect(str(path), check_same_thread=False)
    try:
        saver = SqliteSaver(connection)
        # Creates the checkpoints/writes tables on first use. SqliteSaver calls
        # this itself before its first query; doing it up front means a broken
        # path or an unwritable directory fails now, with a clear traceback,
        # rather than midway through a run that has already called the model.
        saver.setup()
        yield saver
    finally:
        connection.close()


def describe_checkpointer(saver: BaseCheckpointSaver | None) -> str:
    """One line for the CLI banner, so which saver is in use is never a guess."""
    if saver is None:
        return "none (state is discarded when the process exits)"
    if isinstance(saver, SqliteSaver):
        return f"sqlite ({resolve_checkpoint_db()})"
    return "memory (state is discarded when the process exits)"


def list_thread_ids(saver: BaseCheckpointSaver) -> list[str]:
    """Every thread_id the checkpointer has ever written, newest activity first.

    `saver.list(config)` walks checkpoints. Passing `None` instead of a config
    means "every thread", which is how you find a thread_id you forgot to write
    down. Each thread has many checkpoints (one per super-step), so this
    de-duplicates while preserving the newest-first order `list` returns.
    """
    seen: dict[str, None] = {}
    for checkpoint in saver.list(None):
        thread_id = checkpoint.config.get("configurable", {}).get("thread_id")
        if thread_id is not None:
            seen.setdefault(thread_id, None)
    return list(seen)


def count_checkpoints(saver: BaseCheckpointSaver, thread_id: str) -> int:
    """How many snapshots one conversation has accumulated.

    Printed by `graph-threads` because it is the number that makes the cost of
    persistence concrete: it grows per super-step, not per question, and nothing
    prunes it. See the note on `prune_history` in graph.py - removing a message
    from the *current* state does not shrink the rows already on disk.
    """
    return sum(1 for _ in saver.list(thread_config(thread_id)))
