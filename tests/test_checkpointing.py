"""Phase 4 Part A: persistence. Offline - fake model, no API key, no network.

The point these tests prove is the one that is hard to see by reading: a
checkpointer turns `.invoke()` from "run and forget" into "continue a
conversation", and the thing that decides *which* conversation is the config,
not the state.
"""

import sqlite3

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage

from research_copilot.checkpointing import (
    checkpointer_scope,
    count_checkpoints,
    list_thread_ids,
    new_thread_id,
    thread_config,
)
from research_copilot.graph import build_graph, final_answer, run_graph


class FakeToolCallingModel(FakeMessagesListChatModel):
    """Same harness as tests/test_graph.py - scripted AIMessages, no real model."""

    def bind_tools(self, tools, **kwargs):
        return self


def scripted(*texts: str) -> FakeToolCallingModel:
    return FakeToolCallingModel(responses=[AIMessage(content=t) for t in texts])


# --- the core claim -----------------------------------------------------------


def test_without_a_checkpointer_each_run_starts_from_an_empty_transcript():
    """Phase 3's behaviour, kept as the baseline the rest of this file contrasts."""
    graph = build_graph(model=scripted("A1", "A2"), memory_strategy="none")

    first = run_graph("Q1", graph=graph)
    second = run_graph("Q2", graph=graph)

    assert [m.text for m in first["messages"]] == ["Q1", "A1"]
    # Q1 is gone. Nothing carried over, because nothing was stored.
    assert [m.text for m in second["messages"]] == ["Q2", "A2"]


def test_the_same_thread_id_accumulates_messages_across_invocations():
    with checkpointer_scope("memory") as saver:
        graph = build_graph(
            model=scripted("A1", "A2", "A3"),
            checkpointer=saver,
            memory_strategy="none",
        )
        thread = new_thread_id()

        run_graph("Q1", graph=graph, thread_id=thread)
        run_graph("Q2", graph=graph, thread_id=thread)
        state = run_graph("Q3", graph=graph, thread_id=thread)

        # Three separate .invoke() calls, one transcript.
        assert [m.text for m in state["messages"]] == [
            "Q1", "A1", "Q2", "A2", "Q3", "A3",
        ]


def test_different_thread_ids_are_independent_conversations():
    """One graph, one checkpointer, two conversations that cannot see each other."""
    with checkpointer_scope("memory") as saver:
        graph = build_graph(
            model=scripted("A1", "B1", "A2"),
            checkpointer=saver,
            memory_strategy="none",
        )

        run_graph("Q1", graph=graph, thread_id="alice")
        run_graph("Q1", graph=graph, thread_id="bob")
        alice = run_graph("Q2", graph=graph, thread_id="alice")

        assert [m.text for m in alice["messages"]] == ["Q1", "A1", "Q2", "A2"]
        bob = graph.get_state(thread_config("bob")).values
        assert [m.text for m in bob["messages"]] == ["Q1", "B1"]


def test_thread_id_is_config_not_state():
    """The distinction, asserted rather than described.

    Putting thread_id in the state dict does nothing: it is not a declared State
    key, so no node reads it, and the checkpointer never sees it. The run is
    anonymous and nothing accumulates.
    """
    with checkpointer_scope("memory") as saver:
        graph = build_graph(
            model=scripted("A1", "A2"), checkpointer=saver, memory_strategy="none"
        )

        graph.invoke(
            {
                "question": "Q1",
                "mode": "live-search",
                "messages": [HumanMessage(content="Q1")],
                "thread_id": "wishful-thinking",  # not a State key; ignored
            },
            thread_config("real-thread"),
        )

        # The turn was stored under the config's thread_id, not the state's.
        assert graph.get_state(thread_config("real-thread")).values["messages"]
        assert graph.get_state(thread_config("wishful-thinking")).values == {}


# --- the counter that persistence breaks --------------------------------------


def test_iterations_resets_each_turn_despite_being_checkpointed():
    """`iterations` is a per-turn budget living in per-thread storage.

    Without the reset in `turn_input`, turn 2 would start at 1 and turn 6 would
    start at 5 - and `should_continue` would refuse tool calls for work earlier
    turns did. This is the bug a checkpointer silently introduces.
    """
    with checkpointer_scope("memory") as saver:
        graph = build_graph(
            model=scripted("A1", "A2", "A3"),
            checkpointer=saver,
            max_iterations=2,
            memory_strategy="none",
        )
        thread = new_thread_id()

        for _ in range(3):
            state = run_graph("Q", graph=graph, thread_id=thread)
            # One call_model per turn, every turn - never 1, 2, 3.
            assert state["iterations"] == 1

        # And the transcript still grew, so the reset did not wipe the thread.
        assert len(state["messages"]) == 6


# --- SqliteSaver: the part that survives a process --------------------------


def test_sqlite_checkpoints_survive_a_new_graph_and_a_new_connection(tmp_path):
    """The closest offline stand-in for "run the CLI twice".

    Each `with` block opens its own connection and compiles its own graph -
    nothing is shared but the file on disk, which is exactly what two separate
    CLI invocations share.
    """
    db = tmp_path / "checkpoints.sqlite3"
    thread = "cross-process"

    with checkpointer_scope("sqlite", db_path=db) as saver:
        graph = build_graph(
            model=scripted("A1"), checkpointer=saver, memory_strategy="none"
        )
        run_graph("Q1", graph=graph, thread_id=thread)

    assert db.exists()

    with checkpointer_scope("sqlite", db_path=db) as saver:
        graph = build_graph(
            model=scripted("A2"), checkpointer=saver, memory_strategy="none"
        )
        state = run_graph("Q2", graph=graph, thread_id=thread)

    assert [m.text for m in state["messages"]] == ["Q1", "A1", "Q2", "A2"]


def test_sqlite_writes_one_row_per_super_step_not_per_turn(tmp_path):
    """What "unbounded on disk" actually looks like.

    A turn is several super-steps, and every one writes a checkpoint. Nothing
    prunes these rows - `prune_history` shortens the *current* state, it does not
    rewrite history (see test_pruning.py).
    """
    db = tmp_path / "checkpoints.sqlite3"
    with checkpointer_scope("sqlite", db_path=db) as saver:
        graph = build_graph(
            model=scripted("A1", "A2"), checkpointer=saver, memory_strategy="none"
        )
        run_graph("Q1", graph=graph, thread_id="t")
        after_one = count_checkpoints(saver, "t")
        run_graph("Q2", graph=graph, thread_id="t")
        after_two = count_checkpoints(saver, "t")

        assert after_one > 1, "a single turn already writes several checkpoints"
        assert after_two > after_one
        assert list_thread_ids(saver) == ["t"]


def test_reading_a_thread_id_that_does_not_exist_is_not_an_error():
    """A missing thread reads as empty, not as a failure.

    This is worth knowing before you rely on it: a typo'd --thread does not
    raise, it silently starts a new conversation. The CLI checks
    `snapshot.values` to tell the two apart.
    """
    with checkpointer_scope("memory") as saver:
        graph = build_graph(
            model=scripted("A1"), checkpointer=saver, memory_strategy="none"
        )
        snapshot = graph.get_state(thread_config("never-used"))

        assert snapshot.values == {}
        assert snapshot.next == ()


def test_unknown_checkpointer_kind_is_rejected():
    with pytest.raises(RuntimeError, match="Unknown checkpointer"):
        with checkpointer_scope("redis"):
            pass


def test_memory_saver_state_does_not_outlive_its_scope():
    """MemorySaver is a dict: leaving the scope is the whole lifecycle."""
    with checkpointer_scope("memory") as first:
        graph = build_graph(
            model=scripted("A1"), checkpointer=first, memory_strategy="none"
        )
        run_graph("Q1", graph=graph, thread_id="t")
        assert graph.get_state(thread_config("t")).values["messages"]

    with checkpointer_scope("memory") as second:
        graph = build_graph(
            model=scripted("A2"), checkpointer=second, memory_strategy="none"
        )
        # Same thread_id, new saver, nothing there.
        assert graph.get_state(thread_config("t")).values == {}


def test_swapping_checkpointers_mid_conversation_loses_the_thread(tmp_path):
    """MemorySaver -> SqliteSaver is not a migration; it is a fresh start.

    Checkpointers do not share storage, and a thread_id means nothing outside
    the saver that wrote it. The run does not fail - which is exactly the danger.
    It quietly answers with no history.
    """
    thread = "swapped"
    with checkpointer_scope("memory") as saver:
        graph = build_graph(
            model=scripted("A1"), checkpointer=saver, memory_strategy="none"
        )
        run_graph("Q1", graph=graph, thread_id=thread)

    with checkpointer_scope("sqlite", db_path=tmp_path / "cp.sqlite3") as saver:
        graph = build_graph(
            model=scripted("A2"), checkpointer=saver, memory_strategy="none"
        )
        state = run_graph("Q2", graph=graph, thread_id=thread)

    assert [m.text for m in state["messages"]] == ["Q2", "A2"]


def test_state_written_before_a_new_key_existed_still_loads(tmp_path):
    """Adding a State key is backwards compatible; every node uses .get().

    Simulated by writing a thread with a graph that never sets `summary` or
    `status`, then reading it back and continuing. The absent keys read as
    missing, and the defaults carry the run.
    """
    db = tmp_path / "cp.sqlite3"
    with checkpointer_scope("sqlite", db_path=db) as saver:
        graph = build_graph(
            model=scripted("A1"), checkpointer=saver, memory_strategy="none"
        )
        run_graph("Q1", graph=graph, thread_id="old")
        stored = graph.get_state(thread_config("old")).values
        # `summary` is never written by a "none"-strategy run, so this snapshot
        # genuinely lacks the key - the same shape a thread written before the
        # key existed would have. (`draft` *is* present: turn_input resets it
        # every turn, which is itself the reason a missing key is harmless.)
        assert "summary" not in stored

    with checkpointer_scope("sqlite", db_path=db) as saver:
        graph = build_graph(
            model=scripted("A2"),
            checkpointer=saver,
            memory_strategy="summarize",
            max_history_tokens=0,
        )
        state = run_graph("Q2", graph=graph, thread_id="old")

    assert final_answer(state) == "A2"


def test_sqlite_file_is_readable_as_an_ordinary_database(tmp_path):
    """A checkpoint is not a black box - it is rows you can go and look at."""
    db = tmp_path / "cp.sqlite3"
    with checkpointer_scope("sqlite", db_path=db) as saver:
        graph = build_graph(
            model=scripted("A1"), checkpointer=saver, memory_strategy="none"
        )
        run_graph("Q1", graph=graph, thread_id="t")

    connection = sqlite3.connect(db)
    tables = {
        row[0]
        for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    connection.close()
    assert "checkpoints" in tables
