"""Phase 4 at the CLI level: separate `main()` calls sharing one thread.

The unit tests in test_checkpointing.py reuse one compiled graph object. These
go a step further and drive `cli.main()` itself, once per turn, the way you
would from a shell - a new graph, a new SQLite connection, and nothing carried
in memory between calls. The only thing linking them is `--thread` and the file
in data/.

The model is faked by patching the factory `graph.py` calls, so no API key is
needed and no tokens are spent.
"""

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from research_copilot import cli


class FakeToolCallingModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


@pytest.fixture
def fake_cli(monkeypatch, tmp_path):
    """Point the CLI at a throwaway checkpoint DB and a scripted model.

    `graph.py` does `from research_copilot.models import get_chat_model`, which
    binds the name into that module - so the patch has to target
    `research_copilot.graph.get_chat_model`, not models.get_chat_model.
    """
    monkeypatch.setenv(
        "RESEARCH_COPILOT_CHECKPOINT_DB", str(tmp_path / "checkpoints.sqlite3")
    )
    monkeypatch.setenv("LANGSMITH_TRACING", "false")

    def install(*texts: str) -> None:
        model = FakeToolCallingModel(
            responses=[AIMessage(content=t) for t in texts]
        )
        monkeypatch.setattr(
            "research_copilot.graph.get_chat_model", lambda **kwargs: model
        )

    return install


def test_two_separate_cli_invocations_continue_one_conversation(fake_cli, capsys):
    """The Part A proof: `state.messages` survives between processes.

    Each `cli.main([...])` here is what one `research-copilot ...` shell command
    does. The second one answers with the first one's turn already in state.
    """
    fake_cli("first answer", "second answer")
    thread = "cli-thread"

    assert cli.main(
        ["graph-agent", "Q1", "--thread", thread, "--checkpointer", "sqlite",
         "--memory", "none"]
    ) == 0
    capsys.readouterr()

    assert cli.main(
        ["graph-agent", "Q2", "--thread", thread, "--checkpointer", "sqlite",
         "--memory", "none"]
    ) == 0
    captured = capsys.readouterr()

    assert "second answer" in captured.out
    # The state dump goes to stderr and shows all four turns: the second run
    # loaded Q1 and its answer from disk.
    assert "messages:   4" in captured.err
    assert "Q1" in captured.err


def test_omitting_the_thread_starts_a_fresh_conversation(fake_cli, capsys):
    """No thread_id means no continuity - and the CLI prints the generated id
    precisely so the next call can pass it back."""
    fake_cli("answer one", "answer two")

    cli.main(["graph-agent", "Q1", "--checkpointer", "sqlite", "--memory", "none"])
    first = capsys.readouterr().err

    cli.main(["graph-agent", "Q2", "--checkpointer", "sqlite", "--memory", "none"])
    second = capsys.readouterr().err

    assert "(new)" in first and "(new)" in second
    assert "messages:   2" in second, "a new thread starts empty"
    assert "continue this conversation with: --thread" in first


def test_checkpointer_none_reproduces_phase_3(fake_cli, capsys):
    fake_cli("a1", "a2")
    thread = "ignored"

    cli.main(["graph-agent", "Q1", "--thread", thread, "--checkpointer", "none"])
    capsys.readouterr()
    cli.main(["graph-agent", "Q2", "--thread", thread, "--checkpointer", "none"])
    captured = capsys.readouterr()

    assert "messages:   2" in captured.err, "nothing was stored to carry over"


def test_pause_in_one_invocation_and_approve_in_the_next(fake_cli, capsys):
    """The Part C cycle, driven entirely through the CLI.

    `graph-agent --approve` parks the run and exits. `review --thread` is a
    separate command that finds the parked draft from the thread_id alone.
    """
    fake_cli("a draft needing approval")
    thread = "hitl-thread"

    cli.main(
        ["graph-agent", "Q", "--thread", thread, "--checkpointer", "sqlite",
         "--approve", "--memory", "none"]
    )
    paused = capsys.readouterr()

    assert "awaiting approval" in paused.out
    assert "a draft needing approval" in paused.out
    assert f"review --thread {thread}" in paused.err

    # A different command, a new connection, a new graph.
    cli.main(["review", "--thread", thread, "--checkpointer", "sqlite", "--approve"])
    resumed = capsys.readouterr()

    assert "a draft needing approval" in resumed.out
    assert "status:     approved" in resumed.err


def test_reviewing_an_edit_from_the_command_line(fake_cli, capsys):
    fake_cli("model wording")
    thread = "edit-thread"

    cli.main(
        ["graph-agent", "Q", "--thread", thread, "--checkpointer", "sqlite",
         "--approve", "--memory", "none"]
    )
    capsys.readouterr()

    cli.main(
        ["review", "--thread", thread, "--checkpointer", "sqlite",
         "--edit", "human wording", "--note", "tightened"]
    )
    captured = capsys.readouterr()

    assert "human wording" in captured.out
    assert "feedback:   tightened" in captured.err


def test_reviewing_a_rejection_from_the_command_line(fake_cli, capsys):
    fake_cli("speculative draft")
    thread = "reject-thread"

    cli.main(
        ["graph-agent", "Q", "--thread", thread, "--checkpointer", "sqlite",
         "--approve", "--memory", "none"]
    )
    capsys.readouterr()

    cli.main(
        ["review", "--thread", thread, "--checkpointer", "sqlite",
         "--reject", "--note", "no sources"]
    )
    captured = capsys.readouterr()

    assert "withheld" in captured.out
    assert "status:     rejected" in captured.err


def test_reviewing_a_thread_with_nothing_pending_says_so(fake_cli, capsys):
    """An idle thread and a missing thread get different messages on purpose."""
    fake_cli("done")
    thread = "idle-thread"

    cli.main(
        ["graph-agent", "Q", "--thread", thread, "--checkpointer", "sqlite",
         "--memory", "none"]
    )
    capsys.readouterr()

    cli.main(["review", "--thread", thread, "--checkpointer", "sqlite", "--approve"])
    captured = capsys.readouterr()

    assert "nothing is awaiting approval" in captured.err
    assert "has no saved state" not in captured.err


def test_reviewing_an_unknown_thread_reports_a_missing_thread(fake_cli, capsys):
    fake_cli("unused")
    cli.main(["review", "--thread", "never-existed", "--checkpointer", "sqlite",
              "--approve"])
    captured = capsys.readouterr()

    assert "has no saved state" in captured.err


def test_threads_lists_what_is_on_disk(fake_cli, capsys):
    fake_cli("a1", "a2")

    cli.main(["graph-agent", "Q", "--thread", "alpha", "--checkpointer", "sqlite",
              "--memory", "none"])
    cli.main(["graph-agent", "Q", "--thread", "beta", "--checkpointer", "sqlite",
              "--memory", "none"])
    capsys.readouterr()

    cli.main(["threads", "--checkpointer", "sqlite"])
    captured = capsys.readouterr()

    assert "alpha" in captured.out and "beta" in captured.out
    assert "checkpoints" in captured.out


def test_approve_without_a_checkpointer_is_refused_before_any_model_call(
    fake_cli, capsys
):
    """build_graph raises, main() turns it into exit code 1 and a message."""
    fake_cli("never reached")
    assert cli.main(["graph-agent", "Q", "--checkpointer", "none", "--approve"]) == 1
    assert "needs a checkpointer" in capsys.readouterr().err
