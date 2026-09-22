"""Phase 5 at the CLI level: the four reviewer combinations over one graph shape.

`--critic` and `--approve` are independent switches, and the point of keeping
them independent is that all four combinations run against the *same* compiled
structure - the flags choose which paths are taken, not which paths exist.
These drive `cli.main()` to check that each combination behaves as advertised.
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
    """One scripted model shared by writer, critic and planner.

    The CLI has no flags for separate critic/planner models - that separation
    exists for tests and for production cost tuning, not for the command line -
    so a CLI-level fake has to serve all three from one script. Each `install`
    argument is therefore the next reply *whoever* asks for one, in order.
    """
    monkeypatch.setenv(
        "RESEARCH_COPILOT_CHECKPOINT_DB", str(tmp_path / "checkpoints.sqlite3")
    )
    monkeypatch.setenv("LANGSMITH_TRACING", "false")

    def install(*texts: str) -> None:
        model = FakeToolCallingModel(responses=[AIMessage(content=t) for t in texts])
        monkeypatch.setattr(
            "research_copilot.graph.get_chat_model", lambda **kwargs: model
        )

    return install


def test_no_reviewer_is_phase_4_with_approval_off(fake_cli, capsys):
    fake_cli("the answer")

    cli.main(["graph-agent", "Q", "--checkpointer", "none", "--memory", "none"])
    captured = capsys.readouterr()

    assert "the answer" in captured.out
    assert "[review]" not in captured.err


def test_critic_only_needs_no_checkpointer(fake_cli, capsys):
    """`--critic --checkpointer none` is a legal combination; `--approve` is not.

    That asymmetry is the one real difference between the two reviewers:
    interrupt() needs somewhere to park a run, a model call does not.
    """
    fake_cli("draft", "APPROVE")

    assert cli.main(
        ["graph-agent", "Q", "--checkpointer", "none", "--memory", "none", "--critic"]
    ) == 0
    captured = capsys.readouterr()

    assert "draft" in captured.out
    assert "[review] critic only" in captured.err


def test_approve_without_a_checkpointer_is_refused_with_a_usable_message(
    fake_cli, capsys
):
    fake_cli("draft")

    assert cli.main(
        ["graph-agent", "Q", "--checkpointer", "none", "--memory", "none", "--approve"]
    ) == 1
    assert "needs a checkpointer" in capsys.readouterr().err


def test_a_rejecting_critic_revises_and_the_cap_ends_it(fake_cli, capsys):
    fake_cli("v1", "REJECT\nno sources", "v2", "REJECT\nstill none", "v3", "REJECT\nno")

    cli.main(
        ["graph-agent", "Q", "--checkpointer", "none", "--memory", "none",
         "--critic", "--max-revisions", "2"]
    )
    captured = capsys.readouterr()

    assert "withheld after 2 revisions" in captured.out
    assert "revisions:  2" in captured.err


def test_both_reviewers_announce_the_ordering_and_its_consequence(fake_cli, capsys):
    """The warning exists because the consequence is invisible from outside:
    a critic that never approves means `--approve` never pauses."""
    fake_cli("v1", "APPROVE")

    cli.main(
        ["graph-agent", "Q", "--thread", "t", "--checkpointer", "sqlite",
         "--memory", "none", "--critic", "--approve"]
    )
    captured = capsys.readouterr()

    assert "[review] critic, then human" in captured.err
    assert "never reaches you" in captured.err
    # The critic passed it, so the human is now the one being asked.
    assert "awaiting approval" in captured.out


def test_critic_with_no_revision_budget_is_warned_about(fake_cli, capsys):
    """A reviewer that can veto but never ask for a fix is not what anyone means."""
    fake_cli("v1", "REJECT\nno")

    cli.main(
        ["graph-agent", "Q", "--checkpointer", "none", "--memory", "none",
         "--critic", "--max-revisions", "0"]
    )
    captured = capsys.readouterr()

    assert "no budget to revise" in captured.err
    assert "withheld" in captured.out


def test_plan_prints_the_sub_questions_it_decided_on(fake_cli, capsys):
    fake_cli("- cost of RAG\n- cost of long context", "the answer")

    cli.main(
        ["graph-agent", "Q", "--checkpointer", "none", "--memory", "none", "--plan"]
    )
    captured = capsys.readouterr()

    assert "plan:       2 sub-questions" in captured.err
    assert "[1] cost of RAG" in captured.err
