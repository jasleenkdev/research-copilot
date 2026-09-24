"""Phase 6.4: threads, human review, stored policy, and pruning on the
multi-agent graph - offline.

Every "separate process" here is a separate `cli.main([...])` call that
rebuilds the graph from scratch against the same sqlite file. Nothing is
shared between the calls except the checkpointer and the thread_id - the
Phase 4 test for "this actually persisted".
"""

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
import sqlite3

import pytest

from research_copilot import cli
from research_copilot.checkpointing import thread_config
from research_copilot.graph import final_answer
from research_copilot.multi_agent_graph import (
    build_multi_agent_graph,
    multi_agent_turn_input,
    run_multi_agent,
    run_policy,
)

# --- fakes ----------------------------------------------------------------------


class RecordingModel(FakeMessagesListChatModel):
    requests: list = []

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.requests.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def scripted(*replies):
    return RecordingModel(
        responses=[r if isinstance(r, AIMessage) else AIMessage(content=r) for r in replies]
    )


@tool
def search_arxiv(query: str) -> str:
    """Stub search."""
    return f"[1] A paper about {query}"


@tool
def verify_citation(arxiv_id: str) -> str:
    """Stub verifier."""
    return f"FOUND: {arxiv_id} - A paper"


@pytest.fixture
def fake_agents(monkeypatch):
    """Install one scripted model per agent for the CLI (which has no model
    flags). Each call replaces the scripts; the shared checkpoint DB is the
    autouse fixture in conftest.py."""

    def install(*, researcher=("notes",), writer=("draft",), critic=("APPROVE",)):
        models = {
            "researcher": scripted(*researcher),
            "writer": scripted(*writer),
            "critic": scripted(*critic),
        }
        for agent, model in models.items():
            monkeypatch.setattr(
                f"research_copilot.agents.{agent}.get_chat_model", lambda m=model, **k: m
            )
        monkeypatch.setattr("research_copilot.agents.critic.verify_citation", verify_citation)
        monkeypatch.setattr("research_copilot.agents.researcher.search_arxiv", search_arxiv)
        return models

    return install


def thread_from(err: str) -> str:
    return next(line.split()[1] for line in err.splitlines() if line.startswith("[thread] "))


FIXED = ["--routing", "fixed", "--checkpointer", "sqlite", "--memory", "none"]


# --- Part A: threads -----------------------------------------------------------------


def test_a_thread_continues_across_separate_invocations(fake_agents, capsys):
    models = fake_agents(researcher=("n1", "n2"), writer=("answer one", "answer two"))
    assert cli.main(["multi-agent", "first", *FIXED]) == 0
    thread = thread_from(capsys.readouterr().err)

    assert cli.main(["multi-agent", "second", *FIXED, "--thread", thread]) == 0
    out, _ = capsys.readouterr()
    assert "answer two" in out
    # The second turn's Researcher read the first turn's committed answer.
    assert "answer one" in "\n".join(m.text for m in models["researcher"].requests[-1])


def test_approve_pauses_and_multi_review_resumes_in_another_invocation(fake_agents, capsys):
    fake_agents(writer=("the draft",))
    assert cli.main(["multi-agent", "Q", *FIXED, "--approve"]) == 0
    out, err = capsys.readouterr()
    thread = thread_from(err)
    assert "--- awaiting approval ---" in out
    assert f"multi-review --thread {thread}" in err

    assert cli.main(["multi-review", "--thread", thread, "--approve"]) == 0
    out, _ = capsys.readouterr()
    assert out.strip().splitlines()[-1] == "the draft"


def test_multi_review_edit_commits_the_human_text(fake_agents, capsys):
    fake_agents(writer=("model text",))
    cli.main(["multi-agent", "Q", *FIXED, "--approve"])
    thread = thread_from(capsys.readouterr().err)

    cli.main(["multi-review", "--thread", thread, "--edit", "human text"])
    out, err = capsys.readouterr()
    assert "human text" in out
    assert "draft: 'model text'" in err  # the Writer's draft is kept beside it


def test_multi_review_uses_the_stored_policy_not_its_own_flags(fake_agents, capsys):
    """Started with --critic --approve --max-revisions 1. The review process is
    given none of those flags; the revision it triggers still goes through the
    Critic, because the policy came from the thread."""
    models = fake_agents(writer=("draft 1", "draft 2"), critic=("APPROVE", "APPROVE"))
    cli.main(["multi-agent", "Q", *FIXED, "--critic", "--approve", "--max-revisions", "1"])
    thread = thread_from(capsys.readouterr().err)
    assert len(models["critic"].requests) == 1

    cli.main(["multi-review", "--thread", thread, "--reject", "--note", "tighten it"])
    out, err = capsys.readouterr()
    assert "roster: researcher, writer, critic; max revisions 1; human approval on" in err
    assert len(models["critic"].requests) == 2  # the revised draft was critiqued
    assert "[revised] revision 1" in err
    assert "draft 2" in out

    cli.main(["multi-review", "--thread", thread, "--reject", "--note", "still no"])
    out, _ = capsys.readouterr()
    # max_revisions 1 came from the thread, so the second rejection withholds.
    assert "(draft withheld after 1 revision - human: still no)" in out


def test_a_new_turn_on_a_paused_thread_is_refused(fake_agents, capsys):
    fake_agents()
    cli.main(["multi-agent", "Q", *FIXED, "--approve"])
    thread = thread_from(capsys.readouterr().err)

    cli.main(["multi-agent", "another question", *FIXED, "--thread", thread])
    _, err = capsys.readouterr()
    assert "has a draft awaiting approval" in err


def test_why_the_paused_thread_guard_exists():
    """What LangGraph itself does with new input on a parked thread: it runs
    the new turn, and the parked draft is abandoned - leaving two human turns
    adjacent in the transcript. The CLI guard above exists because of this."""
    g = build_multi_agent_graph(
        researcher_model=scripted("n1", "n2"), writer_model=scripted("d1", "d2"),
        tools=[search_arxiv], routing="fixed", require_approval=True,
        checkpointer=MemorySaver(), memory_strategy="none",
    )
    config = {"configurable": {"thread_id": "t"}}
    g.invoke(multi_agent_turn_input("first"), config)
    assert g.get_state(config).interrupts

    g.invoke(multi_agent_turn_input("second"), config)
    types = [m.type for m in g.get_state(config).values["messages"]]
    assert types[:2] == ["human", "human"]


def test_a_new_turn_may_change_policy(fake_agents, capsys):
    """Policy is per turn: the same thread, critic off then on."""
    models = fake_agents(researcher=("n1", "n2"), writer=("a1", "a2"), critic=("APPROVE",))
    cli.main(["multi-agent", "first", *FIXED])
    thread = thread_from(capsys.readouterr().err)
    assert models["critic"].requests == []

    cli.main(["multi-agent", "second", *FIXED, "--thread", thread, "--critic"])
    capsys.readouterr()
    assert len(models["critic"].requests) == 1


# --- cross-graph and pre-6.4 threads ------------------------------------------------------


def saver_for(tmp_path):
    return SqliteSaver(sqlite3.connect(tmp_path / "checkpoints.sqlite3", check_same_thread=False))


def test_single_agent_review_refuses_a_multi_agent_thread(fake_agents, capsys):
    fake_agents()
    cli.main(["multi-agent", "Q", *FIXED, "--approve"])
    thread = thread_from(capsys.readouterr().err)

    cli.main(["review", "--thread", thread, "--approve", "--checkpointer", "sqlite"])
    _, err = capsys.readouterr()
    assert "written by the multi-agent graph" in err
    assert f"multi-review --thread {thread}" in err


def test_get_state_is_filtered_by_the_reading_graphs_schema(fake_agents, capsys, tmp_path):
    """The finding behind `_raw_channels`: the single-agent graph cannot see a
    multi-agent thread's own keys through get_state, so it cannot tell the
    thread is not its own that way."""
    from research_copilot.graph import build_graph

    fake_agents()
    cli.main(["multi-agent", "Q", *FIXED])
    thread = thread_from(capsys.readouterr().err)
    saver = saver_for(tmp_path)

    seen_by_single = build_graph(checkpointer=saver).get_state(thread_config(thread)).values
    assert "run_policy" not in seen_by_single
    assert "messages" in seen_by_single

    raw = saver.get_tuple(thread_config(thread)).checkpoint["channel_values"]
    assert "run_policy" in raw


def test_multi_agent_refuses_to_continue_a_single_agent_thread(fake_agents, capsys, tmp_path):
    from research_copilot.graph import build_graph, run_graph

    run_graph("Q", graph=build_graph(model=scripted("a"), checkpointer=saver_for(tmp_path), memory_strategy="none"), thread_id="single")
    fake_agents()
    cli.main(["multi-agent", "next", *FIXED, "--thread", "single"])
    _, err = capsys.readouterr()
    assert "belongs to the single-agent graph" in err


def test_multi_review_refuses_a_single_agent_thread(capsys, tmp_path, monkeypatch):
    from research_copilot.graph import build_graph, run_graph

    monkeypatch.setenv("RESEARCH_COPILOT_CHECKPOINT_DB", str(tmp_path / "checkpoints.sqlite3"))
    saver = saver_for(tmp_path)
    g = build_graph(model=scripted("draft"), require_approval=True, checkpointer=saver, memory_strategy="none")
    run_graph("Q", graph=g, thread_id="single")

    cli.main(["multi-review", "--thread", "single", "--approve", "--checkpointer", "sqlite"])
    _, err = capsys.readouterr()
    assert "written by the single-agent graph" in err


def test_multi_review_refuses_a_pre_6_4_thread_with_no_stored_policy(capsys, tmp_path):
    saver = saver_for(tmp_path)
    g = build_multi_agent_graph(checkpointer=saver)
    g.update_state(
        thread_config("old"),
        {"messages": [HumanMessage(content="q")], "supervisor_log": []},
        as_node="plan_question",
    )
    cli.main(["multi-review", "--thread", "old", "--approve", "--checkpointer", "sqlite"])
    _, err = capsys.readouterr()
    assert "predates 6.4 and has no stored run policy" in err


def test_multi_review_on_a_thread_with_subgraph_checkpoints_but_nothing_pending(fake_agents, capsys, tmp_path):
    """The Researcher's private channel is on disk under its own namespace, but
    nothing is parked at the parent. That is history, not a pending decision."""
    fake_agents(researcher=(AIMessage(content="", tool_calls=[{"name": "search_arxiv", "args": {"query": "x"}, "id": "1", "type": "tool_call"}]), "notes"))
    cli.main(["multi-agent", "Q", *FIXED])
    thread = thread_from(capsys.readouterr().err)

    saver = saver_for(tmp_path)
    namespaces = {c.config["configurable"]["checkpoint_ns"] for c in saver.list(thread_config(thread))}
    assert any(ns.startswith("researcher:") for ns in namespaces)

    cli.main(["multi-review", "--thread", thread, "--approve", "--checkpointer", "sqlite"])
    _, err = capsys.readouterr()
    assert "nothing is awaiting approval" in err


def test_run_policy_rejects_non_policy_keys():
    with pytest.raises(TypeError, match="checkpointer"):
        run_policy(checkpointer=None, enable_critic=True)
    assert run_policy(enable_critic=True, dispatch_caps=None) == {"enable_critic": True}


# --- Part B: pruning ---------------------------------------------------------------------


def long_thread(*, strategy, budget, turns=4, summary_model=None):
    g = build_multi_agent_graph(
        researcher_model=scripted(*["notes"] * turns),
        writer_model=scripted(*[f"answer {i} " + "word " * 40 for i in range(turns)]),
        summary_model=summary_model,
        tools=[search_arxiv],
        routing="fixed",
        checkpointer=MemorySaver(),
        memory_strategy=strategy,
        max_history_tokens=budget,
    )
    config = {"configurable": {"thread_id": "t"}}
    states = [run_multi_agent(f"question {i}", graph=g, config=config) for i in range(turns)]
    return g, config, states


def test_trim_keeps_the_transcript_bounded_across_turns():
    _, _, states = long_thread(strategy="trim", budget=150)
    lengths = [len(s["messages"]) for s in states]
    assert lengths[-1] < 2 * len(states)
    assert states[-1]["messages"][0].type == "human"


def test_no_pruning_grows_by_exactly_two_per_turn():
    """Small by construction: private channels keep tool traffic out, so each
    turn adds a question and an answer and nothing else."""
    _, _, states = long_thread(strategy="none", budget=0)
    assert [len(s["messages"]) for s in states] == [2, 4, 6, 8]


def test_summarize_folds_dropped_turns_and_the_writer_reads_the_summary():
    summarizer = scripted(*["SUMMARY: earlier questions 0-2"] * 4)
    g, config, states = long_thread(strategy="summarize", budget=150, summary_model=summarizer)
    assert states[-1]["summary"].startswith("SUMMARY")


def test_resuming_a_paused_turn_does_not_prune():
    """Pruning is at the entry; a resume re-enters at the parked node."""
    g = build_multi_agent_graph(
        researcher_model=scripted("n"), writer_model=scripted("d"), tools=[search_arxiv],
        routing="fixed", require_approval=True, checkpointer=MemorySaver(),
        memory_strategy="trim", max_history_tokens=1,
    )
    config = {"configurable": {"thread_id": "t"}}
    g.invoke(multi_agent_turn_input("Q"), config)
    nodes = [node for event in g.stream(Command(resume="approve"), config, stream_mode="updates") for node in event]
    assert "prune_history" not in nodes
    assert "finalize_answer" in nodes


def test_pruning_never_touches_private_channels():
    """Nothing to touch: they are not in `messages`. Pruning deletes parent
    transcript entries only; the Researcher's history lives elsewhere."""
    _, config_state_graph, states = long_thread(strategy="trim", budget=150)
    for s in states:
        assert "research_messages" not in s
        assert not any(m.type == "tool" for m in s["messages"])


# --- found in Studio: the turn reset must live in the graph --------------------------


def test_bare_input_with_no_helper_still_gets_a_fresh_turn():
    """What Studio (and any API caller) sends: a question and a message, not
    `multi_agent_turn_input`. Before 6.4's `begin_turn`, turn 2 inherited turn
    1's dispatches, revisions and decision log."""
    g = build_multi_agent_graph(
        researcher_model=scripted("n1", "n2"), writer_model=scripted("a1", "a2"),
        tools=[search_arxiv], routing="fixed", checkpointer=MemorySaver(), memory_strategy="none",
    )
    config = {"configurable": {"thread_id": "t"}}
    bare = lambda q: {"question": q, "messages": [HumanMessage(content=q)]}  # noqa: E731

    g.invoke(bare("first"), config)
    state = g.invoke(bare("second"), config)

    assert state["dispatches"] == {"researcher": 1, "writer": 1}
    assert len(state["supervisor_log"]) == 3
    assert state["revisions"] == 0
    assert final_answer(state) == "a2"
