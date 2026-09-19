"""Phase 4 Part B: keeping a persisted transcript inside a token budget.

The distinction under test is the one the module docstring in graph.py argues
about: a filtering node changes what the model sees, and a RemoveMessage node
changes what is in the state. `prune_history` does the second, so these tests
assert against the *stored* transcript, not against what was sent.
"""

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langchain_core.messages.utils import count_tokens_approximately

from research_copilot.checkpointing import checkpointer_scope, thread_config
from research_copilot.graph import build_graph, run_graph


class FakeToolCallingModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


class RecordingModel(FakeToolCallingModel):
    """A fake that also remembers what it was asked.

    Needed because the difference between the two pruning strategies is only
    visible from two angles at once: what got stored, and what got sent.
    """

    requests: list = []

    def invoke(self, input, config=None, **kwargs):
        self.requests.append(list(input) if isinstance(input, list) else [input])
        return super().invoke(input, config, **kwargs)


def long_answer(n: int) -> str:
    """A reply big enough to move a token budget."""
    return f"answer {n}: " + ("padding words to burn context budget " * 20)


# --- the budget is enforced on the stored transcript -------------------------


def test_an_unpruned_thread_grows_without_bound():
    """The baseline, and the reason Part B exists at all.

    With pruning off, every turn is added to the checkpoint and nothing ever
    leaves. This is what `--max-history-tokens 0` gives you, and what Phase 3
    would have given you the moment a checkpointer was attached.
    """
    with checkpointer_scope("memory") as saver:
        graph = build_graph(
            model=FakeToolCallingModel(
                responses=[AIMessage(content=long_answer(i)) for i in range(6)]
            ),
            checkpointer=saver,
            max_history_tokens=0,
        )
        for i in range(6):
            state = run_graph(f"Q{i}", graph=graph, thread_id="t")

        assert len(state["messages"]) == 12
        assert count_tokens_approximately(state["messages"]) > 1000


def test_trim_deletes_old_messages_from_the_persisted_state():
    """The Part B claim: the checkpoint itself gets shorter, not just the request."""
    with checkpointer_scope("memory") as saver:
        graph = build_graph(
            model=FakeToolCallingModel(
                responses=[AIMessage(content=long_answer(i)) for i in range(8)]
            ),
            checkpointer=saver,
            memory_strategy="trim",
            max_history_tokens=400,
        )
        for i in range(8):
            run_graph(f"Q{i}", graph=graph, thread_id="t")

        # Read the stored snapshot rather than the invoke() return value - this
        # is what the next process would load.
        stored = graph.get_state(thread_config("t")).values["messages"]

        assert len(stored) < 16, "old turns should be gone from state"
        assert count_tokens_approximately(stored) < 1200
        # The newest turn always survives.
        assert any("Q7" == m.text for m in stored)
        # The oldest is genuinely gone, not merely unsent.
        assert not any("Q0" == m.text for m in stored)


def test_pruning_keeps_the_transcript_valid_for_the_api():
    """Trimming has rules, and `trim_messages` is reused because it knows them.

    A history that starts on an assistant turn is rejected by the Anthropic API,
    so `start_on="human"` is not a stylistic preference.
    """
    with checkpointer_scope("memory") as saver:
        graph = build_graph(
            model=FakeToolCallingModel(
                responses=[AIMessage(content=long_answer(i)) for i in range(8)]
            ),
            checkpointer=saver,
            memory_strategy="trim",
            max_history_tokens=400,
        )
        for i in range(8):
            run_graph(f"Q{i}", graph=graph, thread_id="t")

        stored = graph.get_state(thread_config("t")).values["messages"]
        assert isinstance(stored[0], HumanMessage)


def test_pruning_never_deletes_the_question_being_answered():
    """A budget smaller than the current turn must not eat the current turn.

    Answering a question you just deleted is a worse failure than going over
    budget, so the node falls back to the tail from the last human turn.
    """
    with checkpointer_scope("memory") as saver:
        graph = build_graph(
            model=FakeToolCallingModel(
                responses=[AIMessage(content=long_answer(i)) for i in range(4)]
            ),
            checkpointer=saver,
            memory_strategy="trim",
            max_history_tokens=1,  # absurdly small on purpose
        )
        for i in range(4):
            state = run_graph(f"Q{i}", graph=graph, thread_id="t")

        stored = graph.get_state(thread_config("t")).values["messages"]
        assert stored, "the transcript must never be emptied"
        assert any(m.text == "Q3" for m in stored)


def test_a_transcript_inside_the_budget_is_left_alone():
    with checkpointer_scope("memory") as saver:
        graph = build_graph(
            model=FakeToolCallingModel(
                responses=[AIMessage(content="short"), AIMessage(content="short")]
            ),
            checkpointer=saver,
            memory_strategy="trim",
            max_history_tokens=5000,
        )
        run_graph("Q0", graph=graph, thread_id="t")
        state = run_graph("Q1", graph=graph, thread_id="t")

        assert [m.text for m in state["messages"]] == ["Q0", "short", "Q1", "short"]


def test_strategy_none_disables_the_node_without_removing_it():
    """The node is always in the graph; only its behaviour is switched off.

    Same reasoning as the review nodes: one graph shape regardless of flags.
    """
    graph = build_graph(
        model=FakeToolCallingModel(responses=[AIMessage(content="x")]),
        memory_strategy="none",
    )
    assert "prune_history" in graph.get_graph().nodes


# --- summarize: the same deletion, plus a record of what was deleted ---------


def test_summarize_writes_a_summary_and_still_deletes_the_messages():
    with checkpointer_scope("memory") as saver:
        graph = build_graph(
            model=FakeToolCallingModel(
                responses=[AIMessage(content=long_answer(i)) for i in range(8)]
            ),
            # A separate scripted model for the compression step, which is why
            # build_graph takes summary_model apart from model.
            summary_model=FakeToolCallingModel(
                responses=[AIMessage(content="SUMMARY OF EARLIER TURNS")] * 8
            ),
            checkpointer=saver,
            memory_strategy="summarize",
            max_history_tokens=400,
        )
        for i in range(8):
            run_graph(f"Q{i}", graph=graph, thread_id="t")

        stored = graph.get_state(thread_config("t")).values

        assert stored["summary"] == "SUMMARY OF EARLIER TURNS"
        assert not any(m.text == "Q0" for m in stored["messages"])
        # The summary is NOT a message - see the note on `summary` in state.py.
        assert "SUMMARY OF EARLIER TURNS" not in [m.text for m in stored["messages"]]


def test_the_summary_is_sent_to_the_model_ahead_of_the_surviving_turns():
    """Why `summary` is a state key and not an appended SystemMessage.

    `add_messages` appends, so a summary returned as a message would land after
    the turns it summarizes. call_model instead renders it into position.
    """
    recorder = RecordingModel(
        responses=[AIMessage(content=long_answer(i)) for i in range(8)]
    )
    recorder.requests = []

    with checkpointer_scope("memory") as saver:
        graph = build_graph(
            model=recorder,
            summary_model=FakeToolCallingModel(
                responses=[AIMessage(content="EARLIER GIST")] * 8
            ),
            checkpointer=saver,
            memory_strategy="summarize",
            max_history_tokens=400,
        )
        for i in range(8):
            run_graph(f"Q{i}", graph=graph, thread_id="t")

    last_request = recorder.requests[-1]
    rendered = [m.text for m in last_request]
    assert any("EARLIER GIST" in text for text in rendered)
    # Position matters: system prompt, then the summary, then the live turns.
    summary_index = next(i for i, t in enumerate(rendered) if "EARLIER GIST" in t)
    question_index = next(i for i, t in enumerate(rendered) if t == "Q7")
    assert summary_index < question_index


# --- the mechanism, isolated -------------------------------------------------


def test_prune_history_returns_removemessage_objects():
    """The node's actual output, inspected directly.

    This is what makes it a deletion rather than a filter: `add_messages` reads
    RemoveMessage as "drop this id". A node returning a plain shorter list would
    be read as "append all of these again".
    """
    graph = build_graph(
        model=FakeToolCallingModel(responses=[AIMessage(content="x")]),
        memory_strategy="trim",
        max_history_tokens=30,
    )
    messages = []
    for i in range(8):
        messages.append(HumanMessage(content=long_answer(i), id=f"h{i}"))
        messages.append(AIMessage(content=long_answer(i), id=f"a{i}"))

    update = graph.nodes["prune_history"].invoke({"messages": messages})

    assert update["messages"], "something should have been dropped"
    assert all(isinstance(m, RemoveMessage) for m in update["messages"])


def test_pruning_shortens_the_current_state_but_not_the_checkpoint_history():
    """The honest limit of RemoveMessage, asserted so it is not a surprise.

    A checkpointer writes a new row per super-step and never rewrites old ones.
    Deleting a message removes it from the *latest* snapshot; the earlier rows
    of that thread still contain it. Pruning is a context-window mechanism, not
    a deletion mechanism - `delete_thread` is the one that touches history.
    """
    with checkpointer_scope("memory") as saver:
        graph = build_graph(
            model=FakeToolCallingModel(
                responses=[AIMessage(content=long_answer(i)) for i in range(8)]
            ),
            checkpointer=saver,
            memory_strategy="trim",
            max_history_tokens=400,
        )
        for i in range(8):
            run_graph(f"Q{i}", graph=graph, thread_id="t")

        config = thread_config("t")
        current = graph.get_state(config).values["messages"]
        assert not any(m.text == "Q0" for m in current)

        # But walk the thread's history and Q0 is still there, in an older row.
        historical = [
            m.text
            for snapshot in graph.get_state_history(config)
            for m in snapshot.values.get("messages", [])
        ]
        assert "Q0" in historical
