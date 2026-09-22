"""Phase 5 Parts A-C: the reflection loop, entirely offline.

No real critique is ever produced here. The critic is a scripted model that
says APPROVE or REJECT on cue, which is enough to test everything that is
actually Phase 5: where the run goes after a verdict, which counter moves, and
whether the cap can stop a reviewer that never says yes.

What these tests cannot tell you is whether a real critic writes useful
critiques or whether revised drafts are better than the drafts they replace.
That is a question about output quality and it needs a real key - see the
README's Phase 5 section.
"""

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from research_copilot.checkpointing import checkpointer_scope
from research_copilot.graph import (
    _parse_critique,
    _parse_verdict,
    build_graph,
    final_answer,
    resume_graph,
    revision_input,
    run_graph,
)


class FakeToolCallingModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


@tool
def search_arxiv(query: str) -> str:
    """Stub search."""
    return f"[1] A paper about {query}"


def tool_call(name, args, call_id):
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


def critic(*verdicts):
    """A scripted reviewer. Each entry is one whole critic reply."""
    return FakeToolCallingModel(responses=[AIMessage(content=v) for v in verdicts])


def writer(*drafts):
    return FakeToolCallingModel(
        responses=[d if isinstance(d, AIMessage) else AIMessage(content=d) for d in drafts]
    )


def critic_graph(*, writer_model, critic_model, max_revisions=2, **kwargs):
    """A critic-only graph: no human, so no checkpointer is needed.

    That this compiles at all is half of Part C. `require_approval=True` without
    a checkpointer is refused at build time because interrupt() has nowhere to
    park; `enable_critic=True` needs nothing, because a model call blocks and
    returns like any other function call.
    """
    return build_graph(
        model=writer_model,
        critic_model=critic_model,
        tools=[search_arxiv],
        enable_critic=True,
        max_revisions=max_revisions,
        memory_strategy="none",
        **kwargs,
    )


# --- Part C: the critic is the human, structurally ---------------------------


def test_a_critic_needs_no_checkpointer_although_a_human_does():
    """The one real asymmetry between the two reviewers: only the human parks."""
    graph = critic_graph(writer_model=writer("draft"), critic_model=critic("APPROVE"))
    state = run_graph("Q", graph=graph)

    assert final_answer(state) == "draft"
    assert state["status"] == "approved"

    with pytest.raises(RuntimeError, match="needs a checkpointer"):
        build_graph(model=writer("x"), require_approval=True)


def test_an_approving_critic_commits_the_draft_like_an_approving_human():
    graph = critic_graph(writer_model=writer("the answer"), critic_model=critic("APPROVE"))
    state = run_graph("Q", graph=graph)

    assert [m.type for m in state["messages"]] == ["human", "ai"]
    assert final_answer(state) == "the answer"
    # Nothing was revised, so the second counter never moved.
    assert state.get("revisions", 0) == 0
    assert state["draft"] == ""


def test_the_draft_is_not_in_the_transcript_while_the_critic_is_reading_it():
    """Same reason as the human path: `messages` records what was said."""
    graph = critic_graph(
        writer_model=writer("v1", "v2"),
        critic_model=critic("REJECT\nno sources", "APPROVE"),
    )
    state = run_graph("Q", graph=graph)

    # v1 was rejected and never said. Only v2 is in the transcript.
    assert [m.text for m in state["messages"] if m.type == "ai"] == ["v2"]
    assert "v1" not in [m.text for m in state["messages"]]


def test_a_rejected_draft_is_redrafted_rather_than_ending_the_run():
    """Part B: rejection is a loop now, not an ending."""
    graph = critic_graph(
        writer_model=writer("thin draft", "much better draft"),
        critic_model=critic("REJECT\ncite something", "APPROVE"),
    )
    state = run_graph("Q", graph=graph)

    assert final_answer(state) == "much better draft"
    assert state["revisions"] == 1
    assert state["status"] == "approved"


def test_the_critique_reaches_the_writer_as_the_revision_instruction():
    """The edge carries no data - state does. This is how the note gets there."""
    seen: list[list] = []

    class RecordingWriter(FakeToolCallingModel):
        def invoke(self, input, config=None, **kwargs):
            seen.append(list(input))
            return super().invoke(input, config, **kwargs)

    graph = critic_graph(
        writer_model=RecordingWriter(
            responses=[AIMessage(content="v1"), AIMessage(content="v2")]
        ),
        critic_model=critic("REJECT\nthe cost claim has no source", "APPROVE"),
    )
    run_graph("Q", graph=graph)

    first_request, revision_request = seen
    assert not any("cost claim" in m.text for m in first_request)

    revision_text = "\n".join(m.text for m in revision_request)
    assert "the cost claim has no source" in revision_text
    # The rejected draft goes back too - a critique of text the writer cannot
    # see is not actionable.
    assert "v1" in revision_text
    assert "revision attempt 1 of at most 2" in revision_text


# --- Part B: the cap, and why it has to overrule the verdict ------------------


def test_a_critic_that_always_rejects_is_stopped_by_the_cap():
    """The test the whole design exists for.

    The reviewer never approves. Nothing in any verdict says "stop". The loop
    stops anyway, because the cap is checked before the verdict is trusted.
    """
    graph = critic_graph(
        writer_model=writer("v1", "v2", "v3", "v4", "v5"),
        critic_model=critic(*["REJECT\nstill not good enough"] * 5),
        max_revisions=2,
    )
    state = run_graph("Q", graph=graph)

    # 1 original draft + exactly 2 revisions, then the cap overrules the "no".
    assert state["revisions"] == 2
    assert state["status"] == "rejected"
    assert "withheld" in final_answer(state)
    # The message names the cap, so this is not misdiagnosed as a harsh reviewer.
    assert "after 2 revisions" in final_answer(state)


def test_the_turn_is_still_answered_when_the_cap_is_spent():
    """The cap routes to finalize_answer, not straight to END.

    Ending without it would leave the opening HumanMessage unanswered, and two
    adjacent human turns are rejected by the Anthropic API outright.
    """
    graph = critic_graph(
        writer_model=writer("v1", "v2"),
        critic_model=critic("REJECT\nno", "REJECT\nstill no"),
        max_revisions=1,
    )
    state = run_graph("Q", graph=graph)

    assert [m.type for m in state["messages"]] == ["human", "ai"]
    assert state["draft"] == ""


def test_max_revisions_zero_reproduces_phase_4_exactly():
    """The library default: a new phase is something you switch on."""
    graph = critic_graph(
        writer_model=writer("v1", "unused"),
        critic_model=critic("REJECT\nnope"),
        max_revisions=0,
    )
    state = run_graph("Q", graph=graph)

    assert state.get("revisions", 0) == 0
    assert "withheld" in final_answer(state)


# --- Part A: two counters, two schedules --------------------------------------


def test_each_revision_gets_a_fresh_tool_budget():
    """The reason the counters are separate, demonstrated rather than argued.

    Every round runs the tool loop twice (one tool call, one answer). With a
    shared counter the second round would start at 2 and the third at 4, and a
    cap of 3 would refuse a tool call in round 2 for work round 1 did.
    """
    graph = critic_graph(
        writer_model=writer(
            tool_call("search_arxiv", {"query": "a"}, "c1"),
            AIMessage(content="v1"),
            tool_call("search_arxiv", {"query": "b"}, "c2"),
            AIMessage(content="v2"),
            tool_call("search_arxiv", {"query": "c"}, "c3"),
            AIMessage(content="v3"),
        ),
        critic_model=critic("REJECT\nmore", "REJECT\nmore", "APPROVE"),
        max_revisions=2,
        max_iterations=3,
    )
    state = run_graph("Q", graph=graph)

    assert final_answer(state) == "v3"
    assert state["revisions"] == 2
    # `iterations` describes the *current* round, not the run: 2 calls, not 6.
    assert state["iterations"] == 2
    # All three searches actually ran - no round was starved by an earlier one.
    assert len([m for m in state["messages"] if m.type == "tool"]) == 3


def test_revision_input_resets_the_inner_budget_and_only_that():
    """`turn_input`'s pattern, one level down."""
    update = revision_input(
        {
            "revisions": 1,
            "iterations": 5,
            "status": "rejected",
            "draft": "the rejected text",
            "critique": "why",
        }
    )

    assert update == {"revisions": 2, "iterations": 0, "status": "drafting"}
    # Not cleared: the draft is what is being revised and the critique is the
    # instruction. Clearing either would send the writer back empty-handed.
    assert "draft" not in update
    assert "critique" not in update


def test_a_new_turn_resets_the_revision_counter_and_the_stale_feedback():
    """A checkpointed counter that never resets is a budget that only runs out."""
    with checkpointer_scope("memory") as saver:
        graph = critic_graph(
            writer_model=writer("v1", "v2", "v3"),
            critic_model=critic("REJECT\nturn one problem", "APPROVE", "APPROVE"),
            checkpointer=saver,
            max_revisions=2,
        )
        run_graph("Q1", graph=graph, thread_id="t")
        state = run_graph("Q2", graph=graph, thread_id="t")

        assert state["revisions"] == 0
        # Turn 1's complaint must not follow Q2 around.
        assert state["critique"] in ("", "(approved by the critic)")
        assert final_answer(state) == "v3"


# --- both reviewers at once ---------------------------------------------------


def hitl_critic_graph(saver, *, writer_model, critic_model, max_revisions=2):
    return build_graph(
        model=writer_model,
        critic_model=critic_model,
        tools=[search_arxiv],
        checkpointer=saver,
        require_approval=True,
        enable_critic=True,
        max_revisions=max_revisions,
        memory_strategy="none",
    )


def test_the_critic_goes_first_and_the_human_sees_only_what_it_passed():
    with checkpointer_scope("memory") as saver:
        graph = hitl_critic_graph(
            saver,
            writer_model=writer("v1", "v2"),
            critic_model=critic("REJECT\nv1 is unsupported", "APPROVE"),
        )
        state = run_graph("Q", graph=graph, thread_id="t")

        # The run paused only once, and on v2. The human never saw v1.
        assert "__interrupt__" in state
        assert state["__interrupt__"][0].value["draft"] == "v2"
        assert state["revisions"] == 1

        done = resume_graph(graph, "approve", thread_id="t")
        assert final_answer(done) == "v2"


def test_the_human_can_reject_what_the_critic_approved():
    """The two reviewers disagreeing, in the direction the ordering allows."""
    with checkpointer_scope("memory") as saver:
        graph = hitl_critic_graph(
            saver,
            writer_model=writer("v1", "v2"),
            critic_model=critic("APPROVE", "APPROVE"),
        )
        run_graph("Q", graph=graph, thread_id="t")

        state = resume_graph(
            graph, {"decision": "reject", "note": "misses the actual question"},
            thread_id="t",
        )

        # A human rejection starts a revision exactly as a critic rejection does.
        assert state["revisions"] == 1
        assert "__interrupt__" in state
        assert state["__interrupt__"][0].value["draft"] == "v2"


def test_the_writer_is_shown_both_reviewers_notes_labelled():
    """When they disagree, who said what is the thing the writer needs."""
    seen: list[list] = []

    class RecordingWriter(FakeToolCallingModel):
        def invoke(self, input, config=None, **kwargs):
            seen.append(list(input))
            return super().invoke(input, config, **kwargs)

    with checkpointer_scope("memory") as saver:
        graph = hitl_critic_graph(
            saver,
            writer_model=RecordingWriter(
                responses=[AIMessage(content="v1"), AIMessage(content="v2")]
            ),
            critic_model=critic("APPROVE\nfine but shallow", "APPROVE"),
        )
        run_graph("Q", graph=graph, thread_id="t")
        resume_graph(graph, {"decision": "reject", "note": "wrong scope"}, thread_id="t")

        revision_text = "\n".join(m.text for m in seen[-1])
        assert "Machine critic: fine but shallow" in revision_text
        assert "Human reviewer: wrong scope" in revision_text


def test_a_critic_that_never_approves_means_the_human_is_never_asked():
    """The documented cost of critic-first ordering, pinned as behaviour.

    With --critic --approve and a critic that rejects everything, the revision
    cap is spent before the interrupt is ever reached. `--approve` looks like it
    did nothing; it was simply never got to. cli.py warns about exactly this.
    """
    with checkpointer_scope("memory") as saver:
        graph = hitl_critic_graph(
            saver,
            writer_model=writer("v1", "v2", "v3"),
            critic_model=critic(*["REJECT\nno"] * 3),
            max_revisions=2,
        )
        state = run_graph("Q", graph=graph, thread_id="t")

        assert "__interrupt__" not in state
        assert state["revisions"] == 2
        assert "withheld" in final_answer(state)


# --- parsing: one verdict shape, two reviewers --------------------------------


@pytest.mark.parametrize(
    "raw,decision,note_fragment",
    [
        ("APPROVE", "approve", ""),
        ("approve", "approve", ""),
        ("**APPROVE**", "approve", ""),
        ("REJECT\nno sources", "reject", "no sources"),
        ("REJECT:\nno sources", "reject", "no sources"),
        # A sentence that merely contains the word is not a verdict. This one
        # matters: reading it as consent is the failure the format exists to
        # prevent.
        ("I would not APPROVE this", "reject", "would not APPROVE"),
        ("Looks good to me!", "reject", "Looks good"),
        ("", "reject", "returned nothing"),
    ],
)
def test_critic_replies_reduce_to_the_same_verdicts_a_human_produces(
    raw, decision, note_fragment
):
    parsed = _parse_verdict(_parse_critique(raw))
    assert parsed[0] == decision
    assert note_fragment in parsed[2]


def test_a_critic_that_forgets_the_format_fails_closed_into_a_revision():
    """Fail-closed costs a revision here rather than a wrongly published answer.

    Worth knowing as a failure mode: a critic whose output format drifts becomes
    a critic that always rejects, and the symptom is a run that always exhausts
    its revisions. The raw reply is kept as the note so that is diagnosable.
    """
    graph = critic_graph(
        writer_model=writer("v1", "v2"),
        critic_model=critic("Yeah that seems fine", "APPROVE"),
        max_revisions=1,
    )
    state = run_graph("Q", graph=graph)

    assert state["revisions"] == 1
    assert final_answer(state) == "v2"


def test_an_empty_critique_still_produces_an_actionable_instruction():
    """A rejection with no reason is not actionable; say so rather than send
    the writer an empty bullet list, which reads as "nothing was wrong".

    Two fallbacks are layered here and the outer one wins, which is the right
    way round: `critique_draft` substitutes a note when the critic gives none,
    so the writer is told *who* rejected it without a reason. The fallback in
    `_revision_instruction` is the backstop for the case where neither reviewer
    field was written at all.
    """
    seen: list[list] = []

    class RecordingWriter(FakeToolCallingModel):
        def invoke(self, input, config=None, **kwargs):
            seen.append(list(input))
            return super().invoke(input, config, **kwargs)

    graph = critic_graph(
        writer_model=RecordingWriter(
            responses=[AIMessage(content="v1"), AIMessage(content="v2")]
        ),
        critic_model=critic("REJECT", "APPROVE"),
        max_revisions=1,
    )
    run_graph("Q", graph=graph)

    revision_text = "\n".join(m.text for m in seen[-1])
    assert "Machine critic: (rejected by the critic without a reason given)" in revision_text


# --- one graph shape, whatever the flags say ---------------------------------


def test_the_phase_5_nodes_exist_even_when_the_flags_are_off():
    """Same rule as Phase 4's review nodes: flags choose paths, not structures."""
    drawn = build_graph(model=writer("x"), tools=[search_arxiv]).get_graph()

    assert {"plan_question", "critique_draft", "start_revision"} <= set(drawn.nodes)
