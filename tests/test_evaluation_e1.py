"""Phase 7 E1: the evaluation dataset and the EvalRecord - offline.

What these pin down: every Part A scenario becomes an example without losing
an expectation; each example says where it came from; the fingerprints change
only when what is measured changes; and an evaluator's input carries nothing
from the private channels, checked against a real graph run, not assumed
from the streaming tests (the README's standing rule).
"""

import asyncio
import dataclasses
import json

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel, GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

from research_copilot.evaluation import dataset
from research_copilot.evaluation.dataset import (
    EXAMPLES,
    PROVENANCE,
    Provenance,
    build_examples,
    dataset_version,
    from_scenario,
    reference_of,
    validate,
)
from research_copilot.evaluation.record import EVAL_FIELDS, build_eval_record, private_keys
from research_copilot.live_check.scenarios import SCENARIOS, Scenario
from research_copilot.multi_agent_graph import build_multi_agent_graph, multi_agent_turn_input
from research_copilot.streaming import astream_run

BY_ID = {e.id: e for e in EXAMPLES}


# --- the dataset ---------------------------------------------------------------------------------


def test_every_scenario_is_an_example():
    assert [e.id for e in EXAMPLES] == [s.id for s in SCENARIOS]
    assert len(EXAMPLES) == 39


def test_no_expectation_is_lost_in_migration():
    renamed = {"preferred": "preferred_route", "acceptable": "acceptable_routes",
               "found": "must_find", "not_found": "must_reject_by_lookup"}
    for s in SCENARIOS:
        ref = BY_ID[s.id].reference
        for key, value in s.expect.items():
            assert ref[renamed.get(key, key)] == value, (s.id, key)
        assert ref.get("premise_facts", []) == list(s.premise_facts), s.id


def test_every_example_has_something_to_score_against():
    # Part A's probes and e2e runs had no `expect`; their implicit checks are
    # now written in the reference.
    assert all(e.reference for e in EXAMPLES)
    assert BY_ID["E2E01"].reference == {"finishes": True, "no_fixed_policy_fallback": True,
                                        "no_unsupported_citations": True}


def test_an_expect_key_with_no_reference_field_fails_loudly():
    s = Scenario("X01", "6", "critic", "t", {"draft": "d"}, {"verdict": "reject", "typo_found": ["1"]})
    with pytest.raises(ValueError, match="typo_found"):
        reference_of(s)


def test_a_scenario_without_provenance_cannot_enter_the_dataset(monkeypatch):
    extra = Scenario("NEW01", "6", "critic", "t", {"draft": "d"}, {"verdict": "approve"})
    monkeypatch.setattr(dataset, "SCENARIOS", [*SCENARIOS, extra])
    with pytest.raises(ValueError, match="NEW01"):
        from_scenario(extra)
    del PROVENANCE["CRT01"]
    try:
        with pytest.raises(ValueError, match="CRT01"):
            build_examples(SCENARIOS)
    finally:
        PROVENANCE["CRT01"] = Provenance()


def test_post_hoc_examples_are_labelled_as_such():
    """From git history: the A0 scenarios predate every run."""
    post_hoc = {e.id for e in EXAMPLES if not e.pre_registered}
    assert post_hoc == {"SUP02", "SUP04", "SUP06", "SUP08", "SUP10", "SUP13",
                        "SUPC02", "SUPC06", "SUPC08", "SUPC10",
                        "CRT07", "CRT08", "CRT09", "CRT10"}
    assert BY_ID["SUP01"].pre_registered and BY_ID["E2E01"].pre_registered


def test_controls_and_scale_siblings_are_explicit():
    controls = {e.id: e.control_of for e in EXAMPLES if e.control_of}
    assert controls == {"SUPC02": "SUP02", "SUPC06": "SUP06", "SUPC08": "SUP08", "SUPC10": "SUP10",
                        "CRT08": "CRT07", "CRT10": "CRT09"}
    scaled = {e.id: e.scale_of for e in EXAMPLES if e.scale_of}
    assert scaled == {"CRT09": "CRT07", "CRT10": "CRT08"}
    # Measured, not labelled: E2E01's draft has 13 citations, CRT07's has 2.
    assert BY_ID["CRT09"].scale["draft_citations"] == 13 > BY_ID["CRT07"].scale["draft_citations"] == 2
    assert BY_ID["CRT09"].source == "captured"


def test_a_critic_examples_inputs_state_what_the_critic_is_shown():
    crt01 = BY_ID["CRT01"]
    assert crt01.inputs["research_notes"] and crt01.inputs["question"]
    assert crt01.scale["notes_sources"] == 2
    # The large siblings keep their own notes, not the default.
    assert BY_ID["CRT09"].scale["notes_sources"] == 12


def test_the_schema_rules_reject_bad_examples():
    good = BY_ID["CRT09"]
    small = BY_ID["CRT07"]
    with pytest.raises(ValueError, match="not in the critic schema"):
        validate([dataclasses.replace(good, reference={**good.reference, "vibes": "ok"})])
    with pytest.raises(ValueError, match="not larger"):
        validate([small, dataclasses.replace(good, scale={"draft_citations": 2})])
    with pytest.raises(ValueError, match="different verdict"):
        validate([small, dataclasses.replace(good, reference={**good.reference, "verdict": "approve"})])
    with pytest.raises(ValueError, match="same kind"):
        validate([BY_ID["SUP02"], dataclasses.replace(BY_ID["CRT08"], control_of="SUP02")])
    with pytest.raises(ValueError, match="unknown source"):
        validate([dataclasses.replace(small, source="vibes")])


def test_the_fingerprint_follows_what_is_measured_only():
    e = BY_ID["SUP01"]
    assert dataclasses.replace(e, note="reworded", title="renamed", designed_after="x").example_hash == e.example_hash
    assert dataclasses.replace(e, reference={**e.reference, "preferred_route": "writer"}).example_hash != e.example_hash
    assert dataclasses.replace(e, inputs={"state": {}}).example_hash != e.example_hash


def test_adding_an_example_keeps_every_other_examples_samples_comparable():
    before = {e.id: e.example_hash for e in EXAMPLES}
    extra = dataclasses.replace(BY_ID["CRT01"], id="CRT99")
    after = {e.id: e.example_hash for e in [*EXAMPLES, extra]}
    assert {k: after[k] for k in before} == before
    assert dataset_version([*EXAMPLES, extra]) != dataset_version(EXAMPLES)


def test_provider_only_examples_are_marked():
    assert {e.id for e in EXAMPLES if e.provider_only} == {"S00"}
    assert BY_ID["S00"].provider_only == "anthropic"


def test_examples_are_plain_data():
    json.dumps([dataclasses.asdict(e) for e in EXAMPLES])


# --- the EvalRecord -------------------------------------------------------------------------------


class NoStream(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


class Streams(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


CANARIES = ("CANARY-SEARCH-RESULT", "CANARY-LOOKUP-RESULT")


@tool("search_arxiv")
def search(query: str, max_results: int = 5) -> str:
    """Stub."""
    return "[1] CANARY-SEARCH-RESULT http://arxiv.org/abs/2309.15217"


@tool("verify_citation")
def verify(arxiv_id: str) -> str:
    """Stub."""
    return f"FOUND: {arxiv_id} - CANARY-LOOKUP-RESULT"


SEARCH = AIMessage(content="", tool_calls=[{"name": "search_arxiv", "args": {"query": "ragas"}, "id": "c1", "type": "tool_call"}])
NOTES = "- Findings: RAGAS (http://arxiv.org/abs/2309.15217)\n- Sources: s\n- Gaps: none"


def streamed_run():
    """A whole run with the Critic, through astream_run, with the final state
    read back from the checkpointer - the path E2's runner will take."""
    g = build_multi_agent_graph(
        researcher_model=NoStream(responses=[SEARCH, AIMessage(content=NOTES)]),
        writer_model=Streams(messages=iter([AIMessage(content="RAGAS evaluates RAG [2309.15217].")])),
        critic_model=NoStream(responses=[AIMessage(content="APPROVE")]), enable_critic=True, max_revisions=0,
        tools=[search], critic_tools=[verify], routing="fixed", memory_strategy="none",
        checkpointer=MemorySaver(),
    )
    config = {"configurable": {"thread_id": "eval"}}

    async def run():
        return [e async for e in astream_run(g, multi_agent_turn_input("Q"), config)]

    events = asyncio.run(run())
    return g.get_state(config).values, events


def test_private_keys_come_from_the_agents_schemas():
    assert {"research_messages", "critic_messages", "lookups"} <= private_keys()
    assert not EVAL_FIELDS & private_keys()


def test_a_real_runs_eval_record_carries_nothing_private():
    state, events = streamed_run()
    record = build_eval_record(state, events=events, tokens=0, sdk_retries=0)
    text = json.dumps(record, default=str)
    for leak in (*private_keys(), *CANARIES):
        assert leak not in text, leak
    # ...while carrying what evaluators need.
    assert record["ending"] == {"type": "done"}
    assert record["verdict"] == "approve"
    assert record["citation_checks"] == [{"arxiv_id": "2309.15217", "status": "found"}]
    assert [e["routed_to"] for e in record["supervisor_log"]][:2] == ["researcher", "writer"]
    assert record["tools"] == [{"agent": "researcher", "name": "search_arxiv", "arg": "ragas"},
                               {"agent": "critic", "name": "verify_citation", "arg": "2309.15217"}]
    assert record["answer"] == "RAGAS evaluates RAG [2309.15217]."
    assert set(record) <= EVAL_FIELDS


def test_private_keys_in_the_state_are_dropped_not_passed_on():
    """If someone hands it a subgraph's internal state by mistake."""
    state = {"question": "Q", "draft": "d", **{k: "CANARY-PRIVATE" for k in private_keys()},
             "supervisor_log": [{"step": 1, "routed_to": "writer", "secret": "CANARY-PRIVATE"}],
             "critic_interventions": [{"node": "critic", "kind": "hint_retry", "raw": "CANARY-PRIVATE"}]}
    record = build_eval_record(state)
    assert "CANARY-PRIVATE" not in json.dumps(record)
    assert record["supervisor_log"] == [{"step": 1, "routed_to": "writer"}]
    assert record["interventions"] == [{"node": "critic", "kind": "hint_retry"}]


def test_raw_astream_events_are_refused():
    with pytest.raises(TypeError, match="never raw astream_events"):
        build_eval_record({}, events=[{"event": "on_chain_end", "data": {"output": {}}}])


def test_the_unit_critic_path_carries_nothing_private():
    from research_copilot.agents.critic import build_critic

    critic = build_critic(model=NoStream(responses=[AIMessage(content="APPROVE")]), tools=[verify])
    out = critic.invoke({"question": "Q", "draft": "RAGAS [2309.15217]", "research_notes": NOTES})
    record = build_eval_record(out)
    assert "CANARY-LOOKUP-RESULT" not in json.dumps(record)
    assert record["verdict"] == "approve" and record["ending"] == {"type": "done"}


@pytest.mark.parametrize("last, ending", [
    ({"type": "stopped", "reason": "quota_exhausted", "detail": "used 199939 of 200000"},
     {"type": "stopped", "reason": "quota_exhausted", "detail": "used 199939 of 200000"}),
    ({"type": "error", "kind": "rate_limited", "detail": "RateLimitError: ..."},
     {"type": "error", "kind": "rate_limited", "detail": "RateLimitError: ..."}),
    ({"type": "paused", "payload": {"draft": "d"}}, {"type": "paused"}),
])
def test_the_ending_is_recorded_for_e2s_infrastructure_split(last, ending):
    record = build_eval_record({}, events=[{"type": "node", "phase": "start"}, last])
    assert record["ending"] == ending


def test_unknown_runner_fields_are_refused():
    with pytest.raises(TypeError, match="unknown runner fields"):
        build_eval_record({}, raw_trace=[])
