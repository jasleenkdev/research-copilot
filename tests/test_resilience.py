"""Phase 7 (Part D, pulled forward): recovering from a model call to a tool that
does not exist - the failure Groq's gpt-oss-120b produced live in A1.

Offline: a fake model raises the provider's error on cue. What these pin down
is the recovery shape - one retry with a hint, then a degraded result instead
of a crash - for both agents with a tool loop.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool

from research_copilot.agents.critic import build_critic
from research_copilot.agents.researcher import INVALID_TOOL_NOTE, RESEARCHER_SYSTEM_PROMPT, build_researcher
from research_copilot.agents.supervisor import supervisor_system_prompt
from research_copilot.resilience import ModelCallFailure, invoke_with_recovery, is_invalid_tool_call

GROQ_ERROR = (
    "Error code: 400 - {'error': {'message': \"Tool call validation failed: attempted to call "
    "tool 'open_file' which was not in request.tools\", 'code': 'tool_use_failed'}}"
)


class ScriptedModel:
    """Each script item is a reply (AIMessage / str) or an Exception to raise.
    Records every request, so tests can see the hint arrive."""

    def __init__(self, *script):
        self.script = list(script)
        self.requests = []

    def bind_tools(self, tools, **kwargs):
        return self

    def invoke(self, messages, *args, **kwargs):
        self.requests.append(list(messages))
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item if isinstance(item, AIMessage) else AIMessage(content=item)


def search_call(query, call_id):
    return AIMessage(content="", tool_calls=[{"name": "search_arxiv", "args": {"query": query}, "id": call_id, "type": "tool_call"}])


@tool("search_arxiv")
def search(query: str, max_results: int = 5) -> str:
    """Stub."""
    return f"[1] Paper about {query}\n    URL: http://arxiv.org/abs/2309.15217"


@tool("verify_citation")
def verify(arxiv_id: str) -> str:
    """Stub."""
    return f"FOUND: {arxiv_id} - a paper"


def researcher_input():
    return {"question": "Q", "mode": "live-search", "messages": [HumanMessage(content="Q")]}


# --- the helper ---------------------------------------------------------------------------


def test_recognises_the_groq_invalid_tool_error():
    assert is_invalid_tool_call(RuntimeError(GROQ_ERROR))
    assert not is_invalid_tool_call(RuntimeError("Error code: 429 - rate limited"))


def test_one_retry_with_the_note_after_the_system_messages():
    model = ScriptedModel(RuntimeError(GROQ_ERROR), "ok")
    request = [SystemMessage(content="rules"), HumanMessage(content="Q")]
    result, attempts = invoke_with_recovery(model, request, recoverable=is_invalid_tool_call, note="HINT", where="t")
    assert (result.text, attempts) == ("ok", 2)
    retried = model.requests[1]
    assert [m.type for m in retried] == ["system", "system", "human"]
    assert retried[1].content == "HINT"


def test_two_failures_return_a_failure_instead_of_raising():
    model = ScriptedModel(RuntimeError(GROQ_ERROR), RuntimeError(GROQ_ERROR))
    result, attempts = invoke_with_recovery(model, [HumanMessage(content="Q")], recoverable=is_invalid_tool_call, note="n", where="t")
    assert isinstance(result, ModelCallFailure) and attempts == 2
    assert "tool_use_failed" in result.error


def test_non_recoverable_errors_still_raise():
    model = ScriptedModel(RuntimeError("GROQ_API_KEY is not set"))
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        invoke_with_recovery(model, [HumanMessage(content="Q")], recoverable=is_invalid_tool_call, note="n", where="t")


# --- the Researcher -----------------------------------------------------------------------


def test_researcher_recovers_after_one_invented_tool_call():
    model = ScriptedModel(
        search_call("ragas", "1"),
        RuntimeError(GROQ_ERROR),          # tries open_file after the search
        "- Findings: X (http://arxiv.org/abs/2309.15217)\n- Sources: ...\n- Gaps: none",
    )
    out = build_researcher(model=model, tools=[search]).invoke(researcher_input())
    assert out["research_outcome"] == "findings"
    assert out["research_notes"].startswith("- Findings")
    assert out["research_iterations"] == 3  # the failed attempt counts: it spent tokens
    assert any(m.content == INVALID_TOOL_NOTE for m in model.requests[2])


def test_researcher_that_fails_twice_hands_over_raw_results_instead_of_crashing():
    model = ScriptedModel(search_call("ragas", "1"), RuntimeError(GROQ_ERROR), RuntimeError(GROQ_ERROR))
    out = build_researcher(model=model, tools=[search]).invoke(researcher_input())
    assert out["research_outcome"] == "model_error"
    assert "model failed before it wrote up" in out["research_notes"]
    assert "Paper about ragas" in out["research_notes"]


def test_researcher_prompt_says_search_is_the_only_tool():
    assert "search_arxiv is your only tool" in RESEARCHER_SYSTEM_PROMPT


# --- the Critic ---------------------------------------------------------------------------


def critic_input():
    return {"question": "Q", "draft": "RAGAS [2309.15217]", "research_notes": "notes"}


def test_critic_recovers_after_one_invented_tool_call():
    model = ScriptedModel(RuntimeError(GROQ_ERROR), "APPROVE")
    out = build_critic(model=model, tools=[verify]).invoke(critic_input())
    assert out["verdict"] == "approve"
    assert out["critic_budget"]["used"] == 2
    assert out["citation_checks"] == [{"arxiv_id": "2309.15217", "status": "found"}]


def test_critic_that_fails_twice_is_incomplete_not_rejected():
    """Phase 7: "did not finish" is not "rejected" (E2E01). Still not approved."""
    model = ScriptedModel(RuntimeError(GROQ_ERROR), RuntimeError(GROQ_ERROR))
    out = build_critic(model=model, tools=[verify]).invoke(critic_input())
    assert out["verdict"] == "incomplete"
    assert "could not finish its review" in out["critique"]


# --- the Supervisor's premise-checking instruction ------------------------------------------


def test_supervisor_prompt_asks_it_to_check_a_critiques_premise():
    prompt = supervisor_system_prompt(("researcher", "writer", "critic"))
    assert "A critique can be wrong" in prompt
    assert "establish whether the thing exists at all" in prompt


# --- the Researcher that never stops searching (A1, day two) ------------------------------


class BindRecordingModel(ScriptedModel):
    """ScriptedModel that also records how it was bound, per call."""

    def __init__(self, *script):
        super().__init__(*script)
        self.bound_with = []
        self._pending_kwargs = {}

    def bind_tools(self, tools, **kwargs):
        outer = self

        class Bound:
            def invoke(self_inner, messages, *a, **k):
                outer.bound_with.append(kwargs.get("tool_choice"))
                return outer.invoke(messages)

        return Bound()


def test_every_call_is_told_its_remaining_budget():
    model = BindRecordingModel(search_call("a", "1"), "- Findings: x\n- Sources: y\n- Gaps: none")
    build_researcher(model=model, tools=[search], max_iterations=4).invoke(researcher_input())
    first, second = ("\n".join(m.text for m in r if m.type == "system") for r in model.requests)
    assert "4 of 4" in first and "3 of 4" in second


def test_the_last_call_is_reserved_no_tools_results_as_text():
    model = BindRecordingModel(
        search_call("a", "1"), search_call("b", "2"),
        "- Findings: from the two searches\n- Sources: s\n- Gaps: none",
    )
    out = build_researcher(model=model, tools=[search], max_iterations=3).invoke(researcher_input())
    # Two calls through the tool-bound model; the third through the bare model.
    assert model.bound_with == [None, None]
    final_request = model.requests[-1]
    # No tool-call history in the final request - only plain-text results.
    assert not any(getattr(m, "tool_calls", None) for m in final_request)
    assert not any(m.type == "tool" for m in final_request)
    assert "you have no tools now" in final_request[-1].text
    assert "Paper about a" in final_request[-1].text and "Paper about b" in final_request[-1].text
    # The point of the fix: the cap ends in written notes, not raw results.
    assert out["research_outcome"] == "findings"
    assert out["research_notes"].startswith("- Findings: from the two searches")


def test_a_final_call_that_still_errors_gets_the_final_hint_then_raw_results():
    from research_copilot.agents.researcher import FINAL_RETRY_NOTE

    err = RuntimeError("Error code: 400 - Tool choice is none, but model called a tool - tool_use_failed")
    model = BindRecordingModel(search_call("a", "1"), err, err)
    out = build_researcher(model=model, tools=[search], max_iterations=2).invoke(researcher_input())
    assert any(m.content == FINAL_RETRY_NOTE for m in model.requests[-1] if m.type == "system")
    assert out["research_outcome"] == "model_error"
    assert "Paper about a" in out["research_notes"]


def test_a_model_that_answers_the_final_call_with_a_tool_call_is_not_run():
    """A bare model cannot really call a tool; a fake can. The reply's tool call
    is ignored (nothing routes to research_tools past the cap) and the budget
    fallback hands over the raw results."""
    model = BindRecordingModel(search_call("a", "1"), search_call("b", "2"))
    out = build_researcher(model=model, tools=[search], max_iterations=2).invoke(researcher_input())
    assert out["research_outcome"] == "budget_exhausted"
