"""Phase 7 Part D: who handles which failure - tested against REAL failures only.

Every error here is a real provider response body from Phase 7's live runs
(tests/fixtures/provider_errors.json), served through the REAL Groq SDK over a
mock HTTP transport. So these tests exercise the SDK's actual retry logic, not
a model of it. No failure mode is invented: kinds that were never observed
(5xx, timeouts, refusals) are deliberately absent, and stay with the SDK's
default handling (the README's ownership table).
"""

import asyncio
import json
import logging
from pathlib import Path

import httpx
import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool

import groq._base_client as groq_base
from research_copilot import cli
from research_copilot.agents.critic import build_critic
from research_copilot.agents.supervisor import SupervisorDecision, make_supervisor
from research_copilot.models import (
    AsyncQuotaAwareTransport,
    ModelNotAvailable,
    QuotaAwareTransport,
    check_model_available,
    get_chat_model,
    get_fallback_model,
)
from research_copilot.resilience import (
    INVALID_TOOL,
    MODEL_NOT_FOUND,
    QUOTA_EXHAUSTED,
    RATE_LIMITED,
    TOO_LARGE,
    QuotaExhausted,
    classify,
    invoke_with_recovery,
)

FIXTURES = json.loads((Path(__file__).parent / "fixtures" / "provider_errors.json").read_text())
REAL = {k: v for k, v in FIXTURES.items() if not k.startswith("_")}


@pytest.fixture(autouse=True)
def groq_env(monkeypatch):
    monkeypatch.setenv("RESEARCH_COPILOT_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_not_real")
    # Do not actually wait out the real retry-after headers (13 s for row A).
    monkeypatch.setattr(groq_base.time, "sleep", lambda s: None)


def serving(name: str, *, calls: list, then: dict | None = None):
    """A mock transport serving the named real fixture; `then` (an OpenAI-style
    completion) after the first response, to model 'fails, then works'."""
    fixture = REAL[name]

    def handler(request):
        calls.append(request)
        if then is not None and len(calls) > 1:
            return httpx.Response(200, json=then)
        return httpx.Response(fixture["status"], json=fixture["body"], headers=fixture.get("headers", {}))

    return httpx.MockTransport(handler)


def groq_model(name: str, calls: list, **kwargs):
    """The project's real ChatGroq (get_chat_model), wired to a fixture."""
    model = get_chat_model()
    model.client._client._client = httpx.Client(transport=QuotaAwareTransport(serving(name, calls=calls, **kwargs)))
    return model


def raised_by(name: str) -> tuple[Exception, int]:
    calls: list = []
    with pytest.raises(Exception) as info:
        groq_model(name, calls).invoke("hi")
    return info.value, len(calls)


# --- the fixtures themselves ---------------------------------------------------------------


def test_fixtures_are_real_and_redacted():
    text = (Path(__file__).parent / "fixtures" / "provider_errors.json").read_text()
    assert "org_REDACTED" in text and "org_01m" not in text
    assert all(f["source"] for f in REAL.values())
    assert set(REAL) == {
        "groq_429_tokens_per_day", "groq_429_tokens_per_minute", "groq_413_request_too_large",
        "groq_400_invented_tool", "groq_400_tool_choice_none_ignored", "groq_404_model_not_found",
    }


# --- classification: one row per real failure -------------------------------------------------


@pytest.mark.parametrize("name, kind", [
    ("groq_429_tokens_per_day", QUOTA_EXHAUSTED),
    ("groq_429_tokens_per_minute", RATE_LIMITED),
    ("groq_413_request_too_large", TOO_LARGE),
    ("groq_400_invented_tool", INVALID_TOOL),
    ("groq_400_tool_choice_none_ignored", INVALID_TOOL),
    ("groq_404_model_not_found", MODEL_NOT_FOUND),
])
def test_each_real_failure_is_classified_into_its_row(name, kind):
    exc, _ = raised_by(name)
    assert classify(exc) == kind


def test_the_413_shares_its_code_with_the_429s_so_classification_cannot_use_the_code():
    """The trap the real bodies exposed: 413 says `rate_limit_exceeded` too."""
    codes = {REAL[n]["body"]["error"]["code"] for n in
             ("groq_413_request_too_large", "groq_429_tokens_per_day", "groq_429_tokens_per_minute")}
    assert codes == {"rate_limit_exceeded"}
    assert classify(raised_by("groq_413_request_too_large")[0]) == TOO_LARGE


# --- the SDK layer: before and after ------------------------------------------------------------


def test_before_part_d_the_sdk_sent_seven_requests_for_one_daily_limit_call():
    """Pinned as the reason: max_retries=6 and a plain transport. Every retry
    is guaranteed to fail (retry-after is ~48 minutes) and spends a request of
    the 1,000-per-day limit."""
    from langchain_groq import ChatGroq

    calls: list = []
    model = ChatGroq(model="openai/gpt-oss-120b", api_key="x", max_retries=6,
                     http_client=httpx.Client(transport=serving("groq_429_tokens_per_day", calls=calls)))
    with pytest.raises(Exception):
        model.invoke("hi")
    assert len(calls) == 7


@pytest.mark.parametrize("name, requests", [
    ("groq_429_tokens_per_day", 1),      # row B: the SDK no longer touches it
    ("groq_429_tokens_per_minute", 7),   # row A: the SDK's own 6 retries (calibrated in Part C)
    ("groq_413_request_too_large", 1),   # row C: never an SDK retry
    ("groq_400_invented_tool", 1),       # row D
    ("groq_404_model_not_found", 1),     # row F
])
def test_after_part_d_each_row_gets_exactly_its_owners_retries(name, requests):
    assert raised_by(name)[1] == requests


def test_the_async_path_does_the_same_for_part_c():
    calls: list = []
    model = get_chat_model()
    model.async_client._client._client = httpx.AsyncClient(
        transport=AsyncQuotaAwareTransport(serving("groq_429_tokens_per_day", calls=calls)))
    with pytest.raises(Exception) as info:
        asyncio.run(model.ainvoke("hi"))
    assert len(calls) == 1 and classify(info.value) == QUOTA_EXHAUSTED


def test_sdk_retries_are_counted_for_row_a():
    from research_copilot.live_check.runner import _SdkRetryCounter

    with _SdkRetryCounter() as counter:
        with pytest.raises(Exception):
            groq_model("groq_429_tokens_per_minute", []).invoke("hi")
    assert counter.count == 6


# --- the call level: fallback, or a clean stop ------------------------------------------------------


def test_daily_quota_without_a_fallback_raises_quota_exhausted():
    with pytest.raises(QuotaExhausted):
        invoke_with_recovery(groq_model("groq_429_tokens_per_day", []), [HumanMessage(content="hi")],
                             recoverable=lambda e: False, note="", where="t")


def test_daily_quota_with_a_fallback_answers_from_the_fallback_and_records_it():
    seen = []
    fallback = RunnableLambda(lambda msgs: AIMessage(content="from the fallback"))
    result, attempts = invoke_with_recovery(
        groq_model("groq_429_tokens_per_day", []), [HumanMessage(content="hi")],
        recoverable=lambda e: False, note="", where="t",
        fallback=lambda: (fallback, [HumanMessage(content="hi")]),
        on_intervention=seen.append,
    )
    assert result.text == "from the fallback" and attempts == 2
    assert [s["kind"] for s in seen] == ["fallback_model"]


def test_a_per_minute_429_that_outlasts_the_sdk_is_not_mistaken_for_a_spent_quota():
    """Row A never reaches the fallback: after the SDK's retries it propagates,
    rather than switching models over a limit that resets in seconds."""
    used_fallback = []
    with pytest.raises(Exception) as info:
        invoke_with_recovery(groq_model("groq_429_tokens_per_minute", []), [HumanMessage(content="hi")],
                             recoverable=lambda e: False, note="", where="t",
                             fallback=lambda: used_fallback.append(1))
    assert not isinstance(info.value, QuotaExhausted) and used_fallback == []


# --- the agents --------------------------------------------------------------------------------------


class Fake(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


@tool("verify_citation")
def verify(arxiv_id: str) -> str:
    """Stub."""
    return f"FOUND: {arxiv_id} - a paper"


def test_critic_switches_to_the_fallback_model_on_a_spent_quota():
    critic = build_critic(model=groq_model("groq_429_tokens_per_day", []), tools=[verify],
                          fallback_model=Fake(responses=[AIMessage(content="APPROVE")]))
    out = critic.invoke({"question": "Q", "draft": "RAGAS [2309.15217]", "research_notes": "n"})
    assert out["verdict"] == "approve"
    assert [i["kind"] for i in out["critic_interventions"]] == ["fallback_model"]


def test_supervisor_does_not_swallow_a_spent_quota_into_fixed_policy():
    supervisor = make_supervisor(model=groq_model("groq_429_tokens_per_day", []))
    with pytest.raises(QuotaExhausted):
        supervisor({"question": "Q", "dispatches": {}, "supervisor_log": []})


def test_supervisor_fallback_uses_the_fallback_models_own_structured_output_method():
    asked = []

    class FallbackModel:
        def with_structured_output(self, schema, **kwargs):
            asked.append(kwargs)
            return RunnableLambda(lambda _: SupervisorDecision(rationale="r", next="researcher"))

    supervisor = make_supervisor(model=groq_model("groq_429_tokens_per_day", []), fallback_model=FallbackModel())
    update = supervisor({"question": "Q", "dispatches": {}, "supervisor_log": []})
    assert update["next_agent"] == "researcher"
    assert asked == [{"method": "json_schema"}]   # resolved for the fallback, not the primary
    assert [i["kind"] for i in update["supervisor_interventions"]] == ["fallback_model"]


def test_a_bug_in_the_supervisors_own_code_is_no_longer_hidden_as_fixed_policy(monkeypatch):
    """Part D found it: the Supervisor's error handling wrapped its view-building
    code, so a formatting bug became a silent fixed_policy route."""
    monkeypatch.setattr("research_copilot.agents.supervisor.render_supervisor_view",
                        lambda *a, **k: (_ for _ in ()).throw(KeyError("tokens_before")))
    supervisor = make_supervisor(model=Fake(responses=[AIMessage(content="unused")]))
    with pytest.raises(KeyError):
        supervisor({"question": "Q", "dispatches": {}, "supervisor_log": []})


# --- the run level --------------------------------------------------------------------------------


def test_cli_stops_cleanly_on_a_spent_quota(monkeypatch, capsys):
    monkeypatch.setattr("research_copilot.agents.researcher.get_chat_model",
                        lambda **kw: groq_model("groq_429_tokens_per_day", []))
    monkeypatch.setattr("research_copilot.agents.supervisor.get_chat_model",
                        lambda **kw: groq_model("groq_429_tokens_per_day", []))
    code = cli.main(["multi-agent", "Q", "--checkpointer", "none", "--memory", "none"])
    _, err = capsys.readouterr()
    assert code == 1
    assert "daily quota is exhausted" in err and "stopped rather than degraded" in err
    assert "used 199939 of 200000" in err
    assert "Traceback" not in err


def test_preflight_fails_fast_on_a_model_the_key_does_not_have(monkeypatch):
    class Models:
        def list(self):
            class Page:
                data = [type("M", (), {"id": "openai/gpt-oss-120b"})()]
            return Page()

    monkeypatch.setattr("groq.Groq", lambda **kw: type("G", (), {"models": Models()})())
    check_model_available(provider="groq", model_name="openai/gpt-oss-120b")  # fine
    with pytest.raises(ModelNotAvailable, match="llama-3.3-70b-versatile"):
        check_model_available(provider="groq", model_name="llama-3.3-70b-versatile")


def test_fallback_model_is_configured_as_provider_colon_model(monkeypatch):
    from langchain_groq import ChatGroq

    assert get_fallback_model() is None
    monkeypatch.setenv("RESEARCH_COPILOT_FALLBACK_MODEL", "groq:openai/gpt-oss-20b")
    fallback = get_fallback_model()
    assert isinstance(fallback, ChatGroq) and fallback.model_name == "openai/gpt-oss-20b"
    monkeypatch.setenv("RESEARCH_COPILOT_FALLBACK_MODEL", "gpt-oss-20b")
    with pytest.raises(RuntimeError, match="provider:model"):
        get_fallback_model()


def test_sdk_retry_count_defaults_to_six_and_is_configurable(monkeypatch):
    """Part C, live: 2 retries were not always enough for per-minute 429s under
    a real streamed run (the SDK needed 10 over one run), so the default is 6.
    Safe only because daily-limit 429s never reach the SDK's retry logic -
    pinned by the transport tests above. RESEARCH_COPILOT_SDK_MAX_RETRIES
    overrides it."""
    assert get_chat_model().max_retries == 6
    monkeypatch.setenv("RESEARCH_COPILOT_SDK_MAX_RETRIES", "2")
    assert get_chat_model().max_retries == 2
    monkeypatch.setenv("RESEARCH_COPILOT_SDK_MAX_RETRIES", "6")
    calls: list = []
    with pytest.raises(Exception):
        groq_model("groq_429_tokens_per_minute", calls).invoke("hi")
    assert len(calls) == 7
