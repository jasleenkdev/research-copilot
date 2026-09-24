"""Phase 7, A0: the provider switch and the live-verification harness, offline.

Nothing here calls Groq or Anthropic. What is tested is the harness's own
logic - that a scenario's checks detect what they claim to, that the runner
paces, resumes, stops on budget, and never reports an Anthropic-only item as
anything but PENDING on another provider. Whether the *models* pass is what the
live run is for.
"""

import itertools
import json

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool

from research_copilot import cli
from research_copilot.agents.supervisor import SupervisorDecision
from research_copilot.live_check import report, runner
from research_copilot.live_check import scenarios as sc
from research_copilot.live_check.scenarios import ANTHROPIC_ONLY_ITEMS, SCENARIOS, by_id
from research_copilot.models import get_chat_model, structured_output_method

# --- the provider switch --------------------------------------------------------------


def test_groq_provider_builds_chatgroq_with_the_configured_model(monkeypatch):
    from langchain_groq import ChatGroq

    monkeypatch.setenv("RESEARCH_COPILOT_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    model = get_chat_model()
    assert isinstance(model, ChatGroq)
    assert model.model_name == "openai/gpt-oss-120b"
    assert model.max_tokens == 4096
    assert get_chat_model(max_tokens=512).max_tokens == 512


def test_groq_without_a_key_fails_with_a_useful_message(monkeypatch):
    monkeypatch.setenv("RESEARCH_COPILOT_PROVIDER", "groq")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        get_chat_model()


def test_unknown_provider_is_refused(monkeypatch):
    monkeypatch.setenv("RESEARCH_COPILOT_PROVIDER", "openai")
    with pytest.raises(RuntimeError, match="Unknown provider"):
        get_chat_model()


def test_default_provider_is_still_anthropic(monkeypatch):
    monkeypatch.delenv("RESEARCH_COPILOT_PROVIDER", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    from langchain_anthropic import ChatAnthropic

    assert isinstance(get_chat_model(), ChatAnthropic)


def test_groq_json_schema_models_get_strict_mode():
    from langchain_anthropic import ChatAnthropic
    from langchain_groq import ChatGroq

    from research_copilot.models import structured_output_kwargs

    assert structured_output_kwargs(ChatGroq(model="openai/gpt-oss-120b", api_key="x")) == {"method": "json_schema", "strict": True}
    assert structured_output_kwargs(ChatGroq(model="llama-3.3-70b-versatile", api_key="x")) == {"method": "function_calling"}
    assert structured_output_kwargs(ChatAnthropic(model="claude-opus-5", api_key="x")) == {"method": "json_schema"}


def test_structured_output_method_per_provider(monkeypatch):
    from langchain_anthropic import ChatAnthropic
    from langchain_groq import ChatGroq

    assert structured_output_method(ChatGroq(model="llama-3.3-70b-versatile", api_key="x")) == "function_calling"
    assert structured_output_method(ChatGroq(model="openai/gpt-oss-120b", api_key="x")) == "json_schema"
    assert structured_output_method(ChatAnthropic(model="claude-opus-5", api_key="x")) == "json_schema"
    assert structured_output_method(object()) == "json_schema"


def test_supervisor_asks_the_model_module_which_method_to_use(monkeypatch):
    from research_copilot.agents.supervisor import make_supervisor

    asked = []

    class Model:
        def with_structured_output(self, schema, *, method, **kwargs):
            asked.append(method)
            return RunnableLambda(lambda _: SupervisorDecision(rationale="r", next="researcher"))

    monkeypatch.setattr("research_copilot.agents.supervisor.structured_output_kwargs", lambda m: {"method": "function_calling"})
    make_supervisor(model=Model())({"question": "Q", "dispatches": {}, "supervisor_log": []})
    assert asked == ["function_calling"]


def test_cli_provider_flag_sets_the_provider_for_the_process(monkeypatch, capsys):
    monkeypatch.delenv("RESEARCH_COPILOT_PROVIDER", raising=False)
    cli.main(["--provider", "groq", "live-check", "list", "--only", "P01"])
    import os

    assert os.environ["RESEARCH_COPILOT_PROVIDER"] == "groq"
    monkeypatch.delenv("RESEARCH_COPILOT_PROVIDER")


# --- scenario integrity -----------------------------------------------------------------


def test_scenario_ids_are_unique_and_every_kind_has_a_runner():
    ids = [s.id for s in SCENARIOS]
    assert len(ids) == len(set(ids))
    assert {s.kind for s in SCENARIOS} <= set(runner.RUNNERS)


def test_supervisor_scenarios_have_expectations_and_state():
    for s in SCENARIOS:
        if s.kind == "supervisor":
            assert s.expect["preferred"] in s.expect["acceptable"]
            assert "question" in s.inputs["state"]


def test_draft_edits_in_scenarios_actually_changed_the_text():
    """The scenario drafts are built with str.replace on GOOD_DRAFT; a replace
    that silently matched nothing would make the scenario test the good draft."""
    for sid in ("CRT03", "CRT04", "CRT06"):
        assert by_id()[sid].inputs["draft"] != sc.GOOD_DRAFT, sid
    assert by_id()["SUP01"].inputs["state"]["draft"] != sc.GOOD_DRAFT
    assert by_id()["SUP08"].inputs["state"]["draft"] != sc.GOOD_DRAFT
    assert sc.CLEAN_NOTES != sc.NOTES and "Gaps: none." in sc.CLEAN_NOTES
    assert by_id()["SUPC08"].inputs["state"]["draft"] != sc.CLEAN_DRAFT


def test_supervisor_scenarios_render_without_error():
    from research_copilot.agents.supervisor import AGENTS, default_dispatch_caps, render_supervisor_view

    caps = default_dispatch_caps(enable_critic=True, max_revisions=2)
    for s in SCENARIOS:
        if s.kind == "supervisor":
            assert "Question:" in render_supervisor_view(s.inputs["state"], caps, roster=AGENTS, max_revisions=2)


def test_scenarios_are_in_readme_priority_order():
    order = {"0": 0, "1": 1, "1g": 1, "2": 2, "4": 4, "5": 5, "6": 6, "7": 7}
    ranks = [order[s.item] for s in SCENARIOS]
    assert ranks == sorted(ranks)


# --- the runner, with fakes ---------------------------------------------------------------


class FakeEverything(FakeMessagesListChatModel):
    """One fake that can be every agent: scripted replies, tools ignored, and a
    scripted structured output for the Supervisor."""

    decisions: list = []

    def bind_tools(self, tools, **kwargs):
        return self

    def with_structured_output(self, schema, **kwargs):
        # One shared script across every Supervisor built from this fake:
        # the runner builds a fresh Supervisor per scenario.
        decisions = self.decisions

        def reply(_):
            item = decisions.pop(0)
            if kwargs.get("include_raw"):
                return {"raw": AIMessage(content="", tool_calls=[{"name": "SupervisorDecision", "args": item.model_dump(), "id": "1", "type": "tool_call"}]),
                        "parsed": item, "parsing_error": None}
            return item

        return RunnableLambda(reply)


def cfg_with(tmp_path, *, replies=("x",), decisions=(), provider="groq", **kwargs):
    fake = FakeEverything(responses=[AIMessage(content=r) if isinstance(r, str) else r for r in replies],
                          decisions=list(decisions))

    @tool("verify_citation")
    def verify(arxiv_id: str) -> str:
        """Stub."""
        return f"FOUND: {arxiv_id} - a paper" if arxiv_id != "2401.99999" else f"NOT FOUND: {arxiv_id}"

    @tool("search_arxiv")
    def search(query: str, max_results: int = 5) -> str:
        """Stub."""
        return "[1] RAGAS\n    URL: http://arxiv.org/abs/2309.15217"

    clock = itertools.count(0, 1)
    sleeps = []
    return runner.RunConfig(
        provider=provider, model="fake", results_path=tmp_path / "results.jsonl",
        model_factory=lambda: fake, search_tool=search, verify_tool=verify,
        sleep=sleeps.append, clock=lambda: next(clock), **kwargs,
    ), fake, sleeps


def quiet(_):
    pass


def test_anthropic_only_item_is_pending_on_groq_and_never_run(tmp_path):
    cfg, _, _ = cfg_with(tmp_path)
    results = runner.run(cfg, [by_id()["S00"]], log=quiet)
    assert results["S00"]["status"] == "pending"
    assert "anthropic" in results["S00"]["reason"].lower()
    assert "S00" in {s.id for s in SCENARIOS if s.item in ANTHROPIC_ONLY_ITEMS}


def test_supervisor_scenario_scores_strict_and_lenient(tmp_path):
    cfg, _, _ = cfg_with(tmp_path, decisions=[SupervisorDecision(rationale="the Writer invented a source", next="researcher", researcher_brief="ARES")])
    r = runner.run(cfg, [by_id()["SUP02"]], log=quiet)["SUP02"]  # preferred writer, researcher acceptable
    assert r["checks"] == {"usable_output": True, "strict": False, "lenient": True, "brief_given": True}
    assert r["status"] == "fail"  # strict is a hard check
    assert r["observed"]["rationale"] == "the Writer invented a source"


def test_supervisor_scenario_passes_to_review_not_pass(tmp_path):
    """Right route, but the rationale still needs reading (item 3)."""
    cfg, _, _ = cfg_with(tmp_path, decisions=[SupervisorDecision(rationale="style only", next="writer")])
    assert runner.run(cfg, [by_id()["SUP06"]], log=quiet)["SUP06"]["status"] == "review"


def test_writer_checks_catch_a_citation_not_in_the_notes(tmp_path):
    cfg, _, _ = cfg_with(tmp_path, replies=["RAGAS [2309.15217] and ARES [2311.09476]."])
    r = runner.run(cfg, [by_id()["WRT01"]], log=quiet)["WRT01"]
    assert r["checks"]["cites_only_notes"] is False


def test_writer_gap_check_on_empty_notes(tmp_path):
    cfg, _, _ = cfg_with(tmp_path, replies=["I could not find supporting evidence for this."])
    r = runner.run(cfg, [by_id()["WRT02"]], log=quiet)["WRT02"]
    assert r["checks"] == {"non_empty": True, "cites_only_notes": True, "admits_gap": True}


def test_critic_scenario_reads_verdict_and_lookups(tmp_path):
    cfg, _, _ = cfg_with(tmp_path, replies=[
        AIMessage(content="", tool_calls=[{"name": "verify_citation", "args": {"arxiv_id": "2401.99999"}, "id": "1", "type": "tool_call"}]),
        "REJECT\n2401.99999 does not exist",
    ])
    r = runner.run(cfg, [by_id()["CRT02"]], log=quiet)["CRT02"]
    assert r["checks"] == {"verdict": True, "used_verify_citation": True, "2401.99999_rejected_by_lookup": True}


def test_critic_lookup_failure_scenario_uses_an_error_stub(tmp_path):
    cfg, _, _ = cfg_with(tmp_path, replies=[
        AIMessage(content="", tool_calls=[{"name": "verify_citation", "args": {"arxiv_id": "2309.15217"}, "id": "1", "type": "tool_call"}]),
        "APPROVE",
    ])
    r = runner.run(cfg, [by_id()["CRT05"]], log=quiet)["CRT05"]
    assert r["observed"]["citation_checks"] == [{"arxiv_id": "2309.15217", "status": "error"}]
    assert r["status"] == "pass"


def test_researcher_check_catches_an_invented_id(tmp_path):
    cfg, _, _ = cfg_with(tmp_path, replies=[
        AIMessage(content="", tool_calls=[{"name": "search_arxiv", "args": {"query": "ragas"}, "id": "1", "type": "tool_call"}]),
        "- Findings: X (http://arxiv.org/abs/2309.15217)\n- Findings: Y (http://arxiv.org/abs/2999.00001)\n- Sources: ...\n- Gaps: none",
    ])
    r = runner.run(cfg, [by_id()["RES01"]], log=quiet)["RES01"]
    assert r["checks"]["no_invented_ids"] is False
    assert r["checks"]["has_findings_sources_gaps"] is True


def test_errors_are_recorded_not_raised(tmp_path):
    cfg, _, _ = cfg_with(tmp_path)
    cfg.model_factory = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    r = runner.run(cfg, [by_id()["WRT01"]], log=quiet)["WRT01"]
    assert r["status"] == "error" and "boom" in r["error"]


def test_a_rerun_skips_recorded_scenarios(tmp_path):
    cfg, _, _ = cfg_with(tmp_path, replies=["RAGAS [2309.15217]."] * 2)
    runner.run(cfg, [by_id()["WRT01"]], log=quiet)
    logs = []
    runner.run(cfg, [by_id()["WRT01"]], log=logs.append)
    assert logs == ["[skip] WRT01 already recorded (pass)"]
    assert len(cfg.results_path.read_text().splitlines()) == 1


def test_daily_budget_stops_between_scenarios(tmp_path):
    cfg, _, _ = cfg_with(tmp_path, daily_token_budget=100)
    logs = []
    runner.run(cfg, [by_id()["WRT01"]], log=logs.append)
    assert logs[0].startswith("[stop] daily token budget")
    assert not cfg.results_path.exists()


def test_pacing_waits_for_the_token_window():
    cfg = runner.RunConfig(provider="groq", model="m", results_path=None, tpm_limit=10_000,
                           sleep=lambda s: times.append(s), clock=lambda: now[0])
    times, now = [], [100.0]
    cfg.window.append((95.0, 9_000))

    def sleep(s):
        times.append(s)
        now[0] += s

    cfg.sleep = sleep
    waited = runner._pace(cfg, 3_000)
    assert waited > 0 and now[0] - 95.0 > 60


def test_report_marks_item_1_pending_and_scores_item_2(tmp_path):
    cfg, _, _ = cfg_with(tmp_path, decisions=[
        SupervisorDecision(rationale="evidence gap", next="researcher", researcher_brief="b"),
        SupervisorDecision(rationale="style", next="writer"),
    ])
    results = runner.run(cfg, [by_id()["S00"], by_id()["SUP01"], by_id()["SUP06"]], log=quiet)
    text = report.render(results, limits={"x-ratelimit-limit-tokens": "12000"})
    assert "## Item 1:" in text and "PENDING - needs a real Anthropic key" in text
    assert "strict 2/2, lenient 2/2" in text
    assert "evidence gap" in text  # item 3 prints the rationale
    assert "x-ratelimit-limit-tokens: 12000" in text
    assert "NOT YET RUN" in text


def test_errored_scenarios_are_retried_on_resume(tmp_path):
    cfg, _, _ = cfg_with(tmp_path, replies=["RAGAS [2309.15217]."])
    good_factory = cfg.model_factory
    cfg.model_factory = lambda: (_ for _ in ()).throw(RuntimeError("network down"))
    assert runner.run(cfg, [by_id()["WRT01"]], log=quiet)["WRT01"]["status"] == "error"

    cfg.model_factory = good_factory
    assert runner.run(cfg, [by_id()["WRT01"]], log=quiet)["WRT01"]["status"] == "pass"
    assert runner.load_results(cfg.results_path)["WRT01"]["status"] == "pass"


def test_live_check_run_refuses_to_start_without_a_key(monkeypatch, capsys, tmp_path):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    code = cli.main(["--provider", "groq", "live-check", "run", "--results", str(tmp_path / "r.jsonl")])
    _, err = capsys.readouterr()
    assert code == 1 and "GROQ_API_KEY" in err
    assert not (tmp_path / "r.jsonl").exists()
    monkeypatch.delenv("RESEARCH_COPILOT_PROVIDER", raising=False)


@pytest.mark.parametrize("text", [
    "I\u2019m sorry, but I wasn\u2019t able to locate any usable evidence on this.",
    "I could not find supporting evidence for this.",
    "No evidence was found in the research notes.",
])
def test_admits_gap_recognises_real_phrasings(text):
    assert runner.admits_gap(text)


def test_admits_gap_is_not_fooled_by_a_confident_answer():
    assert not runner.admits_gap("RAGAS measures faithfulness and relevance [2309.15217].")


def test_writer_must_cite_something_when_notes_have_sources(tmp_path):
    cfg, _, _ = cfg_with(tmp_path, replies=["RAG is evaluated with various metrics."])
    r = runner.run(cfg, [by_id()["WRT01"]], log=quiet)["WRT01"]
    assert r["checks"]["cites_something"] is False


def test_rescore_updates_recorded_writer_results_without_a_model(tmp_path):
    cfg, _, _ = cfg_with(tmp_path, replies=["I wasn\u2019t able to locate any usable evidence."])
    path = cfg.results_path
    runner.run(cfg, [by_id()["WRT02"]], log=quiet)
    # Simulate a record scored by an older, stricter check.
    old = runner.load_results(path)["WRT02"]
    old["checks"]["admits_gap"] = False
    old["status"] = "fail"
    with path.open("a") as f:
        f.write(json.dumps(old) + "\n")

    assert runner.rescore(path, log=quiet) == ["WRT02"]
    latest = runner.load_results(path)["WRT02"]
    assert latest["status"] == "pass" and latest["rescored"] is True
    assert latest["previous_checks"]["admits_gap"] is False
