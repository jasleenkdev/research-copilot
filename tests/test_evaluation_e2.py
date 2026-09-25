"""Phase 7 E2: samples, outcomes, scheduling and aggregation - offline.

What these pin down: a sample is recorded whatever happened; a rate limit,
spent quota or outage is infra (not scored, not counted), and a bug in our
code is a crash (recorded, then raised); single-agent runs record their
ending explicitly, tested with the real captured provider errors; the
scheduler fills evenly and resumes; a scripted 2-out-of-3 model comes out
flaky with the right interval; the store only grows. No test writes outside
tmp_path.
"""

import json
import urllib.error
from collections import Counter
from pathlib import Path

import httpx
import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel, GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

import groq._base_client as groq_base
from research_copilot.evaluation import samples as samples_mod
from research_copilot.evaluation.aggregate import differs, infra_rate, summarize, wilson
from research_copilot.evaluation.dataset import EXAMPLES
from research_copilot.evaluation.execute import TOOL_FAILURE_PREFIXES, ExecConfig, execute
from research_copilot.evaluation.samples import (
    CRASH,
    INFRA,
    MEASURED,
    NOT_BEHAVIOURAL,
    Crashed,
    SampleConfig,
    SampleStore,
    code_hash,
    collect,
    config_fingerprint,
    import_legacy,
    next_example,
    outcome_of,
)
from research_copilot.live_check.runner import writer_checks
from research_copilot.live_check.scenarios import by_id as scenario_by_id
from research_copilot.models import QuotaAwareTransport, get_chat_model

BY_ID = {e.id: e for e in EXAMPLES}
FIXTURES = json.loads((Path(__file__).parent / "fixtures" / "provider_errors.json").read_text())
CONFIG = {"provider": "test", "model": "fake", "code_hash": "pinned"}
FP = config_fingerprint(CONFIG)


class NoStream(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


class Streams(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def groq_serving(name, monkeypatch):
    """The project's real ChatGroq, answering with a real captured error body."""
    monkeypatch.setenv("RESEARCH_COPILOT_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_not_real")
    monkeypatch.setattr(groq_base.time, "sleep", lambda s: None)
    fixture = FIXTURES[name]
    model = get_chat_model()
    handler = lambda r: httpx.Response(fixture["status"], json=fixture["body"], headers=fixture.get("headers", {}))  # noqa: E731
    model.client._client._client = httpx.Client(transport=QuotaAwareTransport(httpx.MockTransport(handler)))
    return model


def exec_with(model):
    return ExecConfig(model_factory=lambda: model)


# --- the explicit error path for single-agent runs ------------------------------------------


def test_a_unit_run_cut_short_by_a_rate_limit_records_it_as_infra(monkeypatch):
    """Real per-minute 429 body, through the real SDK, into the Writer. No
    stream reports this; the unit path has to."""
    record, error = execute(BY_ID["WRT01"], exec_with(groq_serving("groq_429_tokens_per_minute", monkeypatch)))
    assert error is not None
    assert record["ending"]["type"] == "error" and record["ending"]["kind"] == "rate_limited"
    assert outcome_of(BY_ID["WRT01"], record) == (INFRA, "rate_limited")
    assert record["sdk_retries"] == 6      # the SDK tried first (row A)


def test_a_unit_run_on_a_spent_quota_is_stopped_not_failed(monkeypatch):
    monkeypatch.delenv("RESEARCH_COPILOT_FALLBACK_MODEL", raising=False)
    record, _ = execute(BY_ID["CRT01"], exec_with(groq_serving("groq_429_tokens_per_day", monkeypatch)))
    assert record["ending"]["type"] == "stopped" and record["ending"]["reason"] == "quota_exhausted"
    assert "used 199939 of 200000" in record["ending"]["detail"]
    assert outcome_of(BY_ID["CRT01"], record)[0] == INFRA


def test_a_rate_limit_the_supervisor_swallows_is_still_infra(monkeypatch):
    """Found while building E2. The Supervisor's handler turns the 429 into
    a fixed_policy route; the run ends normally, and on SUP01 the route is
    even lenient-correct. Only the logged rationale shows it."""
    record, error = execute(BY_ID["SUP01"], exec_with(groq_serving("groq_429_tokens_per_minute", monkeypatch)))
    assert error is None and record["ending"] == {"type": "done"}
    entry = record["supervisor_log"][-1]
    assert entry["override"] == "fallback to fixed policy"
    assert entry["routed_to"] in BY_ID["SUP01"].reference["acceptable_routes"]   # would have "passed"
    outcome, reason = outcome_of(BY_ID["SUP01"], record)
    assert outcome == INFRA and "rate_limited became a fixed_policy route" in reason


def test_an_unusable_supervisor_reply_is_behaviour_not_infra():
    """The other reason for fixed_policy - the model's output - stays measured."""
    record, _ = execute(BY_ID["SUP01"], exec_with(NoStream(responses=[AIMessage(content="not json")])))
    assert record["supervisor_log"][-1]["override"] == "fallback to fixed policy"
    assert outcome_of(BY_ID["SUP01"], record) == (MEASURED, "")


def test_a_bug_in_our_code_during_a_unit_run_is_a_crash(monkeypatch):
    monkeypatch.setattr("research_copilot.agents.writer.unsupported_citations",
                        lambda *a, **k: (_ for _ in ()).throw(KeyError("our bug")))
    record, error = execute(BY_ID["WRT01"], exec_with(Streams(messages=iter([AIMessage(content="d")]))))
    assert isinstance(error, KeyError)
    assert outcome_of(BY_ID["WRT01"], record) == (CRASH, "other")


# --- tools: what the recorder keeps -----------------------------------------------------------


def test_the_failure_prefixes_are_our_tools_real_failure_strings(monkeypatch):
    from research_copilot.tools import arxiv, citations

    def down(*a, **k):
        raise urllib.error.URLError("timed out")

    monkeypatch.setattr(arxiv, "_fetch", down)
    monkeypatch.setattr(citations, "_fetch_by_id", down)
    assert arxiv.search_arxiv.invoke({"query": "x"}).startswith(TOOL_FAILURE_PREFIXES)
    assert citations.check_arxiv_id("2309.15217").startswith(TOOL_FAILURE_PREFIXES)


CANARY = "CANARY-RAW-SEARCH-TEXT"


@tool("search_arxiv")
def canary_search(query: str, max_results: int = 5) -> str:
    """Stub."""
    return f"[1] {CANARY} http://arxiv.org/abs/2309.15217"


@tool("search_arxiv")
def arxiv_down(query: str, max_results: int = 5) -> str:
    """Stub."""
    return "arXiv search failed (network error: timed out). Answer without new sources and say so."


SEARCH = AIMessage(content="", tool_calls=[{"name": "search_arxiv", "args": {"query": "ragas"}, "id": "c1", "type": "tool_call"}])
NOTES = "- Findings: RAGAS (http://arxiv.org/abs/2309.15217)\n- Sources: s\n- Gaps: none"


def test_the_tool_recorder_keeps_ids_and_status_never_the_text():
    cfg = ExecConfig(model_factory=lambda: NoStream(responses=[SEARCH, AIMessage(content=NOTES)]),
                     search_tool=canary_search)
    record, _ = execute(BY_ID["RES01"], cfg)
    assert record["tools"] == [{"agent": "researcher", "name": "search_arxiv", "arg": "ragas",
                                "status": "ok", "result_ids": ["2309.15217"]}]
    assert CANARY not in json.dumps(record)


def test_the_tool_recorder_stripping_is_proven_by_injection():
    """E1's lesson: the real run above passes only if the recorder strips the
    text. Hand the recorder raw text directly, and check it is gone."""
    from research_copilot.evaluation.execute import ToolRecorder
    from research_copilot.evaluation.record import build_eval_record

    rec = ToolRecorder("researcher")
    rec.on_tool_start({"name": "search_arxiv"}, "", run_id="r1", inputs={"query": "q", "raw": CANARY})
    rec.on_tool_end(f"{CANARY} 2309.15217", run_id="r1")
    assert CANARY not in json.dumps(build_eval_record({}, tool_calls=rec.calls))


def test_an_arxiv_outage_is_infra_but_crt05s_injected_one_is_not():
    cfg = ExecConfig(model_factory=lambda: NoStream(responses=[SEARCH, AIMessage(content="NOTHING NEW")]),
                     search_tool=arxiv_down)
    record, _ = execute(BY_ID["RES01"], cfg)
    assert outcome_of(BY_ID["RES01"], record) == (INFRA, "search_arxiv failed (researcher)")
    crt05, _ = execute(BY_ID["CRT05"], exec_with(NoStream(responses=[AIMessage(content="APPROVE")])))
    assert crt05["tools"][0]["status"] == "failed"
    assert outcome_of(BY_ID["CRT05"], crt05) == (MEASURED, "")


def test_the_fallback_model_answering_is_not_a_sample_of_the_model_under_test():
    record = {"ending": {"type": "done"}, "interventions": [{"node": "writer", "kind": "fallback_model"}]}
    assert outcome_of(BY_ID["WRT01"], record)[0] == INFRA


def test_an_e2e_sample_goes_through_the_stream_with_its_final_state(monkeypatch):
    monkeypatch.setattr("research_copilot.multi_agent_graph.get_chat_model", lambda **k: None, raising=False)
    from research_copilot import multi_agent_graph

    real = multi_agent_graph.build_multi_agent_graph

    def fake_build(**kwargs):
        kwargs.pop("model", None)
        return real(researcher_model=NoStream(responses=[SEARCH, AIMessage(content=NOTES)]),
                    writer_model=Streams(messages=iter([AIMessage(content="RAGAS [2309.15217].")])),
                    routing="fixed", **kwargs)

    monkeypatch.setattr(multi_agent_graph, "build_multi_agent_graph", fake_build)
    record, error = execute(BY_ID["E2E02"], ExecConfig(search_tool=canary_search))
    assert error is None and record["ending"] == {"type": "done"}
    assert record["answer"] == "RAGAS [2309.15217]."
    assert record["dispatches"] == {"researcher": 1, "writer": 1}
    assert record["tools"][0]["result_ids"] == ["2309.15217"]
    assert CANARY not in json.dumps(record)
    assert outcome_of(BY_ID["E2E02"], record) == (MEASURED, "")


# --- scheduling ---------------------------------------------------------------------------------


def scripted(outcomes: list[str], tokens: int = 100):
    """A stand-in for execute(): one scripted ending per call."""
    endings = {MEASURED: {"type": "done"}, INFRA: {"type": "error", "kind": "rate_limited"},
               CRASH: {"type": "error", "kind": "other", "detail": "KeyError: our bug"}}
    queue = list(outcomes)
    seen = []

    def run(e, cfg):
        seen.append(e.id)
        outcome = queue.pop(0)
        return {"ending": endings[outcome], "tokens": tokens}, (KeyError("our bug") if outcome == CRASH else None)

    run.seen = seen
    return run


def cfg(**kw):
    """A clock that moves only when the pacing sleeps, so no test waits."""
    now = [0.0]

    def sleep(seconds):
        now[0] += seconds

    return SampleConfig(provider="groq", model="fake", config=CONFIG, sleep=sleep, clock=lambda: now[0], **kw)


THREE = [BY_ID["SUP01"], BY_ID["CRT01"], BY_ID["WRT01"]]


def test_the_scheduler_fills_evenly_in_priority_order(monkeypatch, tmp_path):
    run = scripted([MEASURED] * 9)
    monkeypatch.setattr(samples_mod, "execute", run)
    collect(THREE, cfg(), SampleStore(tmp_path / "s.jsonl"), log=lambda m: None)
    assert run.seen == ["SUP01", "CRT01", "WRT01"] * 3


def test_infra_stops_the_session_does_not_count_and_resume_continues(monkeypatch, tmp_path):
    store = SampleStore(tmp_path / "s.jsonl")
    monkeypatch.setattr(samples_mod, "execute", scripted([MEASURED, INFRA]))
    collect(THREE, cfg(), store, log=lambda m: None)
    assert [s["outcome"] for s in store.samples()] == [MEASURED, INFRA]
    assert store.counted(FP) == Counter({BY_ID["SUP01"].example_hash: 1})
    # Tomorrow: CRT01 is retried first, since its infra sample did not count.
    run = scripted([MEASURED] * 8)
    monkeypatch.setattr(samples_mod, "execute", run)
    collect(THREE, cfg(), store, log=lambda m: None)
    assert run.seen[:2] == ["CRT01", "WRT01"]
    assert all(v == 3 for v in store.counted(FP).values())


def test_a_crash_is_recorded_then_raised(monkeypatch, tmp_path):
    store = SampleStore(tmp_path / "s.jsonl")
    monkeypatch.setattr(samples_mod, "execute", scripted([MEASURED, CRASH]))
    with pytest.raises(Crashed) as info:
        collect(THREE, cfg(), store, log=lambda m: None)
    assert isinstance(info.value.__cause__, KeyError)
    assert [s["outcome"] for s in store.samples()] == [MEASURED, CRASH]


def test_the_daily_budget_stops_between_samples(monkeypatch, tmp_path):
    monkeypatch.setattr(samples_mod, "execute", scripted([MEASURED] * 9, tokens=4000))
    store = SampleStore(tmp_path / "s.jsonl")
    logs = []
    collect(THREE, cfg(daily_token_budget=10_000), store, log=logs.append)
    assert len(store.samples()) == 2
    assert any(m.startswith("[stop] daily token budget") for m in logs)


def test_provider_only_examples_are_never_sampled_elsewhere():
    s00 = BY_ID["S00"]
    assert next_example([s00], Counter(), 3, "groq") is None
    assert next_example([s00], Counter(), 3, "anthropic") is s00


def test_the_store_only_grows(monkeypatch, tmp_path):
    store = SampleStore(tmp_path / "s.jsonl")
    monkeypatch.setattr(samples_mod, "execute", scripted([MEASURED] * 3))
    collect(THREE[:1], cfg(), store, log=lambda m: None)
    first = (tmp_path / "s.jsonl").read_text()
    monkeypatch.setattr(samples_mod, "execute", scripted([MEASURED] * 3))
    collect(THREE[:1], cfg(n=6), store, log=lambda m: None)
    assert (tmp_path / "s.jsonl").read_text().startswith(first)


# --- the flaky model, end to end --------------------------------------------------------------


def score_writer(example, record):
    return writer_checks(scenario_by_id()[example.id], record.get("answer", ""))


def test_a_model_that_passes_two_out_of_three_is_flaky_with_an_honest_interval(tmp_path):
    """The proposal's offline check: a real Writer, a scripted model, the real
    execute/collect/summarize path."""
    good = "RAGAS [2309.15217] scores faithfulness; the original RAG paper [2005.11401] pairs retrieval with generation."
    bad = good + " ARES [2311.09476] too."
    drafts = iter([[good], [good], [bad, bad]])      # the bad one survives the Writer's own retry
    exec_cfg = ExecConfig(model_factory=lambda: Streams(messages=iter([AIMessage(content=d) for d in next(drafts)])))
    store = SampleStore(tmp_path / "s.jsonl")
    collect([BY_ID["WRT01"]], cfg(exec=exec_cfg), store, log=lambda m: None)
    (summary,) = summarize([BY_ID["WRT01"]], store.samples(), FP, score_writer)
    assert (summary.k, summary.n, summary.cls) == (2, 3, "flaky")
    assert summary.interval == pytest.approx((0.208, 0.939), abs=1e-3)
    assert summary.checks["cites_only_notes"] == (2, 3)


# --- aggregation --------------------------------------------------------------------------------


def test_wilson_is_honest_about_three_samples():
    assert wilson(3, 3) == pytest.approx((0.438, 1.0), abs=1e-3)
    assert wilson(0, 3) == pytest.approx((0.0, 0.562), abs=1e-3)
    assert wilson(0, 0) == (0.0, 1.0)


def sample(example, outcome, *, config_fp=FP, example_hash=None, ok=True):
    return {"example_id": example.id, "example_hash": example_hash or example.example_hash,
            "config_fp": config_fp, "outcome": outcome, "outcome_reason": "",
            "record": {"ok": ok, "tokens": 10, "seconds": 1.0}}


def score_ok(example, record):
    return {"ok": record["ok"]}


def test_infra_stale_and_legacy_samples_never_enter_the_score():
    e = BY_ID["CRT01"]
    rows = [sample(e, MEASURED), sample(e, MEASURED), sample(e, INFRA, ok=False),
            sample(e, MEASURED, example_hash="old", ok=False),
            sample(e, MEASURED, config_fp="legacy:results-groq-v8.jsonl", ok=False)]
    (s,) = summarize([e], rows, FP, score_ok)
    assert (s.n, s.k, s.infra, s.stale, s.cls) == (2, 2, 1, 1, "insufficient")
    assert infra_rate(rows, FP) == (1, 4)


def test_a_crash_counts_as_a_failure():
    e = BY_ID["CRT01"]
    (s,) = summarize([e], [sample(e, MEASURED), sample(e, MEASURED), sample(e, CRASH)], FP, score_ok)
    assert (s.n, s.k, s.crash, s.cls) == (3, 2, 1, "flaky")


def test_differs_only_when_intervals_separate():
    """Found building E2: at n=3, even 3/3 against 0/3 overlaps ([0.44, 1] vs
    [0, 0.56]). Three samples classify one example; they cannot tell two
    configurations apart on it. At n=5 the extremes separate."""
    e = BY_ID["CRT01"]

    def of(k, n):
        rows = [sample(e, MEASURED)] * k + [sample(e, MEASURED, ok=False)] * (n - k)
        return summarize([e], rows, FP, score_ok)[0]

    assert not differs(of(3, 3), of(0, 3))
    assert differs(of(5, 5), of(0, 5))
    assert not differs(of(5, 5), of(3, 5))


# --- fingerprints -----------------------------------------------------------------------------


def test_code_hash_follows_behavioural_files_only(tmp_path):
    root = tmp_path / "pkg"
    (root / "agents").mkdir(parents=True)
    (root / "live_check").mkdir()
    (root / "agents" / "writer.py").write_text("PROMPT = 'a'\n")
    (root / "cli.py").write_text("x = 1\n")
    (root / "live_check" / "report.py").write_text("x = 1\n")
    before = code_hash(root)
    (root / "cli.py").write_text("x = 2\n")
    (root / "live_check" / "report.py").write_text("x = 2\n")
    assert code_hash(root) == before
    (root / "agents" / "writer.py").write_text("PROMPT = 'b'\n")
    assert code_hash(root) != before
    (root / "agents" / "new_agent.py").write_text("")       # a new file counts by default
    assert "agents/new_agent.py" not in NOT_BEHAVIOURAL


def test_the_config_fingerprint_changes_with_what_answered():
    assert config_fingerprint({**CONFIG, "sdk_max_retries": 6}) != config_fingerprint({**CONFIG, "sdk_max_retries": 2})
    assert config_fingerprint(dict(reversed(list(CONFIG.items())))) == FP


def test_run_config_reads_the_live_settings(monkeypatch):
    monkeypatch.setenv("RESEARCH_COPILOT_SDK_MAX_RETRIES", "4")
    monkeypatch.setenv("RESEARCH_COPILOT_FALLBACK_MODEL", "groq:openai/gpt-oss-20b")
    config = samples_mod.run_config("groq", "openai/gpt-oss-120b")
    assert config["sdk_max_retries"] == 4 and config["fallback_model"] == "groq:openai/gpt-oss-20b"
    assert config["max_request_tokens"] == 6_500 and len(config["code_hash"]) == 16


# --- Part A's results as history ---------------------------------------------------------------


def test_legacy_results_import_once_flagged_and_through_the_same_rules(tmp_path):
    legacy = tmp_path / "results-groq-v8.jsonl"
    rows = [
        {"id": "SUP01", "kind": "supervisor", "status": "review", "at": "2026-09-01T00:00:00",
         "observed": {"proposed": None, "routed_to": "writer", "override": "fallback to fixed policy",
                      "rationale": "(supervisor output unusable: RateLimitError: Error code: 429 - per minute)"}},
        {"id": "CRT01", "kind": "critic", "status": "pass", "at": "2026-09-01T00:01:00", "tokens": 900,
         "observed": {"verdict": "approve", "critique": "", "citation_checks": [{"arxiv_id": "2309.15217", "status": "found", "raw": "x"}]}},
        {"id": "WRT01", "kind": "writer", "status": "error", "at": "2026-09-01T00:02:00",
         "error": "RateLimitError: Error code: 429 - tokens per day (TPD): Limit 200000"},
        {"id": "S00", "status": "pending", "at": "2026-09-01T00:03:00"},
    ]
    legacy.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    store = SampleStore(tmp_path / "s.jsonl")
    assert import_legacy(legacy, store, EXAMPLES) == 3
    assert import_legacy(legacy, store, EXAMPLES) == 0        # idempotent
    got = {s["example_id"]: s for s in store.samples()}
    assert all(s["legacy"] and s["config_fp"] == "legacy:results-groq-v8.jsonl" for s in got.values())
    assert got["SUP01"]["outcome"] == INFRA                   # the hidden 429, in history too
    assert got["CRT01"]["outcome"] == MEASURED
    assert got["CRT01"]["record"]["citation_checks"] == [{"arxiv_id": "2309.15217", "status": "found"}]
    assert got["WRT01"]["outcome"] == INFRA
