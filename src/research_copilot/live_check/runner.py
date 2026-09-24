"""Run the live-verification scenarios against a real provider, safely. (Phase 7, A0)

CONCEPT: what "safely" means on a free tier
There is no money to cap on Groq's free tier, but there are walls: requests
per minute, tokens per minute, and tokens per *day*. Hitting the daily wall
halfway through a scenario throws away the work done so far and leaves the
scenario ambiguous. So the runner:

  - paces itself: before each scenario, it waits until the tokens used in the
    last 60 seconds plus the scenario's estimate fit under the per-minute
    limit. The Groq SDK also retries a 429 on its own (models.py sets
    max_retries), so pacing is the first defence and retry the second.
  - stops *between* scenarios, never inside one: with a daily token budget
    set, the next scenario only starts if its estimate fits in what is left.
  - is resumable: every result is appended to a JSONL file as soon as it
    exists, and a re-run skips scenarios already recorded. A run cut short
    by the daily limit continues tomorrow where it stopped.
  - reads the real limits: the first call asks Groq for its rate-limit
    headers, so the report states the limits of *this* account rather than
    a number copied from a blog post.

CONCEPT: two kinds of check
Every scenario result has `checks`: named booleans computed by code (did the
verdict match, did the draft cite only allowed ids). Some things code cannot
judge - whether a rationale actually supports its route (README item 3) - so
those are recorded in full and the result is marked `review`: a person reads
them. Part E replaces some of that reading with an LLM judge, which will then
need checking against these human reads.
"""

import json
import re
import time
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from research_copilot.config import PROJECT_ROOT, get_settings
from research_copilot.live_check.scenarios import ANTHROPIC_ONLY_ITEMS, SCENARIOS, Scenario

DEFAULT_RESULTS_DIR = PROJECT_ROOT / "data" / "live_check"

_ARXIV_ID = re.compile(r"\b(\d{4}\.\d{4,5})(?:v\d+)?\b")
_BRACKET_REF = re.compile(r"\[(\d+)\]")
# Phrases a draft uses to admit the evidence is missing. Fixed after the first
# Groq run: "I wasn’t able to locate any usable evidence" was a correct
# admission that this missed (curly apostrophe; "locate"). Apostrophes are
# normalised before matching, and the pattern keys on the *evidence* noun
# rather than on the exact verb.
_GAP_WORDS = re.compile(
    r"\b(no (usable |supporting |relevant )?(evidence|sources|information|research)"
    r"|(not|n't|unable to|wasn't able to|was not able to|could not|couldn't)\s+"
    r"(find|locate|identify)"
    r"|nothing (was )?found|no .{0,20}evidence)\b",
    re.IGNORECASE,
)


def admits_gap(text: str) -> bool:
    return bool(_GAP_WORDS.search((text or "").replace("\u2019", "'")))


def arxiv_ids(text: str) -> set[str]:
    return set(_ARXIV_ID.findall(text or ""))


@dataclass
class RunConfig:
    provider: str
    model: str
    results_path: Path
    tpm_limit: int = 8_000            # Groq free tier, gpt-oss-120b (read from response headers)
    daily_token_budget: int | None = 190_000  # headroom under gpt-oss-120b's 200K tokens/day
    model_factory: Callable = None    # () -> chat model; default get_chat_model
    search_tool: object = None        # default: the real search_arxiv
    verify_tool: object = None        # default: the real verify_citation
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.monotonic
    window: deque = field(default_factory=deque)


# ----------------------------------------------------------------- results file


def load_results(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    results = {}
    for line in path.read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            results[record["id"]] = record
    return results


def _append(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def tokens_today(results: dict[str, dict]) -> int:
    today = datetime.now(timezone.utc).date().isoformat()
    return sum(r.get("tokens", 0) for r in results.values() if str(r.get("at", "")).startswith(today))


# ----------------------------------------------------------------- pacing


def _pace(cfg: RunConfig, estimate: int) -> float:
    """Sleep until the last minute's tokens plus `estimate` fit the TPM limit."""
    waited = 0.0
    while True:
        now = cfg.clock()
        while cfg.window and now - cfg.window[0][0] > 60:
            cfg.window.popleft()
        used = sum(t for _, t in cfg.window)
        if used + min(estimate, cfg.tpm_limit) <= cfg.tpm_limit or not cfg.window:
            return waited
        pause = 60 - (now - cfg.window[0][0]) + 1
        cfg.sleep(pause)
        waited += pause


# ----------------------------------------------------------------- scenario runners
# Each returns (observed: dict, checks: dict[str, bool], needs_review: bool).


def _model(cfg):
    from research_copilot.models import get_chat_model

    return (cfg.model_factory or get_chat_model)()


def _run_probe_answer(s: Scenario, cfg):
    from research_copilot.chains import build_answer_chain

    answer = build_answer_chain(model=_model(cfg)).invoke({"question": s.inputs["question"]})
    return {"answer": answer}, {"non_empty": bool(answer.strip())}, False


def _run_probe_structured(s: Scenario, cfg):
    from research_copilot.chains import build_structured_chain

    try:
        result = build_structured_chain(model=_model(cfg)).invoke(s.inputs)
        return {"parsed": result.model_dump()}, {"parses": True}, False
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}, {"parses": False}, False


def _run_structured_smoke(s: Scenario, cfg):
    from research_copilot.agents.supervisor import SupervisorDecision
    from research_copilot.models import structured_output_kwargs

    model = _model(cfg)
    kwargs = structured_output_kwargs(model)
    method = kwargs["method"]
    runnable = model.with_structured_output(SupervisorDecision, include_raw=True, **kwargs)
    out = runnable.invoke(
        [
            SystemMessage(content="Decide who acts next on a research team."),
            HumanMessage(content="Nothing has been researched yet. Decide who acts next."),
        ]
    )
    raw = out.get("raw")
    # Which field did the model write first? Under json_schema the answer is
    # schema order; under function_calling it is the model's choice.
    order = []
    for call in getattr(raw, "tool_calls", None) or []:
        order = list(call.get("args", {}).keys())
    if not order and isinstance(getattr(raw, "content", None), str):
        try:
            order = list(json.loads(raw.content).keys())  # json.loads keeps key order
        except ValueError:
            pass
    parsed = out.get("parsed")
    return (
        {"method": method, "strict": kwargs.get("strict"), "parsed": parsed.model_dump() if parsed else None,
         "parsing_error": str(out.get("parsing_error") or ""), "field_order": order},
        {"parses": parsed is not None, "route_valid": bool(parsed) and parsed.next in ("researcher", "writer", "critic", "finish")},
        False,
    )


def _run_supervisor(s: Scenario, cfg):
    from research_copilot.agents.supervisor import make_supervisor

    state = s.inputs["state"]
    node = make_supervisor(model=_model(cfg), enable_critic=True, max_revisions=2)
    update = node(state)
    entry = update["supervisor_log"][-1]
    proposed = entry["proposed"]
    checks = {
        "usable_output": proposed is not None,
        "strict": proposed == s.expect["preferred"],
        "lenient": proposed in s.expect["acceptable"],
    }
    if proposed == "researcher":
        checks["brief_given"] = bool(update.get("researcher_brief") or entry.get("brief"))
    return (
        {"proposed": proposed, "routed_to": entry["routed_to"], "override": entry["override"],
         "rationale": entry["rationale"], "brief": entry.get("brief", "")},
        checks,
        True,  # item 3: the rationale is read against the route by a person
    )


def _run_critic(s: Scenario, cfg):
    from research_copilot.agents.critic import build_critic
    from research_copilot.live_check import scenarios as sc
    from research_copilot.tools.citations import verify_citation

    if s.inputs.get("lookup") == "error":
        @tool("verify_citation")
        def verify_citation_down(arxiv_id: str) -> str:
            """Check that an arXiv paper with this id exists, and get its title."""
            return "ERROR: lookup failed (network error: timed out). This says nothing about the citation."

        tools = [verify_citation_down]
    else:
        tools = [cfg.verify_tool or verify_citation]

    critic = build_critic(model=_model(cfg), tools=tools)
    out = critic.invoke({"question": sc.QUESTION, "draft": s.inputs["draft"], "research_notes": sc.NOTES})
    checks_by_id = {c["arxiv_id"]: c["status"] for c in out.get("citation_checks", [])}
    checks = {"verdict": out.get("verdict") == s.expect["verdict"],
              "used_verify_citation": bool(out.get("citation_checks"))}
    for arxiv_id in s.expect.get("found", []):
        checks[f"{arxiv_id}_found"] = any(arxiv_id in k and v == "found" for k, v in checks_by_id.items())
    for arxiv_id in s.expect.get("not_found", []):
        checks[f"{arxiv_id}_rejected_by_lookup"] = any(
            arxiv_id in k and v in ("not_found", "invalid") for k, v in checks_by_id.items()
        )
    return ({"verdict": out.get("verdict"), "critique": out.get("critique"),
             "citation_checks": out.get("citation_checks")}, checks, False)


def _run_researcher(s: Scenario, cfg):
    from research_copilot.agents.researcher import build_researcher
    from research_copilot.tools import search_arxiv

    real = cfg.search_tool or search_arxiv
    seen: list[str] = []

    @tool("search_arxiv")
    def search_arxiv_recorded(query: str, max_results: int = 5) -> str:
        """Search arXiv for academic papers. Returns title, authors, date, URL, and abstract for each."""
        result = real.invoke({"query": query, "max_results": max_results})
        seen.append(result)
        return result

    researcher = build_researcher(model=_model(cfg), tools=[search_arxiv_recorded])
    out = researcher.invoke(
        {
            "question": s.inputs["question"],
            "mode": "live-search",
            "messages": [HumanMessage(content=s.inputs["question"])],
            "research_notes": s.inputs.get("research_notes", ""),
            "researcher_brief": s.inputs.get("brief", ""),
        }
    )
    notes = out.get("research_notes", "")
    previous = s.inputs.get("research_notes", "")
    new_part = notes[len(previous):] if previous and notes.startswith(previous) else notes
    in_results = arxiv_ids("\n".join(seen)) | arxiv_ids(previous)
    first = next((line.strip() for line in new_part.splitlines() if line.strip()), "")
    checks = {
        "outcome": out.get("research_outcome") == s.expect["outcome"],
        "no_invented_ids": arxiv_ids(new_part) <= in_results,
    }
    if s.expect["outcome"] == "findings":
        checks["has_findings_sources_gaps"] = all(
            word in new_part.lower() for word in ("findings", "sources", "gaps")
        )
        checks["notes_not_an_answer"] = first.startswith(("-", "*", "Findings", "**Findings"))
    return ({"outcome": out.get("research_outcome"), "notes": notes,
             "searches": len(seen), "iterations": out.get("research_iterations")}, checks, False)


def _run_writer(s: Scenario, cfg):
    from research_copilot.agents.writer import make_writer

    write = make_writer(model=_model(cfg), max_revisions=2)
    state = {"question": s.inputs["question"], "mode": s.inputs.get("mode", "live-search"),
             "messages": [HumanMessage(content=s.inputs["question"])],
             **{k: v for k, v in s.inputs.items() if k not in ("question", "mode")}}
    draft = write(state)["draft"]
    return {"draft": draft, "cited_ids": sorted(arxiv_ids(draft))}, writer_checks(s, draft), False


def writer_checks(s: Scenario, draft: str) -> dict:
    """Pure function of the draft, so a changed check can re-score a recorded
    draft without another model call (see `rescore`)."""
    cited = arxiv_ids(draft)
    allowed = set(s.expect.get("allowed_ids", []))
    checks = {"non_empty": bool(draft.strip()), "cites_only_notes": cited <= allowed}
    if allowed:
        # Tightened after the first run: "cites only the notes" is trivially
        # true for a draft that cites nothing.
        checks["cites_something"] = bool(cited)
    if s.expect.get("forbidden_ids"):
        checks["dropped_flagged_citation"] = not (cited & set(s.expect["forbidden_ids"]))
    if s.expect.get("admits_gap"):
        checks["admits_gap"] = admits_gap(draft)
    if "max_ref" in s.expect:
        refs = {int(n) for n in _BRACKET_REF.findall(draft)}
        checks["refs_in_range"] = bool(refs) and max(refs) <= s.expect["max_ref"]
    return checks


def rescore(path: Path, *, log: Callable[[str], None] = print) -> list[str]:
    """Recompute checks for recorded Writer results after a check changes.

    Writer checks depend only on the recorded draft, so no model is called.
    The corrected record is appended with `rescored: true` and the previous
    checks kept beside it, so the change is visible in the results file.
    """
    from research_copilot.live_check.scenarios import by_id

    changed = []
    for sid, record in load_results(path).items():
        if record.get("kind") != "writer" or record.get("status") == "error":
            continue
        new = writer_checks(by_id()[sid], record["observed"]["draft"])
        if new == record["checks"]:
            continue
        updated = {**record, "checks": new, "previous_checks": record["checks"], "rescored": True,
                   "status": "pass" if all(new.values()) else "fail",
                   # The tokens were spent by the original call; re-scoring
                   # spends none, but the total must still count them once.
                   "at": datetime.now(timezone.utc).isoformat()}
        _append(path, updated)
        changed.append(sid)
        log(f"[rescored] {sid}: {record['status']} -> {updated['status']} (no model call)")
    return changed


def _run_e2e(s: Scenario, cfg):
    from research_copilot.graph import final_answer
    from research_copilot.multi_agent_graph import build_multi_agent_graph, multi_agent_turn_input

    model = cfg.model_factory
    graph = build_multi_agent_graph(
        model=model() if model else None,
        tools=[cfg.search_tool] if cfg.search_tool else None,
        critic_tools=[cfg.verify_tool] if cfg.verify_tool else None,
        enable_critic=s.inputs["critic"], max_revisions=1, memory_strategy="none",
    )
    state = graph.invoke(multi_agent_turn_input(s.inputs["question"]), {"recursion_limit": 80})
    log = state.get("supervisor_log", [])
    return (
        {"answer": final_answer(state),
         "routes": [(e["proposed"], e["routed_to"], e["override"]) for e in log],
         "rationales": [e["rationale"] for e in log],
         "dispatches": state.get("dispatches"), "revisions": state.get("revisions"),
         "verdict": state.get("verdict"), "budgets": state.get("budgets")},
        {"finished": bool(state.get("messages")) and state["messages"][-1].type == "ai",
         "no_fallbacks": not any(e["override"] == "fallback to fixed policy" for e in log)},
        True,
    )


RUNNERS = {
    "probe_answer": _run_probe_answer,
    "probe_structured": _run_probe_structured,
    "structured_smoke": _run_structured_smoke,
    "supervisor": _run_supervisor,
    "critic": _run_critic,
    "researcher": _run_researcher,
    "writer": _run_writer,
    "e2e": _run_e2e,
}


# ----------------------------------------------------------------- rate-limit probe


def read_groq_limits() -> dict:
    """One minimal request with raw headers, to read this account's real limits."""
    import groq

    settings = get_settings()
    client = groq.Groq()
    raw = client.chat.completions.with_raw_response.create(
        model=settings.groq_model, max_tokens=1, messages=[{"role": "user", "content": "hi"}]
    )
    return {k: v for k, v in raw.headers.items() if k.lower().startswith("x-ratelimit")}


# ----------------------------------------------------------------- the run


def run(
    cfg: RunConfig,
    scenarios: Iterable[Scenario] = SCENARIOS,
    *,
    log: Callable[[str], None] = print,
) -> dict[str, dict]:
    results = load_results(cfg.results_path)
    for s in scenarios:
        # Resume skips anything with a real outcome. An *error* is not an
        # outcome - a missing key, a network blip, a 429 after retries - so it
        # is run again. load_results keeps the latest line per id, so the
        # re-run's record replaces the error.
        if s.id in results and results[s.id]["status"] != "error":
            log(f"[skip] {s.id} already recorded ({results[s.id]['status']})")
            continue

        if s.item in ANTHROPIC_ONLY_ITEMS and cfg.provider != "anthropic":
            record = {
                "id": s.id, "item": s.item, "title": s.title, "provider": cfg.provider,
                "model": cfg.model, "status": "pending",
                "reason": "Anthropic-specific question; this provider cannot answer it. "
                          "Run again with --provider anthropic.",
                "tokens": 0, "at": datetime.now(timezone.utc).isoformat(),
            }
            _append(cfg.results_path, record)
            results[s.id] = record
            log(f"[pending] {s.id} {s.title}")
            continue

        if cfg.daily_token_budget is not None:
            left = cfg.daily_token_budget - tokens_today(results)
            if s.est_tokens > left:
                log(f"[stop] daily token budget: {left} left, {s.id} needs ~{s.est_tokens}. "
                    "Re-run later to resume from here.")
                break

        waited = _pace(cfg, s.est_tokens)
        if waited:
            log(f"[pace] waited {waited:.0f}s for the per-minute token window")

        started = time.monotonic()
        with get_usage_metadata_callback() as usage:
            try:
                observed, checks, review = RUNNERS[s.kind](s, cfg)
                error = None
            except Exception as exc:  # noqa: BLE001 - recorded, not raised
                observed, checks, review = {}, {}, False
                error = f"{type(exc).__name__}: {exc}"
        tokens = sum(u.get("total_tokens", 0) for u in usage.usage_metadata.values())
        cfg.window.append((cfg.clock(), tokens))

        if error:
            status = "error"
        elif not all(checks.values()):
            status = "fail"
        else:
            status = "review" if review else "pass"
        record = {
            "id": s.id, "item": s.item, "kind": s.kind, "title": s.title,
            "provider": cfg.provider, "model": cfg.model, "status": status,
            "checks": checks, "observed": observed, "expect": s.expect, "error": error,
            "note": s.note, "tokens": tokens, "seconds": round(time.monotonic() - started, 1),
            "at": datetime.now(timezone.utc).isoformat(),
        }
        _append(cfg.results_path, record)
        results[s.id] = record
        log(f"[{status}] {s.id} {s.title} ({tokens} tokens, {record['seconds']}s)"
            + (f" - {error}" if error else ""))

        if error and ("429" in error or "rate" in error.lower()):
            log("[stop] rate limited even after the SDK's retries; stopping so the "
                "remaining scenarios are not burned against a closed window.")
            break
    return results
