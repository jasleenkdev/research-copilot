"""Samples: stored append-only, keyed by fingerprint, collected over days. (Phase 7, E2)

CONCEPT: a sample is an observation, and the score is computed at read time
The store holds EvalRecords, never scores. Scoring (E3) and the judge (E4)
read stored records, so a changed evaluator re-scores everything without
another model call, and a scoring bug can never corrupt the observations.
Nothing is overwritten. The file only grows, and a line that was written stays
as it was written.

CONCEPT: what makes two samples comparable
A sample is keyed by (example_hash, config_fp):
  example_hash  what was asked, and what counts as right (dataset.py)
  config_fp     what answered: provider, model, fallback model, SDK retries,
                request-size ceiling, and a hash of the code that runs. The
                hash covers behaviour-bearing source files only. A README
                commit, a report change or a new evaluator does not reset
                samples, while any change to an agent, prompt, tool or
                transport does. The git SHA and a dirty flag are recorded
                for tracing, but they are not part of the key, since the
                content hash already says what ran.

CONCEPT: three outcomes, because variance is the whole system's
Part C's live run died once and passed once on retry timing alone. Scoring
that as model behaviour would make the network look like the model. So each
sample gets an outcome before any scoring:

  measured  the run finished and nothing outside the system under test
            changed what it did. Scored.
  infra     it did not finish, or it finished degraded by something
            external: a spent quota, a rate limit that outlasted the SDK's
            retries, a missing model, arXiv failing, or the fallback model
            answering instead of the one under test. NOT scored, NOT counted
            towards n, and reported as a separate reliability number.
  crash     an error that is not a known provider condition - most likely
            our bug. Scored as a failure, and the session stops loudly after
            recording it (the README's lesson on over-broad handlers).

One of the infra cases was found while building E2, not assumed. The
Supervisor's handler turns a per-minute 429 into `fallback to fixed policy`,
so the run ends normally with a changed route, and on SUP01 that route is
lenient-correct. Only the rationale it logs ("RateLimitError: Error code:
429") shows what happened, so that is what `outcome_of` reads.

CONCEPT: fill evenly, because a day can end at any moment
The daily limit can end a session after any sample. The scheduler always runs
the example with the fewest measured samples next (ties in dataset order,
which is priority order). So a short day still leaves every example at n or
n+1, never some at 3 and others at 0.
"""

import hashlib
import json
import subprocess
import uuid
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from research_copilot.config import PROJECT_ROOT
from research_copilot.evaluation.dataset import Example
from research_copilot.evaluation.execute import ExecConfig, execute
from research_copilot.resilience import MODEL_NOT_FOUND, QUOTA_EXHAUSTED, RATE_LIMITED, classify

DEFAULT_STORE = PROJECT_ROOT / "data" / "evaluation" / "samples.jsonl"
PACKAGE = Path(__file__).resolve().parents[1]

# Source files that cannot change what the system under test does. Everything
# else under src/research_copilot is in the code hash - the safe default for a
# file added later.
NOT_BEHAVIOURAL = frozenset({
    "__main__.py", "cli.py",
    "evaluation/__init__.py", "evaluation/dataset.py",   # the dataset has its own hash
    "evaluation/samples.py", "evaluation/aggregate.py",
    "live_check/report.py", "live_check/reviews.py",
})

INFRA_ERROR_KINDS = frozenset({QUOTA_EXHAUSTED, RATE_LIMITED, MODEL_NOT_FOUND})
MEASURED, INFRA, CRASH = "measured", "infra", "crash"


# --- fingerprints -----------------------------------------------------------------------------


def code_hash(root: Path = PACKAGE) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        if rel in NOT_BEHAVIOURAL or "__pycache__" in rel:
            continue
        digest.update(rel.encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()[:16]


def _git() -> tuple[str, bool]:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "src"], cwd=PROJECT_ROOT,
                                    capture_output=True, text=True, check=True).stdout.strip())
        return sha, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True


def run_config(provider: str, model: str) -> dict[str, Any]:
    """Everything that decides what answered, read from the environment the
    run actually uses."""
    import os

    from research_copilot.request_budget import max_request_tokens

    return {
        "provider": provider,
        "model": model,
        "fallback_model": os.getenv("RESEARCH_COPILOT_FALLBACK_MODEL") or "",
        "sdk_max_retries": int(os.getenv("RESEARCH_COPILOT_SDK_MAX_RETRIES") or 6),
        "max_request_tokens": max_request_tokens(provider),
        "code_hash": code_hash(),
    }


def config_fingerprint(config: dict[str, Any]) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


# --- the store ------------------------------------------------------------------------------


class SampleStore:
    """Append-only JSONL. One line per sample, never rewritten."""

    def __init__(self, path: Path = DEFAULT_STORE):
        self.path = Path(path)

    def append(self, sample: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(sample, default=str) + "\n")

    def samples(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    def counted(self, config_fp: str) -> Counter:
        """Samples towards n, per example_hash: measured and crashed, not infra."""
        return Counter(s["example_hash"] for s in self.samples()
                       if s["config_fp"] == config_fp and s["outcome"] in (MEASURED, CRASH))

    def tokens_today(self) -> int:
        """Tokens every sample in this store spent today (UTC). A local ceiling
        only: the provider's own counter is the authority (see
        live_check.runner.tokens_today for why)."""
        today = datetime.now(timezone.utc).date().isoformat()
        return sum(s.get("record", {}).get("tokens", 0) for s in self.samples()
                   if str(s.get("at", "")).startswith(today))


# --- outcomes -------------------------------------------------------------------------------


def outcome_of(example: Example, record: dict) -> tuple[str, str]:
    """(outcome, reason). Decided before any scoring, from the record alone."""
    ending = record.get("ending") or {}
    if ending.get("type") == "stopped":
        return INFRA, ending.get("reason", "stopped")
    if ending.get("type") == "error":
        kind = ending.get("kind", "")
        return (INFRA, kind) if kind in INFRA_ERROR_KINDS else (CRASH, kind or "error")
    for i in record.get("interventions") or []:
        if i.get("kind") == "fallback_model":
            return INFRA, f"{i.get('node', '?')}: the fallback model answered, not the model under test"
    for entry in record.get("supervisor_log") or []:
        if entry.get("override") == "fallback to fixed policy":
            kind = classify(RuntimeError(entry.get("rationale", "")))
            if kind in INFRA_ERROR_KINDS:
                return INFRA, f"supervisor step {entry.get('step', '?')}: {kind} became a fixed_policy route"
    for t in record.get("tools") or []:
        if t.get("status") in ("failed", "error") and t.get("name") != example.injects_failure:
            return INFRA, f"{t.get('name')} failed ({t.get('agent', '?')})"
    return MEASURED, ""


# --- collecting ---------------------------------------------------------------------------


@dataclass
class SampleConfig:
    provider: str
    model: str
    n: int = 3
    tpm_limit: int = 8_000
    daily_token_budget: int | None = 190_000
    exec: ExecConfig = field(default_factory=ExecConfig)
    # Pacing, shared with the live-check runner (same fields, same meaning).
    sleep: Callable[[float], None] = None
    clock: Callable[[], float] = None
    window: Any = None
    # Tests pin the config instead of reading the environment and the code.
    config: dict | None = None

    def __post_init__(self):
        import time
        from collections import deque

        self.sleep = self.sleep or time.sleep
        self.clock = self.clock or time.monotonic
        self.window = self.window if self.window is not None else deque()


class Crashed(RuntimeError):
    """A crash sample was recorded; the original exception is the cause."""


def next_example(examples: list[Example], counted: Counter, n: int, provider: str) -> Example | None:
    """The example with the fewest counted samples below n; ties in dataset order."""
    eligible = [e for e in examples
                if (not e.provider_only or e.provider_only == provider) and counted[e.example_hash] < n]
    return min(eligible, key=lambda e: counted[e.example_hash], default=None)


def collect(examples: Iterable[Example], cfg: SampleConfig, store: SampleStore, *,
            log: Callable[[str], None] = print) -> list[dict]:
    """Collect samples until every example has n, the budget ends, or an infra
    outcome says the provider will not serve more right now. Resumable: run
    it again tomorrow and it continues where it stopped."""
    from research_copilot.live_check.runner import _pace

    examples = list(examples)
    config = cfg.config or run_config(cfg.provider, cfg.model)
    fp = config_fingerprint(config)
    sha, dirty = _git()
    skipped = [e.id for e in examples if e.provider_only and e.provider_only != cfg.provider]
    if skipped:
        log(f"[pending] {', '.join(skipped)}: only answerable on another provider")
    written: list[dict] = []
    while True:
        counted = store.counted(fp)
        e = next_example(examples, counted, cfg.n, cfg.provider)
        if e is None:
            log(f"[done] every example has {cfg.n} counted samples for config {fp}")
            return written
        if cfg.daily_token_budget is not None:
            left = cfg.daily_token_budget - store.tokens_today()
            if e.est_tokens > left:
                log(f"[stop] daily token budget: {left} left, {e.id} needs ~{e.est_tokens}. Resume later.")
                return written
        waited = _pace(cfg, e.est_tokens)
        if waited:
            log(f"[pace] waited {waited:.0f}s for the per-minute token window")

        record, error = execute(e, cfg.exec)
        cfg.window.append((cfg.clock(), record.get("tokens", 0)))
        outcome, reason = outcome_of(e, record)
        sample = {
            "sample_id": uuid.uuid4().hex, "example_id": e.id, "example_hash": e.example_hash,
            "config_fp": fp, "config": config, "outcome": outcome, "outcome_reason": reason,
            "record": record, "at": datetime.now(timezone.utc).isoformat(),
            "git_sha": sha, "dirty": dirty, "legacy": False,
        }
        store.append(sample)
        written.append(sample)
        log(f"[{outcome}] {e.id} sample {counted[e.example_hash] + (outcome != INFRA)}/{cfg.n}"
            f" ({record.get('tokens', 0)} tokens, {record.get('seconds', 0)}s)" + (f" - {reason}" if reason else ""))
        if outcome == CRASH:
            raise Crashed(f"{e.id}: {record['ending'].get('detail', reason)}") from error
        if outcome == INFRA:
            log("[stop] the provider or a tool is not serving normally; stopping so the "
                "remaining samples are not spent against it. Resume later.")
            return written


# --- Part A's results, kept as history ------------------------------------------------------------


def import_legacy(results_path: Path, store: SampleStore, examples: Iterable[Example]) -> int:
    """Part A's results (live_check JSONL, v2-v8) as legacy samples.

    They predate both fingerprints, so they are keyed apart from every real
    sample ("legacy:<file>") and flagged. Aggregation leaves them out unless
    asked. What was observed is mapped onto the EvalRecord fields it
    corresponds to, and through the same builder, so the allow-list applies to
    history as well.
    """
    from research_copilot.evaluation.record import build_eval_record
    from research_copilot.live_check.runner import classify_error, load_results

    by_id = {e.id: e for e in examples}
    fp = f"legacy:{Path(results_path).name}"
    already = {s["sample_id"] for s in store.samples() if s["config_fp"] == fp}
    added = 0
    for sid, r in load_results(Path(results_path)).items():
        if sid not in by_id or r.get("status") == "pending":
            continue
        sample_id = f"{fp}:{sid}:{r.get('at', '')}"
        if sample_id in already:
            continue
        obs = r.get("observed") or {}
        state = {
            "supervisor_log": [{k: obs[k] for k in ("proposed", "routed_to", "override", "rationale", "brief")
                                if k in obs}] if "proposed" in obs else None,
            "verdict": obs.get("verdict"), "critique": obs.get("critique"),
            "citation_checks": obs.get("citation_checks"),
            "research_notes": obs.get("notes") or obs.get("research_notes"),
            "research_outcome": obs.get("outcome"), "research_iterations": obs.get("iterations"),
            "draft": obs.get("draft"), "dispatches": obs.get("dispatches"),
            "revisions": obs.get("revisions"), "unsupported_citations": obs.get("unsupported_citations"),
            "probe": {k: obs[k] for k in ("answer", "parsed", "method", "strict", "parsing_error", "field_order")
                      if k in obs} if r.get("kind", "").startswith(("probe", "structured")) else None,
        }
        if r.get("kind") == "e2e" and obs.get("answer"):
            state["draft"] = obs["answer"]
        error = r.get("error")
        if error:
            kind = classify_error(error)
            ending = {"type": "error", "kind": kind, "detail": str(error)[:300]}
            outcome = INFRA if kind in INFRA_ERROR_KINDS else CRASH
        else:
            ending, outcome = {"type": "done"}, None
        record = build_eval_record({k: v for k, v in state.items() if v is not None},
                                   ending=ending, tokens=r.get("tokens", 0),
                                   sdk_retries=r.get("sdk_retries", 0), seconds=r.get("seconds", 0))
        if outcome is None:
            # The same rules as a new sample - a 429 hidden in a fixed_policy
            # route is infra in history too. (Legacy records have no tool
            # statuses, so an arXiv outage cannot be seen in them.)
            outcome = outcome_of(by_id[sid], record)[0]
        store.append({"sample_id": sample_id, "example_id": sid, "example_hash": "legacy",
                      "config_fp": fp, "config": {"provider": r.get("provider"), "model": r.get("model")},
                      "outcome": outcome, "outcome_reason": "", "record": record, "at": r.get("at"),
                      "git_sha": "unknown", "dirty": True, "legacy": True})
        added += 1
    return added
