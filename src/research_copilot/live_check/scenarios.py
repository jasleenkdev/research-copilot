"""The live-verification scenarios: README.md's real-key list, as labelled cases.

CONCEPT: a scenario is a test case whose answer key was written *before* the
model ran
Every scenario states what should happen - "this critique is an evidence
problem, so the Supervisor should route to the Researcher" - before any model
is asked. That is what makes a live run *verification* rather than a demo: the
result is compared against a judgement made in advance, not rationalised
afterwards.

Two kinds of expectation:
  preferred     the answer a careful human would give
  acceptable    answers that are defensible but not ideal
Scoring counts both "strict" (== preferred) and "lenient" (in acceptable), and
the gap between the two numbers is itself a finding: a Supervisor that is
often lenient-right and strict-wrong is making the *second*-best call.

Listed in README priority order (item 2, then 4 and 5, then 6, then 7), so
that a run stopped by a daily token limit has covered the most important items
first.

Designed from the start to become Part E's first LangSmith dataset. Each
scenario is (inputs, expected outputs, metadata), which is exactly a dataset
example.

Items follow README.md's "What needs a real ANTHROPIC_API_KEY" numbering:
  0   provider portability probe (not on that list - Phase 1's promise that a
      provider change touches only models.py)
  1   json_schema + server-side-fallback beta. ANTHROPIC ONLY. A Groq run
      records it as pending; see `ANTHROPIC_ONLY_ITEMS`.
  1g  Groq's own structured-output path, labelled separately so it can never
      be read as item 1
  2   Supervisor classification        3   rationale vs. route (reviewed by hand)
  4   Researcher                       5   Writer
  6   Critic                           7   cost / latency (end-to-end runs)
  8   real arXiv behaviour (covered inside 6's live lookups)
"""

from dataclasses import dataclass, field
from typing import Any

# Items whose question is about Anthropic's API specifically. A run on any
# other provider must report these as PENDING, never as passed or failed.
ANTHROPIC_ONLY_ITEMS = frozenset({"1"})


@dataclass(frozen=True)
class Scenario:
    id: str
    item: str
    kind: str
    title: str
    inputs: dict[str, Any]
    expect: dict[str, Any] = field(default_factory=dict)
    # Rough tokens this scenario spends on Groq, used only for pacing under a
    # tokens-per-minute limit. A guess, and deliberately on the high side.
    est_tokens: int = 3000
    note: str = ""
    # Phase 7 A1 (after day one): facts that are TRUE in this scenario's state
    # and that any sound rationale must be consistent with. This is what lets
    # a judge catch "wrong premise, acceptable route" (SUP04, SUP08 on day one):
    # a rationale that agrees with its route but asserts something the state
    # contradicts. Checking rationale-vs-route alone cannot see that failure.
    premise_facts: tuple[str, ...] = ()
    # Set when the expected route itself is questionable, with the reason.
    # The score is kept for comparability across runs, but the report flags it.
    label_disputed: str = ""


# ----------------------------------------------------------------- shared texts

QUESTION = "How are RAG pipelines evaluated, and how reliable are those evaluations?"

NOTES = (
    "- Findings: RAGAS evaluates RAG pipelines with reference-free metrics - "
    "faithfulness, answer relevance, context precision (RAGAS, "
    "http://arxiv.org/abs/2309.15217).\n"
    "- Findings: The original RAG model pairs a seq2seq generator with a dense "
    "retriever over Wikipedia (Retrieval-Augmented Generation for "
    "Knowledge-Intensive NLP Tasks, http://arxiv.org/abs/2005.11401).\n"
    "- Sources: RAGAS: Automated Evaluation of Retrieval Augmented Generation, "
    "http://arxiv.org/abs/2309.15217; Retrieval-Augmented Generation for "
    "Knowledge-Intensive NLP Tasks, http://arxiv.org/abs/2005.11401\n"
    "- Gaps: no evidence found on how well RAGAS agrees with human judgement."
)

GOOD_DRAFT = (
    "RAG pipelines are commonly evaluated with RAGAS "
    "[http://arxiv.org/abs/2309.15217], which scores faithfulness, answer "
    "relevance and context precision without needing reference answers. How "
    "reliable those scores are is less clear: no evidence was found on how "
    "well RAGAS agrees with human judgement."
)

# Controls for item 2 (added after the first Groq run, see SUPC*): a question
# with no reliability half, and notes with no open gap, so a route to the
# Researcher cannot be explained by "part of the question is genuinely
# unanswered".
CONTROL_QUESTION = "How are RAG pipelines evaluated?"
CLEAN_NOTES = NOTES.replace(
    "- Gaps: no evidence found on how well RAGAS agrees with human judgement.", "- Gaps: none."
)
CLEAN_DRAFT = (
    "RAG pipelines are commonly evaluated with RAGAS "
    "[http://arxiv.org/abs/2309.15217], which scores faithfulness, answer "
    "relevance and context precision without needing reference answers."
)

KB_NOTES = (
    "[1] (notes/eval.md)\nRetrieval quality is measured with recall@k against a "
    "labelled set of relevant chunks.\n\n"
    "[2] (notes/eval.md)\nAnswer faithfulness is checked by asking whether each "
    "claim in the answer is supported by a retrieved chunk."
)


def _log(*routes: tuple[str, int] | str, briefs: dict[int, str] | None = None) -> list[dict]:
    """A supervisor_log with the given dispatch order. Each route is a name, or
    (name, revision)."""
    briefs = briefs or {}
    entries = []
    for step, route in enumerate(routes, start=1):
        name, revision = (route, 0) if isinstance(route, str) else route
        entries.append(
            {
                "step": step, "proposed": name, "rationale": "(earlier decision)",
                "routed_to": name, "override": "", "brief": briefs.get(step, ""),
                "revision": revision,
            }
        )
    return entries


def _after_rejection(critique: str, *, draft: str, checks: list[dict] | None = None,
                     notes: str = NOTES, question: str = QUESTION) -> dict:
    """State right after a Critic rejection has gone through start_revision:
    the Supervisor must now decide who fixes it."""
    return {
        "question": question,
        "mode": "live-search",
        "research_notes": notes,
        "research_outcome": "findings",
        "draft": draft,
        "critique": critique,
        "verdict": "reject",
        "citation_checks": checks or [],
        "revisions": 1,
        "dispatches": {"researcher": 1, "writer": 1, "critic": 1},
        "supervisor_log": _log("researcher", "writer", "critic"),
        "budgets": {"researcher": {"used": 3}, "writer": {"used": 0}, "critic": {"used": 0}},
    }


def _route(preferred: str, *acceptable: str) -> dict:
    return {"preferred": preferred, "acceptable": sorted({preferred, *acceptable})}


# ----------------------------------------------------------------- scenarios

SCENARIOS: list[Scenario] = [
    # --- 0: portability ------------------------------------------------------
    Scenario("P01", "0", "probe_answer", "Phase 1 answer chain runs on this provider",
             {"question": "In one sentence, what is retrieval-augmented generation?"},
             est_tokens=500),
    Scenario("P02", "0", "probe_structured", "Phase 1 PydanticOutputParser chain parses on this provider",
             {"question": "What is retrieval-augmented generation?", "notes": ""},
             est_tokens=1500),

    # --- 1 / 1g: structured output --------------------------------------------
    Scenario("S00", "1", "structured_smoke",
             "json_schema + server-side-fallback beta (Anthropic only)", {}, est_tokens=800,
             note="On any other provider this is recorded as PENDING, not run."),
    Scenario("S00g", "1g", "structured_smoke",
             "This provider's structured-output path parses a SupervisorDecision", {},
             est_tokens=800,
             note="NOT a substitute for item 1: a different method on a different provider."),

    # --- 2 / 3: Supervisor classification (and rationale, read by hand) -------
    Scenario("SUP01", "2", "supervisor", "Unsupported quantitative claim -> evidence problem",
             {"state": _after_rejection(
                 "Evidence problem: the draft says RAGAS scores correlate 0.9 with human "
                 "ratings, but the research notes contain no source for that - they list "
                 "agreement with human judgement as a gap. A source is needed.",
                 draft=GOOD_DRAFT.replace("How reliable those scores are is less clear: no evidence was found on how well RAGAS agrees with human judgement.",
                                          "RAGAS scores correlate 0.9 with human ratings [2309.15217]."),
                 checks=[{"arxiv_id": "2309.15217", "status": "found"}])},
             _route("researcher", "writer"),
             note="writer is acceptable: removing the claim also fixes it."),
    Scenario("SUP02", "2", "supervisor", "Writer added a real paper that is not in the notes",
             {"state": _after_rejection(
                 "The draft cites ARES (2311.09476). The paper exists on arXiv, but it is "
                 "not in the research notes - the Writer added a source that was never "
                 "researched.",
                 draft=GOOD_DRAFT + " ARES [2311.09476] is another option.",
                 checks=[{"arxiv_id": "2309.15217", "status": "found"},
                         {"arxiv_id": "2311.09476", "status": "found"}])},
             _route("writer", "researcher"),
             note="The citation is real but unresearched: it is the Writer's error. "
                  "researcher is defensible (research ARES properly).",
             premise_facts=("ARES (2311.09476) does not appear anywhere in the research notes", "ARES exists on arXiv (the citation check says found)")),
    Scenario("SUP03", "2", "supervisor", "Structure: omits covered material, conclusion first",
             {"state": _after_rejection(
                 "Writing problem: the draft never mentions the original RAG paper even "
                 "though the notes cover it, and it states its conclusion before any "
                 "evidence.", draft=GOOD_DRAFT)},
             _route("writer")),
    Scenario("SUP04", "2", "supervisor", "The notes themselves cite a non-existent paper",
             {"state": _after_rejection(
                 "Evidence problem: the draft's claim about BENCH-RAG rests on 2401.99999, "
                 "which the notes list as a source, but no arXiv paper has that id. The "
                 "claim needs a real source.",
                 draft=GOOD_DRAFT + " BENCH-RAG [2401.99999] standardises these metrics.",
                 notes=NOTES + "\n- Findings: BENCH-RAG standardises RAG metrics (BENCH-RAG, http://arxiv.org/abs/2401.99999).",
                 checks=[{"arxiv_id": "2309.15217", "status": "found"},
                         {"arxiv_id": "2401.99999", "status": "not_found"}])},
             _route("researcher"),
             premise_facts=("BENCH-RAG's only source, 2401.99999, is not_found on arXiv, so BENCH-RAG itself may not exist",)),
    Scenario("SUP05", "2", "supervisor", "Part of the question has no evidence at all",
             {"state": _after_rejection(
                 "Evidence problem: the question asks what these evaluations cost to run. "
                 "Neither the notes nor the draft say anything about cost.",
                 draft=GOOD_DRAFT,
                 question="How are RAG pipelines evaluated, and what do those evaluations cost to run?")},
             _route("researcher")),
    Scenario("SUP06", "2", "supervisor", "Pure style: too long and repetitive",
             {"state": _after_rejection(
                 "Writing problem: the answer is three times longer than it needs to be and "
                 "repeats the same point about faithfulness twice.", draft=GOOD_DRAFT * 3)},
             _route("writer"),
             premise_facts=("The critique is about length and repetition only - it names no missing evidence",)),
    Scenario("SUP07", "2", "supervisor", "Speculation presented as fact",
             {"state": _after_rejection(
                 "Writing problem: the draft states that RAGAS will replace human "
                 "evaluation. The notes make no such claim; this is speculation presented "
                 "as established.",
                 draft=GOOD_DRAFT + " RAGAS will soon replace human evaluation entirely.")},
             _route("writer")),
    Scenario("SUP08", "2", "supervisor", "Critic says 'no source', but the notes have one",
             {"state": _after_rejection(
                 "Evidence problem: nothing supports the claim that RAGAS is reference-free.",
                 draft=GOOD_DRAFT.replace(" [http://arxiv.org/abs/2309.15217]", ""))},
             _route("writer", "researcher"),
             note="The notes already support it - the draft just dropped the citation.",
             premise_facts=("The research notes DO contain a source for RAGAS being reference-free (2309.15217); the critique is wrong that nothing supports it",)),
    Scenario("SUP09", "2", "supervisor", "First decision of a turn",
             {"state": {"question": QUESTION, "mode": "live-search", "dispatches": {},
                        "supervisor_log": []}},
             _route("researcher")),
    Scenario("SUP10", "2", "supervisor", "Notes gathered, no draft yet",
             {"state": {"question": QUESTION, "mode": "live-search", "research_notes": NOTES,
                        "research_outcome": "findings", "dispatches": {"researcher": 1},
                        "supervisor_log": _log("researcher")}},
             _route("writer"),
             label_disputed="The notes list an open gap on half the question (reliability), so another research pass is defensible; SUPC10 is the unambiguous version"),
    Scenario("SUP11", "2", "supervisor", "Human rejection: too technical",
             {"state": {**_after_rejection("(approved by the critic)", draft=GOOD_DRAFT),
                        "verdict": "approve", "human_verdict": "reject",
                        "human_feedback": "Too technical for someone new to the field - "
                                          "explain what faithfulness means in plain words."}},
             _route("writer")),
    Scenario("SUP12", "2", "supervisor", "Human rejection: missing recent work",
             {"state": {**_after_rejection("(approved by the critic)", draft=GOOD_DRAFT),
                        "verdict": "approve", "human_verdict": "reject",
                        "human_feedback": "You ignored everything published in 2025 and "
                                          "2026 on RAG evaluation."}},
             _route("researcher")),
    Scenario("SUP13", "2", "supervisor", "Follow-up research found nothing - don't repeat it",
             {"state": {**_after_rejection(
                 "Evidence problem: nothing supports how well RAGAS agrees with human "
                 "judgement; a source is needed.", draft=GOOD_DRAFT),
                 "research_outcome": "nothing_found",
                 "dispatches": {"researcher": 2, "writer": 1, "critic": 1},
                 "supervisor_log": _log("researcher", "writer", "critic", ("researcher", 1),
                                        briefs={4: "studies of RAGAS agreement with human judgement"})}},
             _route("writer"),
             note="The brief was tried and found nothing; writing the gap down is the "
                  "right move. A researcher proposal is the 'same brief after "
                  "nothing_found' failure (the dispatch-cap guard would also refuse it).",
             premise_facts=("A research pass on this gap already ran with a brief and found nothing", "The researcher's dispatch cap for this turn is spent")),

    # --- 2 (controls): the same situations with no open gap -------------------
    # Added after the first Groq run, where every wrong route cited the notes'
    # open gap. Expectations written before these ran, like all the others -
    # but designed in response to results, and labelled so.
    Scenario("SUPC02", "2", "supervisor", "CONTROL of SUP02: Writer-added citation, no open gap",
             {"state": _after_rejection(
                 "The draft cites ARES (2311.09476). The paper exists on arXiv, but it is "
                 "not in the research notes - the Writer added a source that was never "
                 "researched.",
                 draft=CLEAN_DRAFT + " ARES [2311.09476] is another option.",
                 notes=CLEAN_NOTES, question=CONTROL_QUESTION,
                 checks=[{"arxiv_id": "2309.15217", "status": "found"},
                         {"arxiv_id": "2311.09476", "status": "found"}])},
             _route("writer", "researcher"), note="control: added after the first run",
             premise_facts=("ARES (2311.09476) does not appear anywhere in the research notes", "ARES exists on arXiv (the citation check says found)")),
    Scenario("SUPC06", "2", "supervisor", "CONTROL of SUP06: pure style, no open gap",
             {"state": _after_rejection(
                 "Writing problem: the answer is three times longer than it needs to be and "
                 "repeats the same point about faithfulness twice.",
                 draft=CLEAN_DRAFT * 3, notes=CLEAN_NOTES, question=CONTROL_QUESTION)},
             _route("writer"), note="control: added after the first run",
             premise_facts=("The critique is about length and repetition only - it names no missing evidence",)),
    Scenario("SUPC08", "2", "supervisor", "CONTROL of SUP08: critic wrong about a source, no open gap",
             {"state": _after_rejection(
                 "Evidence problem: nothing supports the claim that RAGAS is reference-free.",
                 draft=CLEAN_DRAFT.replace(" [http://arxiv.org/abs/2309.15217]", ""),
                 notes=CLEAN_NOTES, question=CONTROL_QUESTION)},
             _route("writer", "researcher"), note="control: added after the first run",
             premise_facts=("The research notes DO contain a source for RAGAS being reference-free (2309.15217); the critique is wrong that nothing supports it",)),
    Scenario("SUPC10", "2", "supervisor", "CONTROL of SUP10: notes complete, no draft yet",
             {"state": {"question": CONTROL_QUESTION, "mode": "live-search",
                        "research_notes": CLEAN_NOTES, "research_outcome": "findings",
                        "dispatches": {"researcher": 1}, "supervisor_log": _log("researcher")}},
             _route("writer"), note="control: added after the first run"),

    # --- 4: Researcher (real arXiv search) ------------------------------------
    Scenario("RES01", "4", "researcher", "Notes format, sources copied verbatim",
             {"question": "What does the RAGAS framework measure when evaluating RAG pipelines?"},
             {"outcome": "findings"}, est_tokens=9000),
    Scenario("RES02", "4", "researcher", "Follow-up on already-covered ground -> NOTHING NEW",
             {"question": QUESTION, "research_notes": NOTES,
              "brief": "the arXiv id of the RAGAS paper"},
             {"outcome": "nothing_found"}, est_tokens=7000,
             note="Soft: a model that searches and finds something genuinely new is "
                  "not wrong, just not what the brief asked for."),

    # --- 5: Writer --------------------------------------------------------------
    Scenario("WRT01", "5", "writer", "Cites only what the notes contain",
             {"question": QUESTION, "research_notes": NOTES},
             {"allowed_ids": ["2309.15217", "2005.11401"]}),
    Scenario("WRT02", "5", "writer", "Empty notes -> says so, cites nothing",
             {"question": QUESTION, "research_notes": ""}, {"allowed_ids": [], "admits_gap": True}),
    Scenario("WRT03", "5", "writer", "Knowledge-base excerpts -> only [n] citations in range",
             {"question": "How do my notes say RAG answers should be evaluated?",
              "research_notes": KB_NOTES, "mode": "knowledge-base"},
             {"allowed_ids": [], "max_ref": 2}),
    Scenario("WRT04", "5", "writer", "Revision removes the citation the critique flagged",
             {"question": QUESTION, "research_notes": NOTES,
              "draft": GOOD_DRAFT + " ARES [2311.09476] is another option.",
              "revisions": 1, "verdict": "reject",
              "critique": "Remove ARES (2311.09476): it is not in the research notes."},
             {"allowed_ids": ["2309.15217", "2005.11401"], "forbidden_ids": ["2311.09476"]}),

    # --- 6: Critic (real arXiv lookups, except CRT05) --------------------------
    Scenario("CRT01", "6", "critic", "Good draft, real citation",
             {"draft": GOOD_DRAFT}, {"verdict": "approve", "found": ["2309.15217"]}, est_tokens=5000),
    Scenario("CRT02", "6", "critic", "Fabricated arXiv id",
             {"draft": GOOD_DRAFT + " BENCH-RAG [2401.99999] standardises these metrics."},
             {"verdict": "reject", "not_found": ["2401.99999"]}, est_tokens=6000,
             note="If arXiv answers this id with an error entry, the status is 'invalid' "
                  "instead of 'not_found' - both are rejections; the report says which."),
    Scenario("CRT03", "6", "critic", "Real id, wrong paper (Attention cited as RAGAS)",
             {"draft": GOOD_DRAFT.replace("http://arxiv.org/abs/2309.15217", "1706.03762")},
             {"verdict": "reject", "found": ["1706.03762"]}, est_tokens=6000,
             note="Tests whether the Critic reads the title verify_citation returns."),
    Scenario("CRT04", "6", "critic", "Real citation, claim not in the notes",
             {"draft": GOOD_DRAFT.replace("How reliable those scores are is less clear: no evidence was found on how well RAGAS agrees with human judgement.",
                                          "RAGAS agrees with human raters 95% of the time [2309.15217].")},
             {"verdict": "reject"}, est_tokens=5000),
    Scenario("CRT05", "6", "critic", "Every lookup fails (network) - must not reject for that",
             {"draft": GOOD_DRAFT, "lookup": "error"}, {"verdict": "approve"}, est_tokens=5000,
             note="verify_citation is stubbed to return ERROR. ERROR is not evidence."),
    Scenario("CRT06", "6", "critic", "Malformed arXiv id",
             {"draft": GOOD_DRAFT.replace("http://arxiv.org/abs/2309.15217", "arXiv:12345")},
             {"verdict": "reject"}, est_tokens=5000),

    # --- 7: end to end (cost, latency, an overall read) -------------------------
    Scenario("E2E01", "7", "e2e", "Full run with the Critic",
             {"question": "How are RAG pipelines evaluated?", "critic": True},
             est_tokens=45000,
             note="Largest spend. Runs last so a daily token cap stops before it, not in it."),
    Scenario("E2E02", "7", "e2e", "Full run without the Critic",
             {"question": "What problem does retrieval-augmented generation solve?", "critic": False},
             est_tokens=25000),
]


def by_id() -> dict[str, Scenario]:
    return {s.id: s for s in SCENARIOS}


def to_dataset_example(s: Scenario) -> dict:
    """One scenario as a dataset example: (inputs, reference outputs, metadata).

    The shape Part E uploads to LangSmith. Written now so that day one's
    distinction survives into it: a Supervisor example carries BOTH the route
    reference (preferred / acceptable) AND the `premise_facts` a rationale
    must be consistent with. Those are two separate judgements, and a
    dataset with only the first cannot express "wrong premise, acceptable
    route".
    """
    outputs = dict(s.expect)
    if s.kind == "supervisor":
        outputs = {
            "preferred_route": s.expect["preferred"],
            "acceptable_routes": s.expect["acceptable"],
            "premise_facts": list(s.premise_facts),
        }
    return {
        "inputs": {"kind": s.kind, **s.inputs},
        "outputs": outputs,
        "metadata": {"id": s.id, "item": s.item, "title": s.title, "note": s.note,
                     "label_disputed": s.label_disputed},
    }
