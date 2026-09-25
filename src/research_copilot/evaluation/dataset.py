"""The evaluation dataset: Part A's scenarios, as versioned examples. (Phase 7, E1)

CONCEPT: an example is a scenario plus the facts needed to trust its score
A live-check `Scenario` says what to run and what should happen. An `Example`
keeps both and adds what Parts A and B showed a score cannot be read without:

  reference     what "correct" means, in one named shape per kind (below),
                instead of a free-form `expect` dict. A key the schema does not
                know is an error, not a silently ignored expectation.
  source        hand_crafted | captured (built from a real run's artifacts,
                like E2E01's draft and notes) | scaled_variant (a larger
                sibling built for E5)
  designed_after  "" if the example and its reference were written before any
                model ran. Otherwise, what was added after seeing results, and
                when. SUPC* and CRT07-CRT10 were designed in response to runs,
                and the Supervisor's premise facts were added after day one.
                None of that is wrong, but a score on a post-hoc example is
                weaker evidence than one on a pre-registered example, and the
                report has to be able to say which is which.
  control_of    the example this one is a control for (same situation, one
                thing removed - the open gap, the Writer's flag)
  scale_of      the smaller sibling this one repeats at a larger scale (the
                same defect among 13 citations instead of 2)
  scale         measured from the inputs: citations in the draft, sources and
                characters in the notes. CRT10 is why this exists: the Critic
                caught the defect at 2 citations and missed it at 13, and a
                small hand-crafted set cannot show that.

CONCEPT: two fingerprints, for two different comparisons
`example_hash` covers what is measured: kind, inputs, reference. Two samples
of the same example are comparable only if their hashes match. Adding a new
example leaves every other example's hash, and so its accumulated samples,
untouched. `dataset_version` hashes the whole set and names a report's
dataset. Titles, notes and provenance are left out of both: fixing a note's
wording must not reset the samples already collected.

The live-check scenarios stay the single source. Examples are derived from
them, and every scenario id must appear in PROVENANCE, so a new scenario
cannot enter the dataset without someone stating where it came from.
"""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from research_copilot.live_check.runner import arxiv_ids
from research_copilot.live_check.scenarios import ANTHROPIC_ONLY_ITEMS, SCENARIOS, Scenario

SOURCES = frozenset({"hand_crafted", "captured", "scaled_variant"})

# The reference keys each kind may carry. Scoring (E3) reads exactly these.
REFERENCE_KEYS: dict[str, frozenset[str]] = {
    "probe_answer": frozenset({"non_empty"}),
    "probe_structured": frozenset({"parses"}),
    "structured_smoke": frozenset({"parses", "route_valid"}),
    "supervisor": frozenset({"preferred_route", "acceptable_routes", "premise_facts"}),
    "researcher": frozenset({"outcome"}),
    "writer": frozenset({"allowed_ids", "forbidden_ids", "admits_gap", "max_ref"}),
    "critic": frozenset({"verdict", "must_find", "must_reject_by_lookup", "must_name"}),
    "e2e": frozenset({"finishes", "no_fixed_policy_fallback", "no_unsupported_citations"}),
}

# Scenario `expect` key -> reference key, per kind. Migration fails on any
# expect key not listed here, so no expectation can be dropped on the way.
_EXPECT_TO_REFERENCE: dict[str, dict[str, str]] = {
    "supervisor": {"preferred": "preferred_route", "acceptable": "acceptable_routes"},
    "researcher": {"outcome": "outcome"},
    "writer": {k: k for k in ("allowed_ids", "forbidden_ids", "admits_gap", "max_ref")},
    "critic": {"verdict": "verdict", "found": "must_find", "not_found": "must_reject_by_lookup",
               "must_name": "must_name"},
}

# The checks Part A's runner applied without writing them in `expect`. Made
# explicit here, so the reference says everything a sample is scored on.
_IMPLICIT_REFERENCE: dict[str, dict[str, Any]] = {
    "probe_answer": {"non_empty": True},
    "probe_structured": {"parses": True},
    "structured_smoke": {"parses": True, "route_valid": True},
    # `no_unsupported_citations` is new for end-to-end runs, and not invented:
    # it is the defect E2E01's confirming run delivered to the user.
    "e2e": {"finishes": True, "no_fixed_policy_fallback": True, "no_unsupported_citations": True},
}


@dataclass(frozen=True)
class Provenance:
    source: str = "hand_crafted"
    designed_after: str = ""
    control_of: str = ""
    scale_of: str = ""


_PREMISES = "premise_facts added after day one (416d465)"
_DAY_ONE = "designed after the first Groq run (cd7e5ad)"

# Every scenario, stated. Unlisted ids fail `build_examples`. The dates come
# from git history, not memory: the A0 scenarios predate every run.
PROVENANCE: dict[str, Provenance] = {
    **{sid: Provenance() for sid in (
        "P01", "P02", "S00", "S00g",
        "SUP01", "SUP03", "SUP05", "SUP07", "SUP09", "SUP11", "SUP12",
        "RES01", "RES02", "WRT01", "WRT02", "WRT03", "WRT04",
        "CRT01", "CRT02", "CRT03", "CRT04", "CRT05", "CRT06",
        "E2E01", "E2E02",
    )},
    # Written in A0; only the premise facts came later.
    **{sid: Provenance(designed_after=_PREMISES) for sid in ("SUP02", "SUP04", "SUP06", "SUP08", "SUP13")},
    # Not a changed reference: the label was flagged as disputed after day one.
    "SUP10": Provenance(designed_after="label marked disputed after day one (416d465)"),
    "SUPC02": Provenance(designed_after=f"{_DAY_ONE}; {_PREMISES}", control_of="SUP02"),
    "SUPC06": Provenance(designed_after=f"{_DAY_ONE}; {_PREMISES}", control_of="SUP06"),
    "SUPC08": Provenance(designed_after=f"{_DAY_ONE}; {_PREMISES}", control_of="SUP08"),
    "SUPC10": Provenance(designed_after=_DAY_ONE, control_of="SUP10"),
    "CRT07": Provenance(designed_after="designed after E2E01's confirming run (648b53b)"),
    "CRT08": Provenance(designed_after="designed after E2E01's confirming run (648b53b)", control_of="CRT07"),
    "CRT09": Provenance(source="captured", scale_of="CRT07",
                        designed_after="designed after CRT08 passed (ea67541)"),
    "CRT10": Provenance(source="captured", scale_of="CRT08", control_of="CRT09",
                        designed_after="designed after CRT08 passed (ea67541)"),
}


@dataclass(frozen=True)
class Example:
    id: str
    item: str
    kind: str
    title: str
    inputs: dict[str, Any]
    reference: dict[str, Any]
    source: str = "hand_crafted"
    designed_after: str = ""
    control_of: str = ""
    scale_of: str = ""
    scale: dict[str, int] = field(default_factory=dict)
    # A provider this example can only be answered on ("" = any). A run on
    # another provider records it as pending, never as a pass or a fail.
    provider_only: str = ""
    label_disputed: str = ""
    note: str = ""
    est_tokens: int = 3000

    @property
    def example_hash(self) -> str:
        return _hash({"id": self.id, "kind": self.kind, "inputs": self.inputs, "reference": self.reference})

    @property
    def pre_registered(self) -> bool:
        return not self.designed_after


def _hash(obj: Any) -> str:
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def measure_scale(inputs: dict[str, Any]) -> dict[str, int]:
    """What makes an example big, measured rather than labelled."""
    state = inputs.get("state") or inputs
    draft, notes = state.get("draft") or "", state.get("research_notes") or ""
    scale = {}
    if draft:
        scale["draft_citations"] = len(arxiv_ids(draft))
    if notes:
        scale["notes_sources"] = len(arxiv_ids(notes))
        scale["notes_chars"] = len(notes)
    return scale


def reference_of(s: Scenario) -> dict[str, Any]:
    mapping = _EXPECT_TO_REFERENCE.get(s.kind, {})
    unknown = set(s.expect) - set(mapping)
    if unknown:
        raise ValueError(f"{s.id}: expect keys with no reference field: {sorted(unknown)}")
    reference = {**_IMPLICIT_REFERENCE.get(s.kind, {}),
                 **{mapping[k]: v for k, v in s.expect.items()}}
    if s.kind == "supervisor" and s.premise_facts:
        reference["premise_facts"] = list(s.premise_facts)
    return reference


def explicit_inputs(s: Scenario) -> dict[str, Any]:
    """The inputs, with the defaults the runner would fill in written out.

    Part A's Critic scenarios leave the question and notes to the runner's
    defaults. An example states everything its model is shown, so the hash
    changes if a default does, and `scale` measures the notes the Critic
    actually judges against.
    """
    from research_copilot.live_check import scenarios as sc

    if s.kind == "critic":
        return {"question": sc.QUESTION, "research_notes": sc.NOTES, "unsupported_citations": [], **s.inputs}
    return dict(s.inputs)


def from_scenario(s: Scenario) -> Example:
    if s.id not in PROVENANCE:
        raise ValueError(f"{s.id}: no PROVENANCE entry - say where this example came from")
    p = PROVENANCE[s.id]
    inputs = explicit_inputs(s)
    return Example(
        id=s.id, item=s.item, kind=s.kind, title=s.title, inputs=inputs,
        reference=reference_of(s), source=p.source, designed_after=p.designed_after,
        control_of=p.control_of, scale_of=p.scale_of, scale=measure_scale(inputs),
        provider_only="anthropic" if s.item in ANTHROPIC_ONLY_ITEMS else "",
        label_disputed=s.label_disputed, note=s.note, est_tokens=s.est_tokens,
    )


def validate(examples: list[Example]) -> None:
    """The schema's rules, checked on the whole set (relations need both ends)."""
    by_id = {e.id: e for e in examples}
    if len(by_id) != len(examples):
        raise ValueError("duplicate example ids")
    for e in examples:
        if e.kind not in REFERENCE_KEYS:
            raise ValueError(f"{e.id}: unknown kind {e.kind!r}")
        extra = set(e.reference) - REFERENCE_KEYS[e.kind]
        if extra:
            raise ValueError(f"{e.id}: reference keys not in the {e.kind} schema: {sorted(extra)}")
        if not e.reference:
            raise ValueError(f"{e.id}: empty reference - nothing to score against")
        if e.source not in SOURCES:
            raise ValueError(f"{e.id}: unknown source {e.source!r}")
        for relation in ("control_of", "scale_of"):
            other = getattr(e, relation)
            if other and (other not in by_id or by_id[other].kind != e.kind):
                raise ValueError(f"{e.id}: {relation}={other!r} is not an example of the same kind")
        if e.scale_of:
            small = by_id[e.scale_of]
            # A scale sibling tests the same defect, bigger. Same verdict, and
            # larger on the axis that makes it hard.
            if e.reference.get("verdict") != small.reference.get("verdict"):
                raise ValueError(f"{e.id}: scale sibling of {small.id} expects a different verdict")
            if e.scale.get("draft_citations", 0) <= small.scale.get("draft_citations", 0):
                raise ValueError(f"{e.id}: not larger than {small.id}")


def build_examples(scenarios: list[Scenario] = SCENARIOS) -> list[Example]:
    unlisted = {s.id for s in scenarios} ^ set(PROVENANCE) if scenarios is SCENARIOS else set()
    if unlisted:
        raise ValueError(f"PROVENANCE and SCENARIOS disagree: {sorted(unlisted)}")
    examples = [from_scenario(s) for s in scenarios]
    validate(examples)
    return examples


def dataset_version(examples: list[Example]) -> str:
    return _hash(sorted((e.id, e.example_hash) for e in examples))


EXAMPLES: list[Example] = build_examples()
