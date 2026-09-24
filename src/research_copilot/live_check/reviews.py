"""Human review of Supervisor rationales: the second axis. (Phase 7 A1)

CONCEPT: two independent judgements per routing decision
  route      was the chosen agent right? Scored by code against the
             scenario's expected route: strict / lenient / wrong.
  rationale  does the stated reasoning hold up against the state the
             Supervisor was shown? Judged by a person (later, Part E's LLM
             judge, checked against these human reads).

Day one on Groq showed why they must stay separate. SUP08 and SUP04 got an
acceptable route with a rationale built on a false premise - "the claim lacks
a supporting citation" when the notes held one, "find the correct citation for
BENCH-RAG" when nothing says BENCH-RAG exists. Scored on route alone they
pass. Scored on rationale-vs-route agreement they also pass, because the
rationale does argue for the route it took. Only checking the rationale
against the *state* catches them.

Categories for the rationale axis:
"""

import json
from pathlib import Path

RATIONALE_CATEGORIES = {
    "sound": "consistent with the state, and addresses the actual complaint",
    "wrong_premise": "asserts or assumes something the state contradicts or never established "
                     "(e.g. repeats a critique's false claim without checking the notes)",
    "ignores_critique": "consistent with the state, but routes on something other than the "
                        "critique's actual complaint (e.g. an open gap outranks a style critique)",
    "scope_creep": "adds requirements that neither the critique nor the question raised",
}


def load_reviews(path: Path | None) -> dict[str, dict]:
    if not path or not Path(path).exists():
        return {}
    data = json.loads(Path(path).read_text())
    reviews = {k: v for k, v in data.items() if not k.startswith("_")}
    unknown = {v["category"] for v in reviews.values()} - set(RATIONALE_CATEGORIES)
    if unknown:
        raise ValueError(f"unknown review categories: {sorted(unknown)}")
    return reviews
