"""From samples to a claim: k of n, with an honest interval. (Phase 7, E2)

CONCEPT: the Wilson interval, and why not just k/n
3 passes out of 3 reads as "100%", but three samples cannot show that. The
Wilson score interval is the range of pass rates that plausibly produce k
passes in n tries (95% here). For 3/3 it is roughly 44%-100%, and for 2/3
roughly 21%-94%. Unlike the textbook normal approximation, it stays inside
[0, 1] and does not collapse to a zero-width interval at 0/n or n/n, which is
exactly where small samples live. So a report states the interval, and two
configurations are called different only when their intervals do not overlap.

CONCEPT: four classes, and one of them is "not enough data"
  insufficient  fewer than min_n counted samples: shown, never as a rate
  stable_pass   every counted sample passed
  stable_fail   none did
  flaky         some did. Where E2's approval says to sample deeper later
Infra samples are counted beside the score, never inside it. Samples whose
example_hash no longer matches (the example changed) are counted as stale
and left out. Legacy samples have a config of their own, so they never mix
with a current one.

Scoring is a parameter. `score(example, record) -> {check: bool}` comes from
E3's evaluators, and a sample passes when every check passes.
"""

import math
import statistics
from collections.abc import Callable
from dataclasses import dataclass, field

from research_copilot.evaluation.dataset import Example
from research_copilot.evaluation.samples import CRASH, INFRA, MEASURED

Scorer = Callable[[Example, dict], dict[str, bool]]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


@dataclass
class Summary:
    example_id: str
    n: int                       # counted: measured + crash
    k: int                       # passed
    interval: tuple[float, float]
    cls: str
    infra: int = 0
    crash: int = 0
    stale: int = 0
    infra_reasons: list[str] = field(default_factory=list)
    checks: dict[str, tuple[int, int]] = field(default_factory=dict)   # check -> (passed, of)
    tokens: tuple[int, int, int] | None = None                         # (median, min, max)
    seconds: float | None = None                                       # median


def classify_rate(k: int, n: int, min_n: int) -> str:
    if n < min_n:
        return "insufficient"
    return "stable_pass" if k == n else "stable_fail" if k == 0 else "flaky"


def summarize(examples: list[Example], samples: list[dict], config_fp: str, score: Scorer,
              *, min_n: int = 3) -> list[Summary]:
    out = []
    for e in examples:
        mine = [s for s in samples if s["example_id"] == e.id and s["config_fp"] == config_fp]
        current = [s for s in mine if s["example_hash"] == e.example_hash]
        counted = [s for s in current if s["outcome"] in (MEASURED, CRASH)]
        passed, checks = 0, {}
        for s in counted:
            if s["outcome"] == CRASH:
                continue            # a crash is a failure, with no checks to show
            result = score(e, s["record"])
            passed += bool(result) and all(result.values())
            for name, ok in result.items():
                got, of = checks.get(name, (0, 0))
                checks[name] = (got + bool(ok), of + 1)
        tokens = [s["record"].get("tokens", 0) for s in counted]
        seconds = [s["record"].get("seconds", 0) for s in counted]
        infra = [s for s in current if s["outcome"] == INFRA]
        out.append(Summary(
            example_id=e.id, n=len(counted), k=passed, interval=wilson(passed, len(counted)),
            cls=classify_rate(passed, len(counted), min_n),
            infra=len(infra), crash=sum(s["outcome"] == CRASH for s in counted),
            stale=len(mine) - len(current), infra_reasons=[s.get("outcome_reason", "") for s in infra],
            checks=checks,
            tokens=(int(statistics.median(tokens)), min(tokens), max(tokens)) if tokens else None,
            seconds=statistics.median(seconds) if seconds else None,
        ))
    return out


def infra_rate(samples: list[dict], config_fp: str) -> tuple[int, int]:
    """(infra samples, all samples) for one config: the system's reliability,
    reported beside the scores and never inside them."""
    mine = [s for s in samples if s["config_fp"] == config_fp]
    return sum(s["outcome"] == INFRA for s in mine), len(mine)


def differs(a: Summary, b: Summary) -> bool:
    """Only when the intervals do not overlap, and both have enough data."""
    if "insufficient" in (a.cls, b.cls):
        return False
    return a.interval[1] < b.interval[0] or b.interval[1] < a.interval[0]
