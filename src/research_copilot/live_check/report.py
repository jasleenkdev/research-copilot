"""Turn a results file into the item-by-item report. (Phase 7, A0)

The report is organised by README item, because the question it answers is
"which of the unverified claims now have evidence, and which way did it go?".

Three rules it enforces:
  - an Anthropic-only item on another provider is PENDING, whatever else ran.
    It is never "pass", and never inferred from the other items.
  - Supervisor accuracy is reported strict (== preferred) *and* lenient
    (in acceptable), never one number.
  - rationales are printed next to their routes in full. Item 3 is judged by
    reading them, and the report makes that reading possible without opening
    the JSONL.
"""

from collections import defaultdict

from research_copilot.live_check.reviews import RATIONALE_CATEGORIES
from research_copilot.live_check.scenarios import ANTHROPIC_ONLY_ITEMS, SCENARIOS, by_id

ITEM_TITLES = {
    "0": "Provider portability (Phase 1's promise)",
    "1": "json_schema + server-side-fallback beta (Anthropic only)",
    "1g": "This provider's structured-output path (NOT item 1)",
    "2": "Supervisor classification",
    "3": "Rationale vs. route (read by hand)",
    "4": "Researcher",
    "5": "Writer",
    "6": "Critic (+ real arXiv, item 8)",
    "7": "Cost / latency (end to end)",
}


def _route_outcome(r: dict) -> str:
    if r.get("status") == "error":
        return "error"
    if r["checks"].get("strict"):
        return "strict"
    return "lenient" if r["checks"].get("lenient") else "wrong"


def render(results: dict[str, dict], *, limits: dict | None = None,
           reviews: dict[str, dict] | None = None,
           baseline: dict[str, dict] | None = None) -> str:
    order = [s.id for s in SCENARIOS]
    by_item: dict[str, list[dict]] = defaultdict(list)
    for sid in order:
        if sid in results:
            by_item[results[sid]["item"]].append(results[sid])

    providers = sorted({r["provider"] for r in results.values()})
    models = sorted({r.get("model", "") for r in results.values()})
    lines = [
        "# Live verification report",
        "",
        f"Provider(s): {', '.join(providers) or '-'} - model(s): {', '.join(models) or '-'}",
        f"Scenarios recorded: {len(results)} of {len(order)}; "
        f"tokens used: {sum(r.get('tokens', 0) for r in results.values())}",
    ]
    if limits:
        lines += ["", "Rate limits reported by the provider for this account:"]
        lines += [f"  {k}: {v}" for k, v in sorted(limits.items())]

    missing = [sid for sid in order if sid not in results]
    if missing:
        lines += ["", f"NOT YET RUN: {', '.join(missing)}"]

    for item in ["0", "1", "1g", "2", "3", "4", "5", "6", "7"]:
        rows = by_item.get(item, [])
        if item == "3":
            rows = by_item.get("2", [])
        lines += ["", f"## Item {item}: {ITEM_TITLES[item]}"]

        if item in ANTHROPIC_ONLY_ITEMS and any(r["status"] == "pending" for r in rows):
            lines.append("PENDING - needs a real Anthropic key. Nothing a Groq run shows answers "
                         "this, and nothing here should be read as 'probably fine'.")
            continue
        if not rows:
            lines.append("(not run)")
            continue

        if item == "2":
            usable = [r for r in rows if r["status"] != "error"]
            strict = sum(bool(r["checks"].get("strict")) for r in usable)
            lenient = sum(bool(r["checks"].get("lenient")) for r in usable)
            lines.append(f"strict {strict}/{len(usable)}, lenient {lenient}/{len(usable)}, "
                         f"errors {len(rows) - len(usable)}")
            if baseline:
                base = [baseline[r["id"]] for r in usable if r["id"] in baseline and baseline[r["id"]]["status"] != "error"]
                if base:
                    lines.append(f"baseline (before this run's changes): strict "
                                 f"{sum(bool(b['checks'].get('strict')) for b in base)}/{len(base)}, lenient "
                                 f"{sum(bool(b['checks'].get('lenient')) for b in base)}/{len(base)}")
                    for r in usable:
                        b = baseline.get(r["id"])
                        if b and b.get("observed", {}).get("proposed") != r["observed"].get("proposed"):
                            lines.append(f"  CHANGED {r['id']}: {b['observed'].get('proposed')} -> {r['observed'].get('proposed')}")
            for r in rows:
                o = r.get("observed", {})
                mark = "OK " if r["checks"].get("strict") else ("~  " if r["checks"].get("lenient") else "XX ")
                disputed = " [label disputed]" if by_id().get(r["id"]) and by_id()[r["id"]].label_disputed else ""
                lines.append(f"  {mark}{r['id']} {r['title']}{disputed}: expected {r['expect'].get('preferred')}, "
                             f"proposed {o.get('proposed')}"
                             + (f" -> routed {o.get('routed_to')} ({o.get('override')})" if o.get("override") else "")
                             + (f" - ERROR {r['error']}" if r.get("error") else ""))
            continue

        if item == "3":
            reviews = reviews or {}
            if reviews:
                lines.append("Two axes per decision: ROUTE (scored by code) x RATIONALE (read "
                             "against the state by a person).")
                grid: dict[tuple[str, str], list[str]] = defaultdict(list)
                for r in rows:
                    category = reviews.get(r["id"], {}).get("category", "unreviewed")
                    grid[(_route_outcome(r), category)].append(r["id"])
                by_label: dict[str, list[str]] = defaultdict(list)
                for (route, category), ids in sorted(grid.items()):
                    label = {
                        ("strict", "sound"): "correct",
                        ("lenient", "sound"): "acceptable route, sound rationale",
                        ("wrong", "sound"): "misclassified route (rationale consistent)",
                        ("strict", "wrong_premise"): "WRONG PREMISE, acceptable route",
                        ("lenient", "wrong_premise"): "WRONG PREMISE, acceptable route",
                        ("wrong", "wrong_premise"): "WRONG PREMISE, misclassified route",
                        ("wrong", "ignores_critique"): "misclassified route (ignores the critique)",
                    }.get((route, category), f"route {route}, rationale {category}")
                    by_label[label].extend(ids)
                for label, ids in by_label.items():
                    lines.append(f"  {label}: {', '.join(ids)}")
                lines.append("  categories: " + "; ".join(f"{k} = {v}" for k, v in RATIONALE_CATEGORIES.items()))
            lines.append("")
            lines.append("Each rationale beside its route, with the facts it must be consistent with:")
            for r in rows:
                o = r.get("observed", {})
                scenario = by_id().get(r["id"])
                review = (reviews or {}).get(r["id"])
                lines.append(f"  {r['id']} -> {o.get('proposed')}"
                             + (f" (brief: {o.get('brief')})" if o.get("brief") else "")
                             + f"\n      {o.get('rationale', '(none)')}")
                for fact in (scenario.premise_facts if scenario else ()):
                    lines.append(f"      fact: {fact}")
                if review:
                    lines.append(f"      REVIEW: {review['category']}" + (f" - {review['note']}" if review.get("note") else ""))
            continue

        for r in rows:
            failed = [k for k, v in (r.get("checks") or {}).items() if not v]
            lines.append(
                f"  [{r['status']}] {r['id']} {r['title']}"
                + (f" - failed: {', '.join(failed)}" if failed else "")
                + (f" - ERROR {r['error']}" if r.get("error") else "")
                + f" ({r.get('tokens', 0)} tokens, {r.get('seconds', 0)}s)"
            )
            o = r.get("observed", {})
            if r["kind"] == "structured_smoke":
                lines.append(f"      method={o.get('method')} field_order={o.get('field_order')} "
                             f"parsed={o.get('parsed')}")
            if r["kind"] == "critic":
                lines.append(f"      verdict={o.get('verdict')} checks={o.get('citation_checks')}")
                lines.append(f"      critique: {(o.get('critique') or '')[:300]}")
            if r["kind"] == "e2e":
                if o.get("routes") is not None:
                    lines.append(f"      routes={o.get('routes')} revisions={o.get('revisions')} "
                                 f"verdict={o.get('verdict')}")
                if o.get("calls"):
                    lines.append(f"      model calls: {len(o['calls'])}, largest request ~{o.get('largest_request')} tokens")
                    for c in o["calls"]:
                        lines.append(f"        {c['agent']:>10}/{c['node']:<16} ~{c['est_tokens']:>6} tok  {c['status']}")
                if o.get("failed_call"):
                    fc = o["failed_call"]
                    lines.append(f"      FAILED CALL: {fc['agent']}/{fc['node']} (~{fc['est_tokens']} tokens)")
                for t in o.get("trims") or []:
                    lines.append(f"      TRIMMED: {t['node']} {t['part']} {t['tokens_before']}->{t['tokens_after']} (limit {t['limit']})")
    return "\n".join(lines) + "\n"
