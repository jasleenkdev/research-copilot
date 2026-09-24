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

from research_copilot.live_check.scenarios import ANTHROPIC_ONLY_ITEMS, SCENARIOS

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


def render(results: dict[str, dict], *, limits: dict | None = None) -> str:
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
            for r in rows:
                o = r.get("observed", {})
                mark = "OK " if r["checks"].get("strict") else ("~  " if r["checks"].get("lenient") else "XX ")
                lines.append(f"  {mark}{r['id']} {r['title']}: expected {r['expect'].get('preferred')}, "
                             f"proposed {o.get('proposed')}"
                             + (f" -> routed {o.get('routed_to')} ({o.get('override')})" if o.get("override") else "")
                             + (f" - ERROR {r['error']}" if r.get("error") else ""))
            continue

        if item == "3":
            lines.append("Each rationale beside its route. Judge: does the reasoning support the choice?")
            for r in rows:
                o = r.get("observed", {})
                lines.append(f"  {r['id']} -> {o.get('proposed')}"
                             + (f" (brief: {o.get('brief')})" if o.get("brief") else "")
                             + f"\n      {o.get('rationale', '(none)')}")
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
                lines.append(f"      routes={o.get('routes')} revisions={o.get('revisions')} "
                             f"verdict={o.get('verdict')}")
    return "\n".join(lines) + "\n"
