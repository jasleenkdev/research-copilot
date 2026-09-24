# Part A live verification: summary (Groq, openai/gpt-oss-120b)

Two days, four result files, one provider. Read with the per-run reports beside it:
`groq-gpt-oss-120b-day1.md` (baseline), `-v2.md` (after the premise fix), `-v4.md`
(after the Researcher fixes). Raw JSONL is under `data/live_check/` (gitignored).

**What these results are evidence of:** prompt and logic quality on one open model.
**What they are not evidence of:** anything Anthropic-specific. Item 1 is pending.

| README item | Status | Evidence |
| --- | --- | --- |
| 0 portability | not run | Every model comes from `get_chat_model()`; the whole graph ran on Groq unchanged except the structured-output method |
| 1 json_schema + fallback beta | **PENDING (Anthropic only)** | Not answerable on Groq |
| 1g Groq structured output | pass | strict json_schema parses; **`next` generated before `rationale`**: schema order is not generation order |
| 2 Supervisor classification | 12/17 strict, 16/17 lenient (v2) | Single samples; 4 routes flipped between runs with no targeted change |
| 3 rationale vs. state | wrong-premise 2 -> 0 after the premise instruction; scope-creep 1 -> 2 | Two-axis review, `reviews-*.json` |
| 4 Researcher | RES01 pass, RES02 fail (v4) | Stops only via the reserved final call; restates known facts instead of NOTHING NEW |
| 5 Writer | 4/4 (day one) | E2E02 shows gpt-oss citation markup leaking into user-facing answers |
| 6 Critic | 6/6 | Reads lookup titles, ignores ERROR, catches fake/malformed/misattributed ids |
| 7 end to end | E2E01 **withheld**, E2E02 answered | See "the E2E01 failure" |
| 8 real arXiv | pass | Nonexistent id -> empty feed -> NOT FOUND (as assumed); shared throttle held |

## Findings, by kind

### Fixed during Part A
1. **One invented tool call killed the run.** gpt-oss called `open_file` (its trained browsing tool).
   Fixed with `resilience.py`: one retry with a hint, then a degraded result. Also a prompt line.
2. **Wrong-premise rationales** (SUP04, SUP08): the Supervisor repeated a critique's false claim.
   Fixed by instructing it to check a critique's claims against the state. 2 -> 0.
3. **The Researcher never ended its own loop.** Fixed by reserving the last call, which is made
   with *no tools* and the results as text. `tool_choice="none"` was tried first and failed twice:
   gpt-oss ignored it, and langchain-anthropic 1.7 maps `"none"` to a forced tool named "none".
4. **Harness bugs** (not model failures): gap-admission regex (curly apostrophes), a vacuous
   "cites only notes" check, and rescored records dropping their token counts.

### Open, and needing a decision
5. **E2E01: a guard overruled a correct Supervisor.** The Critic's per-round budget (4 calls)
   covers about 3 citation checks, one per call on gpt-oss. The resulting fail-closed "reject" is
   procedural, but the graph treats it as a content rejection: it spends a revision, and the 6.3
   guard forces a rewrite of a draft nobody faulted. The Supervisor had diagnosed it and proposed
   the Critic. Two separate defects:
   - verification costs model calls it does not need to: an arXiv id is a lookup that code can do
     for every cited id *before* the Critic's one judging call
   - "did not finish" and "rejected" are one verdict. An incomplete review should not spend a
     revision or force a rewrite
6. **Scope creep** (Supervisor SUPC08/SUPC10, Critic CRT04): gpt-oss invents "cover the broader
   range of evaluation methods" requirements. In v2 this changed a route.
7. **Gap-vs-critique priority** (SUP06/SUP08): an open gap in the notes can outrank the critique.
   Left as-is by decision; revisit when there are more scenarios.
8. **The Researcher doesn't recognise an answer it already has** (RES02, both runs).
9. **gpt-oss citation markup** (`【1†L1-L7】`) in answers. Model-specific, but it defeats the
   Writer's "cite exactly as the notes give them" rule, and no check catches it.
10. **Single-sample variance** makes item 2 unscorable run-to-run. Deferred to Part E (decision).

## Cost and latency (Groq free tier: 8K tokens/min, 200K/day)
| Run | Tokens | Wall time | Note |
| --- | --- | --- | --- |
| E2E01 with Critic, max_revisions 1 | 69,597 | 449 s | mostly per-minute waits; delivered nothing |
| E2E02 without Critic | 14,427 | 57 s | 5 model calls |
| Researcher pass (RES01) | 14,240 | 41 s | 6 calls; searched 5 times for one paper |
| Supervisor decision | ~1,200-1,900 | 1-2 s | per call |

Total Part A spend: day one about 38k, day two about 132k tokens. No money.
