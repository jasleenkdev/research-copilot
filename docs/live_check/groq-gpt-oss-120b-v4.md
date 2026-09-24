# Live verification report

Provider(s): groq - model(s): openai/gpt-oss-120b
Scenarios recorded: 4 of 35; tokens used: 102755

Rate limits reported by the provider for this account:
  x-ratelimit-limit-requests: 1000
  x-ratelimit-limit-tokens: 8000
  x-ratelimit-remaining-requests: 986
  x-ratelimit-remaining-tokens: 7927
  x-ratelimit-reset-requests: 20m9.6s
  x-ratelimit-reset-tokens: 547ms

NOT YET RUN: P01, P02, S00, S00g, SUP01, SUP02, SUP03, SUP04, SUP05, SUP06, SUP07, SUP08, SUP09, SUP10, SUP11, SUP12, SUP13, SUPC02, SUPC06, SUPC08, SUPC10, WRT01, WRT02, WRT03, WRT04, CRT01, CRT02, CRT03, CRT04, CRT05, CRT06

## Item 0: Provider portability (Phase 1's promise)
(not run)

## Item 1: json_schema + server-side-fallback beta (Anthropic only)
(not run)

## Item 1g: This provider's structured-output path (NOT item 1)
(not run)

## Item 2: Supervisor classification
(not run)

## Item 3: Rationale vs. route (read by hand)
(not run)

## Item 4: Researcher
  [pass] RES01 Notes format, sources copied verbatim (14240 tokens, 41.0s)
  [fail] RES02 Follow-up on already-covered ground -> NOTHING NEW - failed: outcome (4491 tokens, 6.8s)

## Item 5: Writer
(not run)

## Item 6: Critic (+ real arXiv, item 8)
(not run)

## Item 7: Cost / latency (end to end)
  [review] E2E01 Full run with the Critic (69597 tokens, 449.0s)
      routes=[['researcher', 'researcher', ''], ['writer', 'writer', ''], ['critic', 'critic', ''], ['critic', 'writer', 'critic proposed on a draft that needs (re)writing first'], ['critic', 'critic', '']] revisions=1 verdict=reject
  [review] E2E02 Full run without the Critic (14427 tokens, 56.9s)
      routes=[['researcher', 'researcher', ''], ['writer', 'writer', ''], ['finish', 'finish', '']] revisions=0 verdict=

