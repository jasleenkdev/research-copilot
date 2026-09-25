# Live verification report

Provider(s): groq - model(s): openai/gpt-oss-120b
Scenarios recorded: 2 of 39; tokens used: 2255

Rate limits reported by the provider for this account:
  x-ratelimit-limit-requests: 1000
  x-ratelimit-limit-tokens: 8000
  x-ratelimit-remaining-requests: 999
  x-ratelimit-remaining-tokens: 7927
  x-ratelimit-reset-requests: 1m26.4s
  x-ratelimit-reset-tokens: 547ms

NOT YET RUN: P01, P02, S00, S00g, SUP01, SUP02, SUP03, SUP04, SUP05, SUP06, SUP07, SUP08, SUP09, SUP10, SUP11, SUP12, SUP13, SUPC02, SUPC06, SUPC08, SUPC10, RES01, RES02, WRT01, WRT02, WRT03, WRT04, CRT02, CRT03, CRT04, CRT05, CRT06, CRT08, CRT09, CRT10, E2E01, E2E02

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
(not run)

## Item 5: Writer
(not run)

## Item 6: Critic (+ real arXiv, item 8)
  [pass] CRT01 Good draft, real citation (1032 tokens, 1.2s)
      verdict=approve checks=[{'arxiv_id': '2309.15217', 'status': 'found'}]
      critique: (approved by the critic)
  [pass] CRT07 Real paper not in the notes, WITH the Writer's flag (1223 tokens, 6.5s)
      verdict=reject checks=[{'arxiv_id': '2309.15217', 'status': 'found'}, {'arxiv_id': '2311.09476', 'status': 'found'}]
      critique: Evidence problem: the draft mentions “ARES [2311.09476] is another option,” but the citation 2311.09476 (ARES) is not present in the provided research notes, so this claim is unsupported.

## Item 7: Cost / latency (end to end)
(not run)

