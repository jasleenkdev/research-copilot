# Live verification report

Provider(s): groq - model(s): openai/gpt-oss-120b
Scenarios recorded: 5 of 39; tokens used: 11743

Rate limits reported by the provider for this account:
  x-ratelimit-limit-requests: 1000
  x-ratelimit-limit-tokens: 8000
  x-ratelimit-remaining-requests: 999
  x-ratelimit-remaining-tokens: 7927
  x-ratelimit-reset-requests: 1m26.4s
  x-ratelimit-reset-tokens: 547ms

NOT YET RUN: P01, P02, S00, S00g, SUP01, SUP02, SUP03, SUP04, SUP05, SUP06, SUP07, SUP08, SUP09, SUP10, SUP11, SUP12, SUP13, SUPC02, SUPC06, SUPC08, SUPC10, RES01, RES02, WRT01, WRT02, WRT03, WRT04, CRT02, CRT03, CRT04, CRT05, CRT06, E2E01, E2E02

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
  [pass] CRT01 Good draft, real citation (1077 tokens, 1.7s)
      verdict=approve checks=[{'arxiv_id': '2309.15217', 'status': 'found'}]
      critique: (approved by the critic)
  [pass] CRT07 Real paper not in the notes, WITH the Writer's flag (1054 tokens, 6.7s)
      verdict=reject checks=[{'arxiv_id': '2309.15217', 'status': 'found'}, {'arxiv_id': '2311.09476', 'status': 'found'}]
      critique: Evidence problem – the draft cites ARES (arXiv 2311.09476) as “another option,” but this source is not listed in the research notes, so the claim is unsupported.
  [pass] CRT08 CONTROL: the same draft WITHOUT the flag (1110 tokens, 7.1s)
      verdict=reject checks=[{'arxiv_id': '2309.15217', 'status': 'found'}, {'arxiv_id': '2311.09476', 'status': 'found'}]
      critique: The draft cites ARES (arXiv:2311.09476), but this source is not listed in the provided research notes, so the claim “ARES is another option” is unsupported by the available evidence. This is an evidence problem that must be fixed (remove the unsupported citation or add appropriate research).
  [pass] CRT09 E2E01 scale: 13 citations, one unresearched, WITH the flag (4248 tokens, 33.9s)
      verdict=reject checks=[{'arxiv_id': '2504.14891', 'status': 'found'}, {'arxiv_id': '2405.07437', 'status': 'found'}, {'arxiv_id': '2504.17137', 'status': 'found'}, {'arxiv_id': '2404.13781', 'status': 'found'}, {'arxiv_id': '2507.06956', 'status': 'found'}, {'arxiv_id': '2309.15217', 'status': 'found'}, {'arxiv_id': '2512.04343', 'status': 'found'}, {'arxiv_id': '2510.24870', 'status': 'found'}, {'arxiv_id': '2409.03759', 'status': 'found'}, {'arxiv_id': '2408.12398', 'status': 'found'}]
      critique: The draft cites arXiv:2408.12398 (“A Comparative Analysis of Faithfulness Metrics and Humans in Citation Evaluation”), but this paper is not listed in the provided research notes. This is an evidence problem – the claim relying on that citation lacks a supported source and must be removed or replace
  [fail] CRT10 CONTROL at E2E01 scale: the same, WITHOUT the flag - failed: verdict, critique_names_2408.12398 (4254 tokens, 31.8s)
      verdict=approve checks=[{'arxiv_id': '2504.14891', 'status': 'found'}, {'arxiv_id': '2405.07437', 'status': 'found'}, {'arxiv_id': '2504.17137', 'status': 'found'}, {'arxiv_id': '2404.13781', 'status': 'found'}, {'arxiv_id': '2507.06956', 'status': 'found'}, {'arxiv_id': '2309.15217', 'status': 'found'}, {'arxiv_id': '2512.04343', 'status': 'found'}, {'arxiv_id': '2510.24870', 'status': 'found'}, {'arxiv_id': '2409.03759', 'status': 'found'}, {'arxiv_id': '2408.12398', 'status': 'found'}]
      critique: (approved by the critic)

## Item 7: Cost / latency (end to end)
(not run)

