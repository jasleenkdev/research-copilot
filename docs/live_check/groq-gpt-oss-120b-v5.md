# Live verification report

Provider(s): groq - model(s): openai/gpt-oss-120b
Scenarios recorded: 7 of 35; tokens used: 46293

Rate limits reported by the provider for this account:
  x-ratelimit-limit-requests: 1000
  x-ratelimit-limit-tokens: 8000
  x-ratelimit-remaining-requests: 955
  x-ratelimit-remaining-tokens: 7927
  x-ratelimit-reset-requests: 1h4m48s
  x-ratelimit-reset-tokens: 547ms

NOT YET RUN: P01, P02, S00, S00g, SUP01, SUP02, SUP03, SUP04, SUP05, SUP06, SUP07, SUP08, SUP09, SUP10, SUP11, SUP12, SUP13, SUPC02, SUPC06, SUPC08, SUPC10, RES01, RES02, WRT01, WRT02, WRT03, WRT04, E2E02

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
  [pass] CRT01 Good draft, real citation (881 tokens, 0.9s)
      verdict=approve checks=[{'arxiv_id': '2309.15217', 'status': 'found'}]
      critique: (approved by the critic)
  [pass] CRT02 Fabricated arXiv id (1123 tokens, 7.1s)
      verdict=reject checks=[{'arxiv_id': '2309.15217', 'status': 'found'}, {'arxiv_id': '2401.99999', 'status': 'not_found'}]
      critique: Evidence problem: the draft cites BENCH‑RAG [2401.99999] and claims it “standardises these metrics,” but no such arXiv paper exists (lookup says NOT FOUND) and the research notes contain no information about BENCH‑RAG, so the claim is unsupported.
  [pass] CRT03 Real id, wrong paper (Attention cited as RAGAS) (1079 tokens, 1.7s)
      verdict=reject checks=[{'arxiv_id': '1706.03762', 'status': 'found'}]
      critique: *Evidence*: The draft cites RAGAS as [1706.03762], but that arXiv ID corresponds to “Attention Is All You Need”, not the RAGAS paper (which is 2309.15217). The claim that RAG pipelines are evaluated with RAGAS is not supported by the provided citation. The draft must use the correct reference (2309.
  [pass] CRT04 Real citation, claim not in the notes (1180 tokens, 3.4s)
      verdict=reject checks=[{'arxiv_id': '2309.15217', 'status': 'found'}]
      critique: The draft makes an unsupported claim that “RAGAS agrees with human raters 95 % of the time” and cites 2309.15217 for it, but the research notes explicitly state there is no evidence on how well RAGAS aligns with human judgment. This is an evidence problem. Additionally, the answer only mentions RAGA
  [pass] CRT05 Every lookup fails (network) - must not reject for that (941 tokens, 0.7s)
      verdict=approve checks=[{'arxiv_id': '2309.15217', 'status': 'error'}]
      critique: (approved by the critic)
  [pass] CRT06 Malformed arXiv id (936 tokens, 1.0s)
      verdict=reject checks=[{'arxiv_id': '12345', 'status': 'invalid'}]
      critique: The draft uses an invalid arXiv identifier for RAGAS (“[arXiv:12345]”). The correct citation should be arXiv:2309.15217. This citation error means the claim is not properly supported. (Evidence)

## Item 7: Cost / latency (end to end)
  [error] E2E01 Full run with the Critic - ERROR RateLimitError: Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01m39avnrzevttbzvea348jyn2` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199939, Requested 6726. Please try again in 47m59.28s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}} (40153 tokens, 215.9s)
      routes=None revisions=None verdict=None

