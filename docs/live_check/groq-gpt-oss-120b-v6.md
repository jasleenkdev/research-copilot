# Live verification report

Provider(s): groq - model(s): openai/gpt-oss-120b
Scenarios recorded: 1 of 35; tokens used: 48487

Rate limits reported by the provider for this account:
  x-ratelimit-limit-requests: 1000
  x-ratelimit-limit-tokens: 8000
  x-ratelimit-remaining-requests: 998
  x-ratelimit-remaining-tokens: 7927
  x-ratelimit-reset-requests: 2m52.8s
  x-ratelimit-reset-tokens: 547ms

NOT YET RUN: P01, P02, S00, S00g, SUP01, SUP02, SUP03, SUP04, SUP05, SUP06, SUP07, SUP08, SUP09, SUP10, SUP11, SUP12, SUP13, SUPC02, SUPC06, SUPC08, SUPC10, RES01, RES02, WRT01, WRT02, WRT03, WRT04, CRT01, CRT02, CRT03, CRT04, CRT05, CRT06, E2E02

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
(not run)

## Item 7: Cost / latency (end to end)
  [review] E2E01 Full run with the Critic (48487 tokens, 290.9s)
      routes=[['researcher', 'researcher', ''], ['writer', 'writer', ''], ['researcher', 'critic', 'researcher round budget (6 model calls) spent']] revisions=0 verdict=approve
      model calls: 12, largest request ~6010 tokens
        supervisor/supervisor       ~   727 tok  ok
        researcher/research_model   ~   404 tok  ok
        researcher/research_model   ~  2539 tok  ok
        researcher/research_model   ~  3636 tok  ok
        researcher/research_model   ~  5741 tok  ok
        researcher/research_model   ~  6010 tok  ok
        researcher/research_model   ~  5999 tok  ok
        supervisor/supervisor       ~  1838 tok  ok
            writer/writer           ~  1552 tok  ok
            writer/writer           ~  3079 tok  ok
        supervisor/supervisor       ~  2926 tok  ok
            critic/critic_model     ~  4468 tok  ok

