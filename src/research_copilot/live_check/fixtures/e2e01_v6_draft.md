**How RAG pipelines are evaluated**

RAG (Retrieval‑Augmented Generation) systems are judged on four main dimensions: (1) how well the retrieval component finds relevant evidence, (2) how well the generator produces fluent and accurate text, (3) how faithfully the answer is grounded in the retrieved material, and (4) how efficiently the whole pipeline runs.  The most widely used automatic metrics and human‑evaluation practices are listed below, together with the sources that introduce or apply them.

---

### 1. Retrieval‑quality metrics  
* **Recall@k, Precision@k, MRR, nDCG** – standard relevance measures for the top‑k retrieved passages.  These appear in the two recent survey papers and in the MIRAGE benchmark [Retrieval Augmented Generation Evaluation in the Era of Large Language Models: A Comprehensive Survey – http://arxiv.org/abs/2504.14891v1] [Evaluation of Retrieval‑Augmented Generation: A Survey – http://arxiv.org/abs/2405.07437v2] [MIRAGE: A Metric‑Intensive Benchmark for Retrieval‑Augmented Generation Evaluation – http://arxiv.org/abs/2504.17137v1].

* **Fine‑grained per‑document impact** – the eRAG approach evaluates each retrieved document individually by feeding it to the LLM and measuring generation scores (BLEU, ROUGE, Exact Match) per document, exposing how retrieval quality translates into answer quality [Evaluating Retrieval Quality in Retrieval‑Augmented Generation – http://arxiv.org/abs/2404.13781v1].

* **Robustness to query perturbations** – a study measures how small changes in the query affect Recall@k and MRR and then observes downstream drops in Exact Match and F1 [Investigating the Robustness of Retrieval‑Augmented Generation at the Query Level – http://arxiv.org/abs/2507.06956v1].

---

### 2. Generation‑quality metrics  
* **Lexical overlap** – BLEU, ROUGE, METEOR.  
* **Semantic similarity** – BERTScore (and the semantic variant of METEOR).  
* **Task‑specific token metrics** – F1 and Exact Match (common in QA).  

These metrics are part of the reference‑free RAGAS suite and are also reported in Ragas, MIRAGE, and the Personalization Paradox study [Ragas: Automated Evaluation of Retrieval Augmented Generation – http://arxiv.org/abs/2309.15217v2] [MIRAGE: A Metric‑Intensive Benchmark for Retrieval‑Augmented Generation Evaluation – http://arxiv.org/abs/2504.17137v1] [The Personalization Paradox: Semantic Loss vs. Reasoning Gains in Agentic AI Q&A – http://arxiv.org/abs/2512.04343v1].

---

### 3. Grounding / factuality metrics  
* **Faithfulness scores** – RAGAS provides a set of reference‑free grounding metrics that assess how much of the answer is supported by the retrieved evidence [Ragas: Automated Evaluation of Retrieval Augmented Generation – http://arxiv.org/abs/2309.15217v2].

* **InfoF1 and CiteF1** – claim‑centric metrics introduced in the multimodal MIRAGE benchmark to measure factual coverage (InfoF1) and citation completeness (CiteF1).  Human judges evaluate these scores in the “Seeing Through the MiRAGE” paper [MIRAGE: A Metric‑Intensive Benchmark for Retrieval‑Augmented Generation Evaluation – http://arxiv.org/abs/2504.17137v1] [Seeing Through the MiRAGE: Evaluating Multimodal Retrieval Augmented Generation – http://arxiv.org/abs/2510.24870v3].

* **Safety / factuality checks** – VERA adds cross‑encoder validation and reports combined retrieval‑generation metrics together with safety/factuality assessments [VERA: Validation and Evaluation of Retrieval‑Augmented Systems – http://arxiv.org/abs/2409.03759v1].

* **Citation‑faithfulness studies** – a separate analysis of how well generated citations match the source material provides additional human‑grounded metrics [A Comparative Analysis of Faithfulness Metrics and Humans in Citation Evaluation – http://arxiv.org/abs/2408.12398v1].

---

### 4. Efficiency metrics  
* **Latency** – end‑to‑end response time, retrieval latency, and generation latency are logged in Ragas and VERA [Ragas: Automated Evaluation of Retrieval Augmented Generation – http://arxiv.org/abs/2309.15217v2] [VERA: Validation and Evaluation of Retrieval‑Augmented Systems – http://arxiv.org/abs/2409.03759v1].

* **Throughput** – measured as queries per second or tokens‑per‑second; reported anecdotally in code‑focused RAG work [DeepCodeSeek: Real‑Time API Retrieval for Context‑Aware Code Generation – http://arxiv.org/abs/2509.25716v1].

* **Retrieval depth vs. latency** – a few papers (e.g., DeepCodeSeek) give single‑system numbers (e.g., top‑40 accuracy and real‑time latency) but systematic trade‑off analyses are still missing [DeepCodeSeek: Real‑Time API Retrieval for Context‑Aware Code Generation – http://arxiv.org/abs/2509.25716v1].

---

### 5. Composite evaluation frameworks  

| Framework | What it bundles | Example citations |
|-----------|----------------|-------------------|
| **RAGAS** (used in Ragas and ragR) | Retrieval relevance (Recall@k, MRR), generation quality (BLEU, ROUGE, BERTScore, F1, Exact Match), factual grounding (faithfulness scores) | [Ragas: Automated Evaluation of Retrieval Augmented Generation – http://arxiv.org/abs/2309.15217v2] |
| **Ragas** | Separate reporting of retrieval metrics, generation metrics, and system latency/throughput | [Ragas: Automated Evaluation of Retrieval Augmented Generation – http://arxiv.org/abs/2309.15217v2] |
| **ragR** (R package) | Structured logging of the same RAGAS metrics for reproducible benchmarking | [ragR: Retrieval‑Augmented Generation and RAG Assessment in R – http://arxiv.org/abs/2604.23515v1] |
| **VERA** | Cross‑encoder validation, combined retrieval‑generation scores, safety/factuality checks, latency | [VERA: Validation and Evaluation of Retrieval‑Augmented Systems – http://arxiv.org/abs/2409.03759v1] |
| **MIRAGE** | Standard generation metrics + retrieval metrics (Recall@k, MRR, nDCG) + novel grounding metrics (InfoF1, CiteF1) for multimodal RAG | [MIRAGE: A Metric‑Intensive Benchmark for Retrieval‑Augmented Generation Evaluation – http://arxiv.org/abs/2504.17137v1] |

---

### 6. Human‑evaluation practices  

* **Expert factuality / grounding rating** – annotators judge how well each claim is supported by retrieved sources (survey papers) [Retrieval Augmented Generation Evaluation in the Era of Large Language Models: A Comprehensive Survey – http://arxiv.org/abs/2504.14891v1] [Evaluation of Retrieval‑Augmented Generation: A Survey – http://arxiv.org/abs/2405.07437v2].

* **Pairwise preference judgments** – two system outputs are shown side‑by‑side and judges pick the better one (common across surveys).

* **Likert‑scale assessments** – relevance, coherence, usefulness are rated on a 1‑5 scale (survey practice).

* **Citation‑support annotation** – judges mark whether each generated citation correctly corresponds to a retrieved document (citation‑faithfulness study) [A Comparative Analysis of Faithfulness Metrics and Humans in Citation Evaluation – http://arxiv.org/abs/2408.12398v1].

* **Multimodal claim evaluation** – human judges assess InfoF1 and CiteF1 scores for image‑text or video‑text RAG outputs [Seeing Through the MiRAGE: Evaluating Multimodal Retrieval Augmented Generation – http://arxiv.org/abs/2510.24870v3].

---

### 7. Domain‑specific and “what‑if” evaluations  

* **Code‑focused RAG** – DeepCodeSeek reports top‑k retrieval accuracy (e.g., 87.86 % at top‑40) together with real‑time latency to illustrate efficiency for API‑based code generation [DeepCodeSeek: Real‑Time API Retrieval for Context‑Aware Code Generation – http://arxiv.org/abs/2509.25716v1].

* **Metadata‑aware retrieval** – Experiments compare metadata‑driven retrieval strategies using Recall@k and downstream BLEU/ROUGE to quantify the trade‑off between richer retrieval and generation quality [Utilizing Metadata for Better Retrieval‑Augmented Generation – http://arxiv.org/abs/2601.11863v1].

* **Personalization paradox** – The study applies lexical (BLEU, ROUGE‑L), semantic (METEOR, BERTScore), and grounding (RAGAS) metrics, and supplements them with expert human ratings of answer quality [The Personalization Paradox: Semantic Loss vs. Reasoning Gains in Agentic AI Q&A – http://arxiv.org/abs/2512.04343v1].

---

### 8. Remaining gaps  

* **Standardized latency/throughput reporting** – most papers give only anecdotal timing or a single system’s numbers; no community‑wide benchmark exists.  
* **Unified human grounding/factuality protocol** – studies use varied annotator setups, scales, and citation‑support definitions, so results are not directly comparable.  
* **Multimodal RAG evaluation** – MIRAGE is one of the few benchmarks; comprehensive retrieval metrics for images, audio, or video are still lacking.  
* **Systematic analysis of retrieval depth vs. latency** – few works (e.g., DeepCodeSeek) provide isolated numbers, but a broader trade‑off study across large‑scale deployments is absent.

---

**In short**, evaluation of RAG pipelines today combines a suite of automatic relevance, generation, and grounding metrics (Recall@k, BLEU, BERTScore, InfoF1, etc.), efficiency measurements (latency, throughput), and human judgments (factuality, preference, Likert ratings).  Frameworks such as RAGAS, Ragas, ragR, VERA, and MIRAGE package these components for reproducible benchmarking, while domain‑specific papers illustrate how the same metrics can be applied to code generation, metadata‑enhanced retrieval, or multimodal settings.  The field still needs standardized latency reporting, a shared human‑grounding rubric, and richer multimodal benchmarks.