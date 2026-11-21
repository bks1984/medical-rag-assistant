# Evaluation

## Method

Each of the four use-case questions was answered three ways: by the base LLM, with prompt engineering, and with RAG. An LLM-as-a-judge then rated each answer from 1 to 5 on:

| Metric | Definition |
|---|---|
| **Groundedness** | The answer is derived only from the retrieved manual context |
| **Relevance** | The answer covers all the important aspects of the question, and only those |

## Results from this codebase (Groq)

Setup: `openai/gpt-oss-120b` on Groq (reasoning effort low) for answers and judging, `BAAI/bge-small-en-v1.5` local embeddings, 500/50-token chunks, k = 3. The full answers are in [evaluation_results_groq.csv](evaluation_results_groq.csv).

| Question | Base: G / R | Prompt-eng: G / R | RAG: G / R |
|---|---|---|---|
| Sepsis (Critical Care) | 2 / 5 | 2 / 5 | **5 / 5** |
| Appendicitis (Surgery) | 2 / 5 | 2 / 5 | **5 / 5** |
| Patchy hair loss (Dermatology) | 2 / 5 | 2 / 5 | **5 / 5** |
| Brain injury (Neurology) | 1 / 5 | 1 / 5 | **5** / 4 |
| **Mean** | **1.75 / 5.00** | **1.75 / 5.00** | **5.00 / 4.75** |

Every score was parsed. There are no blanks, unlike the notebook.

- **Groundedness separates the approaches sharply.** Answers without retrieval are relevant and read well, but they are not supported by the manual. RAG answers are fully grounded and cite the chapter and printed page, for example *Chapter 227, Sepsis & Septic Shock, pp. 2444–2445* and *Chapter 11, pp. 164–165* for appendicitis.
- **Neurology relevance was 4.** The answer covered rehabilitation from Ch. 324 and Ch. 350 but not acute management (ICP control, surgery), because 2 of the 3 retrieved excerpts were rehab-focused. Retrieving more candidates (k = 8) and re-ranking them should fix this. See *Next experiments* below.

## Notebook results

| Question | Base: G / R | Prompt-eng: G / R | RAG: G / R |
|---|---|---|---|
| Sepsis (Critical Care) | 5 / 3 | 2 / 3 | 5 / – |
| Appendicitis (Surgery) | 5 / 3 | 5 / 5 | 5 / 5 |
| Patchy hair loss (Dermatology) | 5 / 5 | 5 / 5 | 3 / – |
| Brain injury (Neurology) | 5 / – | 5 / 5 | 5 / 5 |
| **Mean** | **5.00 / 3.67** | **4.25 / 4.50** | **4.50 / 5.00** |

"–" means the score could not be parsed (NaN). Means are taken over the parsed scores only.

## Interpretation

- **RAG wins on relevance (5.0).** The retriever finds the right sections, and the answers stick to the question.
- **RAG is the only approach with verifiable sources.** Its answers cite chapter and page, for example *Ch. 222, Approach to the Critically Ill Patient* for sepsis and *Ch. 11, p. 163* for appendicitis.
- **The base LLM's 5.0 groundedness is an artefact.** In the notebook the judge saw the retrieved context but was asked to rate answers that never used it. Generic medical text often overlaps with the manual and is "consistent" with it, without being *derived* from it.
- **The dermatology RAG answer scored 3 on groundedness.** The 500-token chunks split the alopecia areata section, so part of the answer went beyond the retrieved text.

## Problems found in the notebook and fixed here

| Issue | Effect | Fix in `medical_rag/evaluate.py` |
|---|---|---|
| Scores were read with `reply[-2]`, the second-to-last *character* | NaN whenever the judge's reply didn't end in `N}`. This explains the missing values above | `parse_score()` uses a regex on `groundedness_score: N`, with a fallback |
| Prompt-eng Q4 was scored with `questions[2]` and `responses[2]` | The neurology score was really a second dermatology score | Each answer is paired with its own question in a loop |
| The judge model was `gpt-3.5-turbo` (retired) | Weaker and inconsistent rating | `JUDGE_MODEL` defaults to the chat model (gpt-oss-120b on Groq, `gpt-4o-mini` on OpenAI) and can be configured |
| Context was joined with `". "` and had no source labels | The model could not cite pages reliably | Excerpts are labelled `[Excerpt n | manual, page N]` |

Run `python scripts/evaluate.py` to reproduce the scores with these fixes.

## Next experiments

1. Chunk size 800–1,000 tokens with 150–200 overlap: `python scripts/ingest.py --chunk-size 900 --chunk-overlap 180`
2. Retrieve k = 8, then re-rank to the top 3 with a cross-encoder.
3. Hybrid BM25 + vector retrieval for exact drug and condition names.
4. A 30–50 question benchmark, with a clinician-written reference answer for each question.
