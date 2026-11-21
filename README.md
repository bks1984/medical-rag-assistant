# 🩺 Medical RAG Assistant

A **Retrieval-Augmented Generation (RAG)** assistant that answers clinical questions using the
**Merck Manual of Diagnosis & Therapy**, citing the pages it used. It was built to help healthcare
professionals cope with information overload and get fast, evidence-based answers from a trusted source.

The project starts from a Jupyter notebook ([`notebooks/medical_assistant_rag.ipynb`](notebooks/medical_assistant_rag.ipynb))
and turns it into a re-usable Python package. It has a command-line tool, a Streamlit chat UI, an evaluation
harness and unit tests.

---

## 1. Problem statement

**Business context.** Clinicians face growing volumes of medical data and must make time-critical decisions.
Searching large references such as the 4,000-page Merck Manual by hand is slow, and general-purpose chatbots
answer from memory, give no citations and can hallucinate.

**Objective.** Build a RAG-based AI solution over a renowned medical manual that:

- **understands** the information-overload problem,
- **applies** AI (LLMs, embeddings and vector search) to streamline decision-making,
- **analyses** the effect on answer quality,
- **evaluates** whether it can standardise care information, and
- **creates** a working prototype.

**Use-case questions** (one per specialty):

| Specialty | Question |
|---|---|
| Critical Care | What is the protocol for managing sepsis in a critical care unit? |
| General Surgery | What are the common symptoms for appendicitis, and can it be cured via medicine? If not, what surgical procedure should be followed to treat it? |
| Dermatology | What are the effective treatments or solutions for addressing sudden patchy hair loss, commonly seen as localized bald spots on the scalp, and what could be the possible causes behind it? |
| Neurology | What treatments are recommended for a person who has sustained a physical injury to brain tissue, resulting in temporary or permanent impairment of brain function? |

**Data.** *The Merck Manual of Diagnosis & Therapy* is a PDF of over 4,000 pages in 23 sections. It covers
disorders, tests, diagnoses and drugs.

---

## 2. Solution overview

The project compares three approaches to the same questions:

| # | Approach | What it does |
|---|---|---|
| 1 | **Base LLM** | Sends the question straight to the LLM |
| 2 | **Prompt engineering** | Adds a "medical research assistant" system prompt |
| 3 | **RAG** | Retrieves the most relevant manual excerpts and answers **only** from them, citing chapter and page |

**Two provider setups, chosen automatically from the key in `.env`:**

| | Groq (default, free tier) | OpenAI (the notebook's setup) |
|---|---|---|
| Chat LLM | `openai/gpt-oss-120b` on Groq (reasoning effort: low) | `gpt-4o-mini` |
| Embeddings | `BAAI/bge-small-en-v1.5`, run locally (free, no key) | `text-embedding-3-small` |
| Key needed | `GROQ_API_KEY` | `OPENAI_API_KEY` |

```mermaid
flowchart LR
    subgraph Ingestion["Ingestion (run once)"]
        A[Merck Manual PDF] -->|PyMuPDF + watermark/footer cleanup| B[4,104 pages]
        B -->|token splitter<br/>500 tok, 50 overlap| C[~8,250 chunks<br/>+ chapter & page]
        C -->|bge-small local<br/>or OpenAI embeddings| D[(Chroma<br/>vector store)]
    end
    subgraph Query["Question answering"]
        Q[Clinician question] -->|embed| R{Top-k<br/>similarity search}
        D --> R
        R -->|3 labelled excerpts| P[Grounded prompt<br/>system + context + question]
        P -->|gpt-oss-120b on Groq<br/>or gpt-4o-mini, temp 0| ANS[Answer + chapter/page citation]
    end
    subgraph Eval["Evaluation"]
        ANS --> J[LLM-as-a-judge]
        J --> S[Groundedness 1-5<br/>Relevance 1-5]
    end
```

Read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for a component-by-component explanation.

---

## 3. Results

LLM-as-a-judge scores from 1 to 5, averaged over the 4 use-case questions.

**This codebase** (Groq `gpt-oss-120b` + local `bge-small` embeddings, fixed evaluation, `python scripts/evaluate.py`):

| Approach | Groundedness | Relevance |
|---|---|---|
| Base LLM | 1.75 | 5.00 |
| Prompt engineering | 1.75 | 5.00 |
| **RAG** | **5.00** | 4.75 |

**Original notebook** (OpenAI `gpt-4o-mini`, judge `gpt-3.5-turbo`):

| Approach | Groundedness | Relevance |
|---|---|---|
| Base LLM | 5.00* | 3.67 |
| Prompt engineering | 4.25 | 4.50 |
| RAG | 4.50 | 5.00 |

\*The notebook's scores came from a flawed evaluation. Scores were read from the second-to-last
character of the judge's reply, so several came out blank, and one answer was scored against the wrong
question. Once fixed, the result is much clearer: answers without retrieval sound right but are **not
supported by the manual** (groundedness 1.75), while RAG answers are fully grounded (5.0). See
[docs/EVALUATION.md](docs/EVALUATION.md) for per-question scores.

**Example RAG answer** (General Surgery, from this codebase):

> **Answer:** Common symptoms of acute appendicitis include epigastric or periumbilical pain that later shifts
> to the right lower quadrant, brief nausea, vomiting, anorexia… and classic signs such as tenderness at
> McBurney's point, Rovsing, psoas and obturator signs. Appendicitis cannot be cured solely with medication;
> … definitive treatment requires surgical removal of the appendix (appendectomy), open or laparoscopic.
>
> **Source:** The Merck Manual of Diagnosis & Therapy, Chapter 11. Acute Abdomen & Surgical Gastroenterology, pages 164–165.

### Key insights

1. **RAG is the only approach grounded in the evidence (5.0 vs 1.75).** Answers without retrieval are fluent and relevant, but not traceable to the manual, which makes them unsafe for clinical decisions.
2. **RAG answers can be traced.** Every answer cites the chapter and printed page, so a clinician can verify it in seconds.
3. **Relevance is limited by retrieval.** The neurology answer (4/5) covered rehabilitation but missed acute brain-injury management, because 2 of the 3 retrieved excerpts were about rehab. Fetching more candidates and re-ranking them would fix this.
4. **Prompt engineering alone adds no evidence.** It changes tone and structure, not where the facts come from.

### Business recommendations

1. Use RAG over authoritative manuals as the default for clinical Q&A. Do not use bare LLMs.
2. Increase chunks to 800–1,000 tokens with 150–200 overlap, and add a **cross-encoder re-ranker** and hybrid (BM25 + vector) retrieval to raise groundedness.
3. Expand evaluation to a 30–50 question benchmark across specialties before any clinical pilot.
4. Make citations mandatory, with chunk IDs, page numbers and document metadata, for traceability and regulatory readiness.
5. Add safety guardrails, fallback responses and PHI-safe logging.
6. Standardise the answer format as **Assessment → Management → Red Flags → Evidence**.
7. Monitor groundedness, relevance, latency and token cost on a dashboard.

---

## 4. Project structure

```
medical-rag-assistant/
├── app.py                    # Streamlit chat UI
├── medical_rag/              # Core package
│   ├── config.py             # Settings (env-overridable) + the 4 use-case questions
│   ├── prompts.py            # System/user prompts for prompt-eng, RAG and the judges
│   ├── llm.py                # Groq/OpenAI chat client; base LLM + prompt-engineered answers
│   ├── ingest.py             # PDF -> chunks -> embeddings -> Chroma
│   ├── retriever.py          # Load store, top-k search, labelled context
│   ├── rag.py                # Retrieve + generate grounded, cited answer
│   └── evaluate.py           # LLM-as-a-judge: groundedness & relevance
├── scripts/
│   ├── ingest.py             # Build the vector store
│   ├── ask.py                # CLI Q&A (rag / compare modes)
│   └── evaluate.py           # Run the 3-way evaluation, save CSV
├── tests/test_pipeline.py    # Offline unit tests (no API key needed)
├── notebooks/                # Original analysis notebook with outputs
├── docs/                     # Architecture & evaluation write-ups
├── data/                     # Put the manual PDF here (git-ignored)
├── .vscode/                  # Run/debug configs for VS Code
├── requirements.txt
└── .env.example
```

---

## 5. Getting started

### Prerequisites
- Python 3.11–3.13
- A **Groq API key** (free at https://console.groq.com/keys), or an OpenAI API key
- The Merck Manual PDF (see [data/README.md](data/README.md))

### Setup (Windows PowerShell)

```powershell
git clone https://github.com/bks1984/medical-rag-assistant.git
cd medical-rag-assistant
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env      # then add GROQ_API_KEY (or OPENAI_API_KEY)
```

macOS and Linux: run `source .venv/bin/activate` and `cp .env.example .env`.

### Build the knowledge base

Copy the PDF to `data/medical_diagnosis_manual.pdf`, then run:

```powershell
python scripts/ingest.py
# try a different chunking strategy:
python scripts/ingest.py --chunk-size 800 --chunk-overlap 150
```

### Ask questions

```powershell
streamlit run app.py                                         # chat UI
python scripts/ask.py "What are the symptoms of appendicitis?"
python scripts/ask.py --mode compare "How is sepsis managed?" # base vs prompt-eng vs RAG
python scripts/ask.py                                        # interactive
```

### Evaluate

```powershell
python scripts/evaluate.py                 # the 4 use-case questions
python scripts/evaluate.py --questions my_questions.json
```

The script writes `results/evaluation_<timestamp>.csv` and prints average scores for each approach.

### Test

```powershell
pytest -q
```

The tests run offline: they mock the LLM and the vector store.

### Configuration

All settings live in `.env`. See [.env.example](.env.example). The provider is detected from the key you set, or you can force it with `LLM_PROVIDER=groq|openai`. Other settings include `CHAT_MODEL`,
`EMBEDDING_PROVIDER`, `EMBEDDING_MODEL`, `CHUNK_SIZE`, `CHUNK_OVERLAP` and `TOP_K`.

With Groq, the first ingest downloads the ~130 MB embedding model and embeds the manual on your CPU. This takes a few minutes and costs nothing.

---

## 6. Tech stack

Python · Groq (`openai/gpt-oss-120b`) or OpenAI (`gpt-4o-mini`) · sentence-transformers (`bge-small-en-v1.5`) · LangChain · ChromaDB · PyMuPDF · tiktoken · Streamlit · pandas · pytest

---

## ⚠️ Disclaimer

This is an educational prototype. It summarises reference text for healthcare professionals and is
**not** a medical device or a substitute for clinical judgement. Do not enter patient-identifiable information.
