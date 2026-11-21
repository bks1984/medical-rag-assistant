# Architecture

This document walks through the pipeline stage by stage. For each stage it names the module, explains the design choice, and links it to the notebook.

## 1. Ingestion: `medical_rag/ingest.py`

| Step | Tool | Setting | Why |
|---|---|---|---|
| Load PDF | `PyMuPDFLoader` | one `Document` per page | Fast, accurate text extraction, and it keeps the page index |
| Clean | `clean_page_text()` | strips the licence watermark and parses the running footer | Removes noise and personal data from every page. Captures the **chapter** and **printed page number** for citations |
| Chunk | `RecursiveCharacterTextSplitter.from_tiktoken_encoder` | `cl100k_base`, 500 tokens, 50 overlap | Measuring in tokens (not characters) matches the model's limits. The overlap keeps sentences that cross a boundary |
| Tag | `tag_chunks()` | `page_number` (printed page), `chapter`, `manual`, `chunk_id` | Used for citations and traceability |
| Embed | `HuggingFaceEmbeddings` (Groq setup) or `OpenAIEmbeddings` | `BAAI/bge-small-en-v1.5`, 384-dim, run locally; or `text-embedding-3-small`, 1,536-dim | Groq has no embedding API, so a small, strong open model runs on the CPU for free. Each provider gets its own Chroma collection because the vector sizes differ |
| Store | `Chroma` (persisted to `chroma_store/`) | batches of 100, 0.3 s pause | Stays under per-request token limits and rate limits |

After cleaning, the Merck Manual's 4,104 content pages yield about **8,257 chunks** with these settings. The notebook's 8,875 included the watermark text. Ingestion runs once. Run `scripts/ingest.py` again to rebuild the store with different settings.

## 2. Retrieval: `medical_rag/retriever.py`

- The persisted store is loaded once and cached.
- `retrieve(question, k=3)` runs a cosine-similarity search, as in the notebook (`search_type="similarity"`, `k=3`).
- `format_context()` labels each excerpt as `[Excerpt n | <manual>, Chapter X. <title>, page N]`, so the LLM can cite real page numbers. The notebook joined chunks with `". "`, which lost where each one came from.

## 3. Generation: `medical_rag/rag.py` and `medical_rag/prompts.py`

- **System prompt (`RAG_SYSTEM`)**: the role is "AI clinical assistant", and it must use **only** the context. The output format is fixed (`Answer:` / `Source:`). When the context doesn't cover the question, it must reply exactly "Sorry, this is out of my knowledge base."
- **User prompt (`RAG_USER_TEMPLATE`)**: `###Context` followed by `###Question`. It is filled with `str.replace`, so braces in the manual text (dosage notation, for example) can't break `str.format`.
- **Model**: `openai/gpt-oss-120b` on Groq (reasoning effort: low), through its OpenAI-compatible endpoint, or `gpt-4o-mini` on OpenAI. Temperature 0 and top-p 0.95 keep answers deterministic and reproducible. The client retries with back-off to cope with Groq's free-tier rate limits, and any `<think>` reasoning block is removed from the output.
- **Short-circuits**: an empty question or zero retrieved chunks returns without calling the LLM.

## 4. Baselines: `medical_rag/llm.py`

- `ask_llm()`: the bare question, with no system prompt (notebook §"Question Answering using LLM").
- `ask_with_prompt_engineering()`: adds a "medical research assistant" system prompt (notebook §"Prompt Engineering").

## 5. Evaluation: `medical_rag/evaluate.py`

An LLM-as-a-judge scores each answer twice, with separate system prompts:

- **Groundedness**: is the answer derived *only* from the retrieved context?
- **Relevance**: does it address all the important aspects of the question, and only those?

The judge reasons step by step, then ends with `groundedness_score: N` or `relevance_score: N`. `parse_score()` reads that line, and falls back to the last 1–5 digit on the final line. All three approaches are judged against the **same** retrieved context, so the comparison is fair.

## 6. Interfaces

| Interface | File | Use |
|---|---|---|
| Streamlit chat | `app.py` | Sample-question buttons, adjustable *k*, an expandable "Sources" panel, and an optional side-by-side baseline comparison |
| CLI | `scripts/ask.py` | One-off or interactive questions. `--mode compare` shows all three approaches |
| Batch evaluation | `scripts/evaluate.py` | Writes a CSV and prints average scores |

## Design decisions

- **The package is separate from the scripts.** All logic lives in `medical_rag/` and can be imported from the notebook, the CLI, the UI and the tests.
- **Heavy imports are lazy.** LangChain, Chroma and OpenAI are imported inside functions, so the unit tests run without a key or a network.
- **All configuration comes from the environment.** Chunk size, *k* and the models are set in `.env`, so experiments need no code changes.
- **The copyrighted PDF is never committed.** `data/*.pdf` is git-ignored.
