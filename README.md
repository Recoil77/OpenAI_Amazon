# Amazonia DeepScan

**AI-powered pipeline for digitizing, analyzing, and exploring lost settlements and historic documents from the Amazon.**

---

## Project Overview

Amazonia DeepScan is a modular research platform for discovering and analyzing historical settlements, events, and artifacts in the Amazon basin. It combines OCR, LLM-based text cleaning, translation, semantic chunking, entity extraction, vector search, and multi-step reasoning in a fully transparent and reproducible pipeline. Designed for historians, archaeologists, and data scientists, it allows automated exploration of large-scale archival data to surface evidence and generate new research leads.

---

## Main Features

* Modular pipeline: OCR, text cleaning, translation, entity extraction, semantic search
* LLM-based research agent: multi-step reasoning, variant handling, cross-source evidence
* Vector-based and entity-based search; hybrid queries; web and general knowledge integration
* All code and pipelines are under user control—no black-box services

---

## Project Structure

```
app/               # FastAPI application with all endpoints
agent.py           # CLI reasoning agent using the API
agent_runs/        # JSON logs produced by the agent
docs_to_vectors/   # Helper scripts for converting raw docs to vectors
gateway/           # Thin client for calls to the OpenAI Gateway
utils/             # Small utility functions
README.md          # This file
```

---

## Requirements

Install all required packages with:

```bash
python -m venv venv
source venv/bin/activate
pip install fastapi uvicorn httpx asyncpg pydantic tiktoken structlog torch transformers python-dotenv aiohttp aiofiles tqdm Pillow
```

To launch API endpoints:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
uvicorn app.agent_endpoints:app --host 0.0.0.0 --port 8100
```

---

## API Endpoints

The following FastAPI endpoints power the core pipeline:

### app/main.py

* `POST /ocr_main_text` — OCR an uploaded image, returning extracted text as JSON.
* `POST /clean_ocr_extended` — Clean and optionally translate OCR fragments; returns cleaned text and quality score.
* `POST /generate_metadata` — Extract semantic entities from a text chunk for downstream search.
* `POST /rerank_bge` — Use a BGE cross‑encoder to rerank candidate text blocks.
* `POST /rerank_semantic_v5` — Apply an LLM-based semantic reranker to filter and order blocks.
* `POST /extract_facts` — Extract concise supporting facts from a set of chunks given a question.

### app/agent\_endpoints.py

* `POST /llm_reasoning` — Run a reasoning agent iteration; returns structured JSON output.
* `POST /web_search` — Web search (returns Evidence objects).
* `POST /general_knowledge` — Provide a concise factual answer as Evidence.
* `POST /entity_search` — Search stored metadata for given entities; returns occurrence counts.
* `POST /entity_hybrid` — Entity-based search with BGE + LLM reranking (returns evidence snippets).
* `POST /chunk_summary` — Summarize a text chunk (factual content).
* `POST /vector_search` — Multi-step vector search pipeline with reranking and fact extraction.
* `POST /vector_search_v2` — Updated search pipeline with thresholds.

These endpoints implement OCR, text cleaning, metadata generation, search, reranking, fact extraction, and multi-step reasoning for the application.

---

## Example Usage

**Run agent with a research question:**

```bash
python agent.py
```

The agent iteratively queries semantic and entity endpoints, accumulates evidence, and outputs a detailed answer trace. Example question:

> Find all clues related to the geographical location of the object or place named Bararoá.

---

## Logging & Output

All results and reasoning traces from the agent are automatically saved as JSON files in the `agent_runs/` directory.

Each run produces a file:

```
agent_run_<timestamp>.json
```

This log includes all API calls, reasoning steps, accumulated evidence, and final answers for full traceability and reproducibility.

---

## Data & Sources

* All processed books and documents are listed in the [Project Books Register (Google Sheet)](https://docs.google.com/spreadsheets/d/1cnPDxvQQ_Lr0sy1_2w06rreumNOKLq_b-U5Sk6Ja3Mw)
* You can add new sources by editing this shared document

---

## Limitations

* Answers may vary in precision depending on prompt and evidence
* Stability and prompt tuning are in progress
* OCR and translation errors are possible; pipeline is robust but not perfect

---

## Contact

* Anton Sibilev — [Kaggle project page](https://www.kaggle.com/code/antonsibilev/amazonia-deepsearch)
* Telegram: @anton\_sibilev
