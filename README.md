# Amazon Historic Text Bridge

## Project Overview

This repository contains a production ready pipeline for processing and searching digitised historical texts from the Amazon region.  The project combines OCR, text cleaning, translation, metadata extraction and semantic search in a single FastAPI service backed by PostgreSQL and OpenAI models.  It is designed to help researchers explore scattered archives and automatically discover relevant facts.

## Folder Structure

```
app/               # FastAPI application with all endpoints
agent.py           # CLI reasoning agent using the API
agent_runs/        # JSON logs produced by the agent
docs_to_vectors/   # Helper scripts for converting raw docs to vectors
gateway/           # Thin client for calls to the OpenAI Gateway
utils/             # Small utility functions
```

The `docs/` folder (ignored in git) stores intermediate OCR results and processed chunks when running the pipeline.

## Setup

1. **Python environment** – install Python 3.10+ and create a virtual environment.
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install fastapi uvicorn httpx asyncpg pydantic tiktoken structlog torch transformers python-dotenv
   ```
2. **PostgreSQL** – create a database and set the `DATABASE_URL` environment variable.  Example:
   ```bash
   export DATABASE_URL=postgresql://user:password@localhost:5432/amazon_db
   ```
3. **Server address** – the agent and data-prep scripts read the API host from `SERVER_ADDRESS`:
   ```bash
   export SERVER_ADDRESS=127.0.0.1
   ```
4. **Run the API** – start the FastAPI service using Uvicorn:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8100
   ```

## Main Endpoints

* `/ocr_main_text` – OCR a page image and return JSON text.
* `/clean_ocr_extended` – clean OCR output and optionally translate.
* `/generate_metadata` – extract entities for semantic search.
* `/vector_search_v2` – similarity search across stored document chunks.
* `/rerank_bge` – BGE cross‑encoder reranking of candidate chunks.
* `/rerank_semantic_v5` – final LLM semantic rerank.
* `/extract_facts` – pull structured facts from text fragments.
* `/general_knowledge` – answer common factual questions from the LLM.
* `/web_search` – optional external web search.

## Example Usage

1. **Simple vector search**
   ```bash
   curl -X POST http://localhost:8100/vector_search_v2 \
        -H "Content-Type: application/json" \
        -d '{"query": "Jesuit missions", "k": 5}'
   ```
2. **Run the reasoning agent**
   ```bash
   python agent.py
   ```
   The agent will query the API iteratively and store its reasoning trace in `agent_runs/`.

Further examples and a live demo are available in the [Kaggle notebook](https://www.kaggle.com/code/antonsibilev/amazon-data-bridge-v2).

## Limitations

* Some images may fail OCR or translation, so coverage is not always 100 %.
* Web search integration is planned but not yet implemented.

## Roadmap

Work continues on a controller LLM to orchestrate deeper reasoning, improved fact extraction and dynamic prompt generation for historical research.

## Contacts

- **Anton Sibilev** – [Kaggle project page](https://www.kaggle.com/code/antonsibilev/amazon-data-bridge-v2)
- Telegram: `@anton_sibilev`
- OpenAI & ChatGPT (co‑authors)
